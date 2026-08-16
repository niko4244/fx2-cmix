#!/usr/bin/env bash
# Deliberate-tamper self-check: proves the two reconstruction guardrails
# actually FIRE on a broken reconstruction instead of silently passing.
#
# Round 1 — new_sqsum: the final horizontal reduce's half-swap shuffle is
#   changed 0x1 -> 0x0.
#   * redtest: "sqsum N=201" (currently BIT-EQUAL and NOT in the registry)
#     flips to a plain unregistered DIFFER, so redtest exits non-zero.
#   * compare_fp_codegen: the reduce-tail marker (vshufpd-1 vaddps
#     vmovshdup vaddss) can no longer match the standalone stream (the 0x0
#     shuffle removes vshufpd-1 under any lowering) -> guaranteed WARN.
#
# Round 2 — new_alpha: the t<LIMIT rsqrt+Newton block is replaced with the
#   plain 1/sqrtf expression, making new_alpha bit-identical to old_alpha.
#   * redtest: the REGISTERED "alpha t=1"/"alpha t=100" cases flip from
#     EXPECTED-DIFFER to BIT-EQUAL, which classify() labels
#     "BIT-EQUAL (STALE REGISTRY)" -> redtest exits non-zero.
#   * compare_fp_codegen: vsqrtss appears standalone but absent from the
#     production BackwardPass(NeuronLayer&) body, and the rsqrt-newton
#     marker no longer matches -> WARN.
#
# Each round works on a COPY of the harness in a temp dir — the repo tree
# is never modified. A round fails (exit non-zero) if the tamper does not
# apply cleanly, if redtest does not fire, or if compare_fp_codegen does
# not print the expected WARNs, so the guardrails cannot silently rot.
#
# The clean-baseline side is established by the job steps that run before
# this one (the harness must exit 0 and the compare step must report the
# tree's normal MATCH count); this step only proves the DELTAS fire.
#
# Usage: tamper_selfcheck.sh <redtest.cpp> <full.dis>
set -uo pipefail

SRC="${1:?usage: tamper_selfcheck.sh <redtest.cpp> <full.dis>}"
FULL="${2:?usage: tamper_selfcheck.sh <redtest.cpp> <full.dis>}"
[ -f "$SRC" ] || { echo "FATAL: $SRC not found"; exit 1; }
[ -f "$FULL" ] || { echo "FATAL: $FULL not found (run the production disasm step first)"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FATAL: $*"; exit 1; }

# Flags must match the harness step in ci.yml exactly.
CXX=(clang++-17 -O3 -march=core-avx2 -ffp-model=fast -fno-inline -std=c++17 -DUPDATE_LIMIT=3000)

# run_round <name> <src> <expected-redtest-grep> <expected-cmp-grep...>
# Builds a tampered harness, runs redtest (must exit non-zero with the
# expected verdict line), objdumps it, and runs compare_fp_codegen.sh
# (must print the expected WARN lines).
run_round() {
  local name="$1" src="$2" rt_grep="$3"; shift 3
  echo "----- round $name: $src -----"

  [ -f "$src" ] || fail "round $name: tampered source missing"
  local bin="$TMP/redtest_$name" out="$TMP/out_$name" dis="$TMP/dis_$name" cmp="$TMP/cmp_$name"

  "${CXX[@]}" "$src" -o "$bin" || fail "round $name: tampered harness did not compile"

  "$bin" > "$out" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "----- round $name: redtest output (unexpected pass) -----"
    cat "$out"
    fail "round $name: redtest did NOT fire on the tamper (exit 0) — registry lost its teeth"
  fi
  grep -q "$rt_grep" "$out" || {
    echo "----- round $name: redtest output -----"
    cat "$out"
    fail "round $name: redtest exited $rc but printed no \"$rt_grep\""
  }
  echo "redtest fired: exit=$rc, verdict \"$rt_grep\" present"
  grep -E "$rt_grep" "$out"

  objdump -d --no-show-raw-insn -M intel -C "$bin" > "$dis" || fail "round $name: objdump failed"
  bash "$HERE/compare_fp_codegen.sh" "$dis" "$FULL" > "$cmp" 2>&1
  for pat in "$@"; do
    grep -qE "$pat" "$cmp" || {
      echo "----- round $name: compare_fp_codegen output -----"
      grep -E "^(WARN|FATAL|=====)" "$cmp"
      fail "round $name: compare_fp_codegen did not warn on \"$pat\" — codegen check lost its teeth"
    }
  done
  echo "compare_fp_codegen fired: $(grep -cE '^WARN' "$cmp") WARN line(s), including:"
  for pat in "$@"; do grep -E "$pat" "$cmp"; done
}

echo "===== Tamper self-check: prove the oracles fire on a broken reconstruction ====="

# ---- round 1: new_sqsum reduce tail 0x1 -> 0x0 -----------------------
# The vshufpd 0x1 line exists in BOTH new_sqsum and new_sftsum, so scope
# the edit to new_sqsum only (from its signature to the next static fn).
# NOTE: patterns avoid escaped parens (awk treats \( as a group open).
awk '
  /^static float new_sqsum/ { infn = 1 }
  /^static / && infn && $0 !~ /^static float new_sqsum/ { infn = 0 }
  infn && /shuffle_pd/ && /0x1/ { sub(/0x1/, "0x0") }
  { print }
' "$SRC" > "$TMP/redtest_sqsum.cpp"
# The shuffle is nested inside _mm_add_ps, so the source line is
# "xv = _mm_add_ps(xv, _mm_shuffle_pd(xv, xv, 0x1));" — match the
# shuffle_pd( literal, not the _mm_add_ps( prefix.
grep -c "_mm_shuffle_pd(xv, xv, 0x0));" "$TMP/redtest_sqsum.cpp" | grep -qx 1 \
  || fail "round 1: tamper did not apply (new_sqsum shuffle line not found — source moved?)"
grep -c "_mm_shuffle_pd(xv, xv, 0x1));" "$TMP/redtest_sqsum.cpp" | grep -qx 1 \
  || fail "round 1: tamper over-applied (new_sftsum shuffle line was also changed)"

# The multiset WARN is lowering-dependent (clang may emit vmovddup
# instead of vshufpd-0 for the 0x0 shuffle; vmovddup is not in the FP
# filter), but the reduce-tail marker REQUIRES vshufpd-1, so its absence
# is a guaranteed WARN under any lowering.
run_round "1-sqsum" "$TMP/redtest_sqsum.cpp" "^sqsum N=201.*DIFFER" \
  "WARN +new_sqsum +marker reduce-tail"

# ---- round 2: new_alpha rsqrt+Newton -> plain 1/sqrtf -----------------
# Replace the whole t<LIMIT rsqrt+Newton block with the plain expression
# (textually identical to old_alpha), anchored on the unique fmaf x-line
# (new_adam uses "xv = __builtin_fmaf(...)", so the anchor is unambiguous).
# The rsqrt+Newton block sits inside an extra "{ ... }" pragma scope
# whose opening brace must stay balanced: emit the plain expression AND
# the scope's closing brace (the x-line anchor is unambiguous — new_adam
# uses "xv = __builtin_fmaf(...)").
awk '
  /const float x = __builtin_fmaf/ { intamper = 1 }
  intamper && /^    }$/ {
    print "    alpha = lr * 0.1f / sqrtf(5e-5f * t + 1.0f);";
    print "    }";
    intamper = 0;
    next;
  }
  intamper { next }
  { print }
' "$SRC" > "$TMP/redtest_alpha.cpp"
grep -q "const float x = __builtin_fmaf(5e-5f, t, 1.0f);" "$TMP/redtest_alpha.cpp" \
  && fail "round 2: tamper did not apply (new_alpha block not found — source moved?)"
grep -q "alpha = lr \* 0.1f / sqrtf(5e-5f \* t + 1.0f);" "$TMP/redtest_alpha.cpp" \
  || fail "round 2: tamper replacement missing"

run_round "2-alpha" "$TMP/redtest_alpha.cpp" "BIT-EQUAL (STALE REGISTRY)" \
  "WARN +new_alpha +op vsqrtss" \
  "WARN +new_alpha +marker rsqrt-newton"

echo "===== TAMPER SELF-CHECK PASS: both oracles fire on a broken reconstruction ====="

#!/usr/bin/env bash
# Compare each new_* reconstruction's standalone FP instruction stream
# (from the -fno-inline harness disasm) against the production binary's
# in-context codegen for the LSTM function(s) that inline it.
#
# Why: inlining into a large function is the one place clang's fast-math
# lowering can still differ from the standalone harness (re-pairing,
# contraction, scalarization, different vector width). The identity job is
# the ground truth for byte equality, but this step gives an immediate,
# per-site signal: a new_* FP opcode that is absent from the production
# body means the reconstruction compiled differently in context than it
# does standalone.
#
# WARN-only by design: a divergence here does not fail the job (register-
# permutation / accumulator-slot differences are invisible at this level
# and are handled by the volatile pins + identity job). The step DOES fail
# if the comparison machinery itself breaks (missing files, a new_*
# function or production symbol that cannot be located), so the oracle
# cannot silently rot.
#
# Usage: compare_fp_codegen.sh <redtest.dis> <cmix.dis>
set -uo pipefail

HARNESS="${1:?usage: compare_fp_codegen.sh <redtest.dis> <cmix.dis>}"
PROD="${2:?usage: compare_fp_codegen.sh <redtest.dis> <cmix.dis>}"
[ -f "$HARNESS" ] || { echo "FATAL: $HARNESS not found"; exit 1; }
[ -f "$PROD" ] || { echo "FATAL: $PROD not found"; exit 1; }

awk -v harness="$HARNESS" -v prod="$PROD" '
BEGIN {
  nf = split("new_matvec new_proj new_sqsum new_sftsum new_chain1 new_chain2 new_chain3 new_chain4 new_alpha new_adam", fns, " ");
  # new_* function -> demangled production symbol prefixes that inline the
  # site (DEBUG=1 build keeps symbols). Multiple targets are unioned; a
  # harness opcode must be present in at least one of them.
  target["new_matvec"]  = "LstmLayer::ForwardPass(NeuronLayer&"
  target["new_proj"]    = "Lstm::Predict(unsigned int)\x1fLstm::Perceive(unsigned int)"
  target["new_sqsum"]   = "LstmLayer::ForwardPass(NeuronLayer&\x1fLstmLayer::BackwardPass("
  target["new_sftsum"]  = "Lstm::Predict(unsigned int)"
  target["new_chain1"]  = "LstmLayer::BackwardPass(std::valarray"
  target["new_chain2"]  = "LstmLayer::BackwardPass(std::valarray"
  target["new_chain3"]  = "LstmLayer::BackwardPass(std::valarray"
  target["new_chain4"]  = "LstmLayer::BackwardPass(std::valarray"
  target["new_alpha"]   = "LstmLayer::BackwardPass(NeuronLayer&"
  target["new_adam"]    = "LstmLayer::BackwardPass(NeuronLayer&"

  # Documented harness artifacts that ALWAYS differ from production and so
  # are not regression signals: printed as NOTE, not counted as WARN
  # (same philosophy as the redtest kExpectedDiff registry). A new WARN on
  # any opcode not listed here is the real signal.
  expect["new_alpha vucomiss"] = "harness compares float t < LIMIT; production compares integer update_steps_, so no FP compare exists there";
  expect["new_adam vucomiss"]  = "same float-vs-int t comparison artifact";
  expect["new_adam vsubsd"]    = "harness keeps pow(beta1, UPDATE_LIMIT) in double precision; production folds it to the FOLD constant";
  expect["new_adam vcvtsd2ss"] = "same folded-pow artifact";
}

# Normalize one FP instruction to "mnemonic" (shuffle imms kept: they are
# structural — 0x1b reversed blocks, 0x4e cross-lane permutes, 0x1 half-
# swaps). Returns "" for non-FP instructions.
function norm(m, rest,   a, i, n, imm) {
  if (m ~ /^v?(fmadd|fmsub|fnmadd|fnmsub)(132|213|231)?(ps|pd|ss|sd)$/ ||
      m ~ /^v?(fmaddsub|fmsubadd|fnmaddsub|fnmsubadd)(132|213|231)?(ps|pd)$/ ||
      m ~ /^v?(mul|add|sub|div|sqrt|rsqrt|rcp|max|min)(ps|pd|ss|sd)$/ ||
      m ~ /^v?(movshdup|movsldup)(ps|pd)?$/ ||
      m ~ /^v?(shufps|shufpd|permpd|perm2f128|blendps|blendpd)$/ ||
      m ~ /^v?(unpck[hl](ps|pd)|round(ps|pd|ss|sd))$/ ||
      m ~ /^v?cvt[a-z0-9]*(ps|pd|ss|sd)$/ ||
      m ~ /^v?(comiss|comisd|ucomiss|ucomisd|movmskps)$/) {
    if (m ~ /^(v)?(shufps|shufpd|permpd|perm2f128|blendps|blendpd|roundps|roundpd)$/) {
      n = split(rest, a, " ");
      imm = "";
      for (i = 1; i <= n; i++)
        if (a[i] ~ /^0x[0-9a-f]+$/) imm = a[i];
      if (imm != "") return m "-" imm;
    }
    return m;
  }
  return "";
}

FNR == 1 { inprod = (FILENAME == prod) }

{
  # Greedy name match: demangled C++ names contain a closing angle
  # bracket inside template args (std::valarray<float>), so a [^>]*
  # pattern would miss those symbols entirely.
  if ($0 ~ /^[0-9a-f]+ <.*>:$/) {
    name = $0;
    sub(/^[0-9a-f]+ </, "", name);
    sub(/>:$/, "", name);
    if (inprod) {
      # Which reconstructions does this production symbol inline? Prefix
      # match against each target list; union when several apply.
      curfns = "";
      for (k = 1; k <= nf; k++) {
        fn = fns[k];
        nb = split(target[fn], prefixes, "\x1f");
        for (q = 1; q <= nb; q++)
          if (index(name, prefixes[q]) == 1) {
            curfns = curfns " " fn;
            prod_found[fn] = 1;
            break;
          }
      }
    } else {
      if (name ~ /^new_/) { curfn = name; sub(/\(.*/, "", curfn); }
      else curfn = "";
    }
    next;
  }
  if ($0 !~ /^[ \t]+[0-9a-f]+:[ \t]/) next;
  line = $0;
  sub(/^[ \t]+[0-9a-f]+:[ \t]+/, "", line);
  m = line; sub(/ .*/, "", m);
  rest = line; sub(/^[^ ]+[ \t]+/, "", rest);
  op = norm(m, rest);
  if (op == "") next;
  if (!inprod && curfn != "") h[curfn, op]++;
  if (inprod && curfns != "") {
    n = split(curfns, arr, " ");
    for (i = 1; i <= n; i++) p[arr[i], op]++;
  }
}

END {
  printf "===== FP codegen: new_* standalone vs production in-context =====\n";
  nwarn = 0; nmatch = 0; nerr = 0; ndoc = 0;
  for (k = 1; k <= nf; k++) {
    fn = fns[k];
    if (!prod_found[fn]) {
      printf "FATAL %-12s no production symbol matched %s\n", fn, target[fn];
      nerr++; continue;
    }
    nops = 0; miss = 0;
    for (idx in h) {
      split(idx, kv, SUBSEP);
      if (kv[1] != fn) continue;
      nops++;
      if (p[fn, kv[2]] == 0) {
        if (expect[fn " " kv[2]] != "") {
          ndoc++;
          printf "NOTE  %-12s op %-16s documented artifact: %s\n",
                 fn, kv[2], expect[fn " " kv[2]];
        } else {
          miss++;
          t = target[fn]; gsub(/\x1f/, "; ", t);
          printf "WARN  %-12s op %-16s present standalone, absent from production (%s)\n",
                 fn, kv[2], t;
        }
      }
    }
    if (nops == 0) {
      printf "FATAL %-12s no FP ops found in harness disasm (symbol missing?)\n", fn;
      nerr++; continue;
    }
    if (miss == 0) {
      nmatch++;
      printf "MATCH %-12s %d FP ops, all present in production\n", fn, nops;
    } else {
      nwarn += miss;
    }
  }
  printf "===== %d matched, %d warnings, %d documented artifacts, %d fatal =====\n",
         nmatch, nwarn, ndoc, nerr;
  exit (nerr ? 1 : 0);
}
' "$HARNESS" "$PROD"

# Changelog

All notable changes to fx2-cmix are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to the Hutter Prize rules of the [Prize](http://prize.hutter1.net/).

## [Unreleased] — dev branch

### Changed
- **LSTM matvec reconstructed BY CONSTRUCTION** (`lstm-layer.hpp`
  `ForwardPass(NeuronLayer)`): replaced the valarray dot-product loop with
  an explicit intrinsic sequence that reproduces clang-17's exact FP
  operation tree. Derived from the disassembly of the production Linux
  binary (new manual `disasm` job): 4×8-lane FMA accumulators over
  32-element blocks, the `weights_[i][input_symbol]` seed fused into
  accumulator lane 0, the horizontal reduce `(y1+y0)→(y3+y2)→t1+t0`,
  `low128+high128`, the 64-bit-half-swap `vshufpd`+`add`, `vmovshdup`+
  `vaddss` (i.e. `(s0+s2)+(s1+s3)`), then a scalar FMA tail. The tail
  runs under `reassociate(off)` so fast-math cannot re-tree it.
  Semantics of the emitted reduce were pinned empirically by
  `tools/redtest.cpp` (new): clang's fast-math tree ≠ fixed-order FMA,
  and a replica of the emitted tail matches clang's tree (the
  `vshufpd x,x,1` with identical operands is a half-swap, not a copy).
  Byte-exactness is gated by the identity job comparing against
  `a45952b` (the pre-rewrite tree).

  Two fast-math hazards were found and fixed during the gate:
  (1) clang re-paired the horizontal reduce's partial-sum tree under
  `-ffp-model=fast` (`(y3+y0)+(y2+y1)` vs the emitted `(y1+y0)+(y3+y2)`)
  — the pairing lives in register allocation, so it is invisible to
  instruction-level diffs and survived `#pragma clang fp reassociate(off)`
  in the real function's context (the standalone harness held the
  pairing, which is why it reported a false BIT-EQUAL). Fixed by forcing
  the two partial sums through a volatile round-trip so no pass can
  regroup them; (2) the scalar FMA tail runs under `reassociate(off)`
  so fast-math cannot re-tree it. **Identity gate PASSED**: the
  reconstructed LSTM produces byte-identical compressed output to
  `a45952b` (both 180642 bytes on the CI corpus).

  **Speed measured** (same-run A/B via the benchmark override vs
  `a45952b`): **-0.5%** (HEAD 16244 ms vs 16333 ms, reps 16244/16502/
  16778 vs 16515/16333/16803, spreads 3.3%/2.9%, no reliability flag) —
  neutral within noise: the compact reconstruction roughly breaks even
  with the 10-copy-unrolled valarray loop. The durable win is not speed
  but determinism: every FP reduction tree in the LSTM matvec is now
  explicit (intrinsics + volatile-pinned partial sums + pragma'd tail),
  so no future compiler/flag change can silently alter the compressed
  output. `tools/redtest.cpp` now reports BIT-EQUAL for every probed
  length (657, 457 [the real gate length], 201, 128, 40, 33, 32, 31, 17,
  all-ones).

- **Harness is now an instruction-level oracle**: `tools/redtest.cpp`
  builds with `-fno-inline` so every `old_*`/`new_*` function compiles
  standalone — no inliner-driven FP lowering can change between runs or
  between harness and a future edit. The disasm job objdumps the whole
  harness, so each reduction pair (matvec, projection, sqsum, softmax,
  gate chains, Adam) is verifiable at the instruction level, and dumps
  the full `.rodata` of both HEAD and the identity baseline (constants
  are materialized as immediates in code, but the 0.5f tables are
  confirmed identical across builds). Baseline fetch fixed to use the
  full 40-char SHA (GitHub rejects abbreviated SHAs as refs).

- **Harness expected-failure registry**: `tools/redtest.cpp` now
  distinguishes documented proxy artifacts from genuine regressions. The
  standalone `-fno-inline` harness compiles every `old_*` function
  out-of-line, but production inlines the same math into large functions,
  so a few cases legitimately DIFFER in the harness even though production
  is byte-identical: `sqsum N=200`/`N=33` (standalone `(xv*xv).sum()`
  lowers to a different tree than the inlined `SumRevProduct` at
  `num_cells_=200`) and `alpha`/`adam t<LIMIT` (standalone `sqrt()`
  lowers to `vsqrtss+vdivss`, production inlines rsqrt+Newton), plus the
  `t>=LIMIT` folded Adam w-path proxy gap. Those cases are registered in
  `kExpectedDiff` with reasons and print `EXPECTED-DIFFER`; the harness
  exits non-zero on any *unregistered* DIFFER or a registered case that
  flips BIT-EQUAL (stale entry), so the disasm job fails on a genuine
  regression instead of relabeling it. Production equivalence remains the
  identity job's authority.

- **Standalone-vs-in-context FP codegen cross-check**: new `tools/compare_fp_codegen.sh` (run as a step in the disasm job) extracts each `new_*` reconstruction's FP instruction stream from the `-fno-inline` harness disasm and compares it, opcode by opcode, against the production binary's in-context bodies (mapped by symbol: gate matvec/ivar to `LstmLayer::ForwardPass(NeuronLayer&…)`, projection/softmax to `Lstm::Predict`/`Lstm::Perceive`, gate-error chains to the outer `LstmLayer::BackwardPass(…)`, Adam to `LstmLayer::BackwardPass(NeuronLayer&…)`). Divergence prints `WARN` (warn-only: register-permutation differences are invisible at this level and are covered by the volatile pins + identity job); the step fails only if the machinery breaks. Four harness artifacts are codified as documented `NOTE`s (float-vs-int `t` comparison `vucomiss`, and the double-precision `pow` in the folded Adam path that production constant-folds).

### Added
- GitHub Actions CI (`.github/workflows/ci.yml`):
  - **Output identity (HEAD vs parent)** job: every push must produce
    byte-identical compressed output to its parent commit (same platform,
    same flags). Guards all "safe speedup" work — an optimization that
    changes the compressed bytes fails CI.
  - **gprof profile job (manual only)**: `workflow_dispatch`-triggered,
    never runs on normal pushes. Builds with `PROFILE=1` and reports the
    top self-time functions on `prof_input/input2`, so "safe speedup" work
    can re-profile the current tree on demand. The other four jobs are
    gated out of `workflow_dispatch` runs, so a manual profile dispatches
    only the profile job.
  - **Profiling caveat documented**: cross-run comparisons on shared GitHub
    runners are unreliable — a re-profile of *unchanged* code showed the
    LSTM dropping 66s → 37s (runner variance), and per-call noise on
    unchanged functions spans ±25-59% (e.g. `Lstm::Perceive` -50%, `E1::get`
    +59% in the same pair of runs; round-trip wall-clock varied 117s vs
    125s for byte-identical code). Speedups are judged by mechanism,
    within-run share, and byte-identity; exact percentages need a
    same-machine A/B (e.g. `verify_enwik9.sh` on a fixed box).
  - **CI bug fixed**: the output-identity "Determine comparison baseline"
    step died under `set -euo pipefail` whenever a commit message had no
    `Identity-baseline:` line (`grep` exits 1 on no match). Masked by
    earlier commits all carrying the override; the `always_inline`
    experiment commit (no override) exposed it. Fixed with `|| true` on
    the grep pipeline — commits without an override now correctly default
    to `HEAD~1` again.
  - **Wall-clock benchmark job (new)**: same-run A/B on every push — builds
    HEAD and its parent in one job and times both interleaved on the same
    runner (`-n prof_input/input`, **min of 3 reps each**), reporting the
    delta, each binary's per-rep spread (max−min, as % of the min — how
    noisy that particular run was), and uploading a `benchmark-report`
    artifact. Cross-run totals on shared runners are noise (117s vs 125s
    for byte-identical code), but an interleaved same-job comparison
    cancels drift, giving speedup/regression work a stable measurement
    channel. Informational — the output-identity job remains the
    correctness gate.
  - **`Benchmark-baseline: <sha>` commit-message override** (benchmark
    job): mirrors the `Identity-baseline` mechanism — with the override,
    the same-run A/B compares HEAD against an arbitrary baseline commit
    instead of the default `HEAD~1`, so one run can measure a whole
    accumulated change set (e.g. all safe speedups since a pre-optimization
    tree) against a known-good commit in a single interleaved A/B.
  - **Benchmark channel tightened to min-of-3 + per-rep spread**
    (`6ab23ea`): three interleaved reps per binary (HEAD, baseline, HEAD,
    baseline, HEAD, baseline) with min-of-3 reported, and each line shows
    the per-rep spread (max−min, as % of the min) so a run states how
    noisy it was. If either binary's own reps scatter >5%, the report
    flags the delta as not reliable below ~5% — the channel now
    self-audits instead of silently presenting a noisy number as truth.
    Revalidation on identical code (CHANGELOG-only pushes): the first
    probe measured **-1.2%** with a flagged 7.4% HEAD spread (delta not
    trustworthy), and the second, clean probe (0.5%/0.6% spreads)
    measured **-0.7%**. Clean min-of-3 runs now resolve deltas to roughly
    ±0.5-1% — below the old ±1.5% floor — and noisy runs say so instead
    of presenting a misleading number.
  - **Benchmark default path validated**: the `1a792fa` push exercised the
    no-override fallback end-to-end (no grep-under-`set -euo pipefail`
    trap, unlike the identity job's pre-fix bug) and measured **-0.6%**
    (15434 vs 15525 ms) between HEAD and its code-identical parent — the
    channel's own noise floor on identical code, consistent with the
    earlier +1.5% observation.

### Changed
- **Third gprof pass on the current tree** (`-n` on `prof_input/input2`,
  manual profile job, run `31894478917`). Re-ranking the remaining
  non-LSTM targets after the miss-path and GetContextData work settled:
  - `Mixer::Mix` **12.0%** (29.2s, 178.7M calls) — still the single biggest
    non-LSTM, but its self time is the weight dot products + squash, i.e.
    the FP reductions that are policy-excluded (reassociation would change
    bytes; the LSTM rewrite proved the trap). No safe slice identified.
  - `Predictor::Predict` **7.1%** self / **23.2%** with children — the
    per-bit driver; its subtree (Mix + all models) is the largest after the
    LSTM. Self time includes the banked Logit inlining; restructuring the
    dispatch order is byte-unsafe, so not a target.
  - `fxcmv1::ContextMap2::mix3` **3.7%** (311M calls — the codebase's
    highest call count), plus `ContextMap1::mix3` 1.6% and
    `ContextMap::mix3` 1.3% → the **mix3 family ≈ 6.6%** (whole ContextMap
    family ≈ 10%). All integer leaves (StateMap update + table lookups +
    setter writes, zero child calls) — the largest *safe-category* (no FP
    reductions) aggregate, though the always_inline experiment showed this
    region is i-cache-sensitive, so any change needs the same-run
    benchmark.
  - `Mixer1::p1` **2.7%** + `Mixer1::update` 0.6% — integer leaf, second-
    stage mixer; safe-category but small.
  - `update1` 1.3% + `SSE_sh::M_T1::M_Estimate` 1.2% — SSE chain,
    integer, leaf-ish.
  - PPMD (~2.2% across `ConvertSQ`, `processSymbol2_T`, `CreateSuccessors`)
    — memory-bound on its heap structures; the remap fix was about the
    crash, not speed.
  - **Settled, not targets**: `E1/E/E::get` ≈ 9.3% (out-of-line wins,
    miss path tightened, both inlining directions measured and rejected),
    `Mixer::Perceive` 3.9% (rewrite landed), `GetContextData` 2.2%
    (slot-map landed — held at ~2.2% vs 3.2% pre-change). LSTM ≈ 41%
    remains excluded by policy.
  - **Bottom line**: the safe-speedup well is largely tapped. Above the
    settled/optimized functions, the remaining surface is `Mix` (blocked
    by FP policy) plus a long tail of 1-4% integer leaves whose expected
    yield is small and i-cache-risky. The one big lever left is the LSTM
    (~41%), still blocked unless a bit-exact restructure is found.
- **Second gprof pass on the current tree** (`-n` on `prof_input/input2`, via
  the new manual profile job). LSTM still dominates (~55%, its reductions
  are excluded from optimization work by policy — they cannot be restructured
  byte-identically). Next non-LSTM targets and what was done:
  - `Mixer::Perceive` (~2.6% + hidden allocation/copy cost, 178.7M calls):
    the valarray expressions `weights -= update * inputs_` allocate
    temporaries and make a second pass, and the read-modify-write loop
    cannot auto-vectorize because clang can't prove `weights` and `inputs`
    don't alias. Rewritten as direct per-element updates with
    `__restrict__` pointers and `#pragma clang fp contract(off)` — same
    two-rounding arithmetic as the valarray version (no FMA fusion), so
    compressed output is byte-identical, but now single-pass, allocation-
    free, and vectorizable. Before/after gprof (same input): per-call cost
    42.7ns → 33.5ns (-21%).
  - `fxcmv1::E1::get` (~4.7%, 102M calls): two experiments, both reverted.
    (1) Dropping `noinline` was a no-op — the call count stayed identical
    (clang's inliner declined the probe loop + memset at the hot call
    sites). (2) `__attribute__((always_inline))` **made it worse**: the
    byte-context path (E1::get + its three `ContextMap*::mix` callers) went
    from ~15.6s to ~23.1s (+48%) and the whole profile drifted up 7-56%
    across untouched functions — classic i-cache damage in this
    i-cache-sensitive codebase. The out-of-line version wins; both changes
    reverted (final state = original `noinline`).
  - `Mixer::GetContextData` (2.5-3.2%, 178.7M calls): the emhash6 map held
    full `ContextData` objects as values, so every probe touched scattered
    ~2KB buckets across a ~20MB working set. The map now stores only
    context → slot indices (~16B buckets, cache-resident) with the weight
    vectors in a parallel `std::vector` touched only on a hit — a ~100×
    reduction of the probe working set. Identical lookup semantics and the
    same 10000-context fallback threshold (verified byte-identical by the
    output-identity job). Within-run gprof: 3.23% → 2.13%.
  - `Mixer::Mix` (9.4%) left untouched: its dot products are pure reads,
    already auto-vectorized under `-ffp-model=fast`; any manual
    restructuring risks reassociating the reduction.
  - **`fxcmv1::E1/E::get` miss path examined** (`f53de0f`): the
    replacement-priority scan cannot be reordered (tie-breaking between
    equal-priority slots picks the first found — reordering changes which
    slot is replaced), and the scan is already L1-bound within the 128B
    entry; the dominant cost is the unavoidable random entry fetch. Two
    zero-risk tightenings were applied to both union copies: explicit
    7-byte stores replace the constant-size `memset` (no libc-inlining
    dependency) and a `#pragma clang loop unroll(enable)` hints the
    fixed-trip-count scan. Same-run benchmark measured **-0.5% vs parent
    (    within the ±1.5% channel noise)** — neutral; kept for the documented
    tie-breaking constraint and the self-contained miss path.
- **Accumulated safe-speedups measured in one same-run A/B** (via the new
  `Benchmark-baseline:` override): HEAD (Perceive rewrite + GetContextData
  slot-map + E1 miss-path tighten) vs the pre-optimization tree `5d0768a`
  — **-2.1%** (14830 vs 15145 ms on `-n prof_input/input`, min of 2 reps
  each). That is above the channel's noise floor (identical-code runs
  measured +1.5% and -0.6%), so the accumulated wins are real; each piece
  was individually below single-run resolution, which is exactly why the
  accumulated A/B against a known-good tree was needed.
- **CI trimmed** now that the PPMd heap-remap crash is fixed and proven
  byte-neutral: dropped the gdb/debug backtrace probe step (its job —    catching the flaky crash — is done). The Linux job now runs real
  lossless round-trips (`-n`, `-c`, and dictionary paths) instead of the
  compress-only time -v diagnostics. Jobs remaining: Linux round-trips,
  Windows round-trips + binary artifact, PPMd-fix neutrality, output
  identity — correctness guards only, no probes.
- **Performance (byte-identical, verified by CI):** gprof on
  `prof_input/input2` showed ~30% of runtime in the LSTM (valarray
  temporary churn), ~14% in out-of-line `Sigmoid::Logit` +
  `MixerInput::SetInput` (called ~num_models times per bit), and ~10% in
  `Mixer::Mix`/`GetContextData` (double hash lookup per bit):
  - `Sigmoid::Logit` and the `MixerInput` setters are now inline in their
    headers (identical arithmetic, call overhead removed). Kept.
  - `Mixer::Mix` caches the resolved `ContextData*`; `Perceive` reuses it
    for the same bit instead of a second hash lookup (contexts only change
    in `UpdateContexts`, after the mixer Perceive loop). Kept.
  - An LSTM valarray-to-scalar-loop rewrite (the ~30% target) was tried
    and **reverted**: the output-identity job proved it could not be made
    byte-identical. Even with `#pragma clang fp contract(off)` on the
    elementwise loops and the reductions kept in their exact valarray form,
    compressed output still differed (byte 446 vs the pre-optimization
    baseline) — the LSTM's reduction loops (matvec, transpose) are
    auto-vectorized with reassociation, and their partial-sum trees are
    sensitive to the surrounding code, so any restructuring perturbs the
    rounded results. Byte-identity is a hard requirement (Hutter Prize
    archive), so the rewrite stays out until/unless it can be proven
    bit-exact.
  - The kept changes are pure call-overhead/lookup reductions with no FP
    arithmetic change; the output-identity and PPMd-neutrality jobs verify
    them byte-identical against the pre-optimization baseline.
- **mix3 hot-loop study (ContextMap2 311M / ContextMap1 155M /
  ContextMap 119M calls, ≈6.6% combined):** audited `mix3`, the
  `StateMap::set`/`update` it calls, and the `Inputs::add` setter for
  safe (byte-identical) redundancy. Findings:
  - The scan/set order and the prediction stream are hard constraints:
    `mix3`'s branch order is load-bearing, and the outer mixer reads the
    *entire* `model_predictions` array every bit (not just
    `[0, prediction_index)`), so the "decremented" last add's value is
    read and cannot be skipped.
  - `StateMap::update` is already minimal (one load + shift + add +
    store; `set` adds one dependent load) — no removable redundancy.
  - One real micro-opt shipped: `Inputs::add` now writes
    `sqtf[p+2047]` from a precomputed float squash table
    (`sqtf[i] = (float)sqt[i]*conversion_factor`) instead of
    `AddPrediction(squash(p))`. Removes the per-add int→float convert,
    float multiply, and the two `squash` clamp branches from the
    hottest call path; the float arithmetic is element-identical, and
    every `add()` call site was verified to pass in-range `p` (clp'd
    tables, constants, clamped `p1()`/`st>>2`/`length<<5`).
    Measured by the same-run benchmark (min-of-3, run `31897685308`):
    HEAD 15474 ms (spread 2.3%) vs `HEAD~1` 15759 ms (spread 0.3%) →
    **-1.8%**, above the identical-code noise floor (-0.7% / -1.2%) —
    the mix3 family's clamp branches + cvt/mul cost more than the
    self-time share suggested. Identity job: `PASS: HEAD and HEAD~1
    produce byte-identical compressed output`.

### Added
- GitHub Actions CI (`.github/workflows/ci.yml`):
  - **Linux (clang-17)** job: builds `cmix` and `remap`, then verifies
    lossless round-trips on `prof_input/input` — both with no preprocessing
    (`-n`) and with the full preprocess + dictionary path (`-c`/`-d`).
  - **Windows (MinGW clang)** job: builds under MSYS2/MINGW64 and runs the
    same round-trip tests, then rebuilds with `mmap_to_disk` forced on to
    exercise the Win32 mmap shim end-to-end.
  - **PPMd fix output-neutrality** job: builds the pre-fix commit
    (`2eeda28^`, which has the unsafe heap remap) and fixed HEAD with
    identical flags, compresses the same input, and requires byte-identical
    output — empirical proof the fix is content-transparent. If the pre-fix
    build crashes (its known bug), that is reported instead of failing.
- `verify_enwik9.sh` — turnkey full-scale verification: downloads canonical
  `enwik9`, PGO-builds like the submission, runs the full `-e` pipeline and
  the self-extracting archive, and verifies the restored file against the
  canonical MD5/SHA-1; optional cross-check against the official archive.
- `docs/ENWIK9_VERIFICATION.md` — the full-scale procedure, canonical
  enwik9 checksums, expected outputs, and resource requirements.
- `CHANGELOG.md` — this file.
- README: CI badge, a short Development section, and the corrected
  decompressor output filename (`enwik9_uncompressed`).

### Fixed
- `src/models/ppmd.cpp`: removed the periodic PPM heap `munmap`/`mmap`
  remap (every 20000 bytes). The new `mmap(NULL, ...)` could return a
  different base address, orphaning every sub-allocator pointer
  (`UnitsStart`, `MaxContext`, `LoUnit`, `HiUnit`) and segfaulting on the
  next access. This produced the intermittent Linux crash (~40% of
  `prof_input/input`, ASLR-dependent; deterministic under gdb at
  `counter_=20001` with `MaxContext` pointing at the old, unmapped base)
  and the Windows `re-open(mmap_path): Invalid argument` failure. The
  file-backed `MAP_SHARED` heap already spills to disk under memory
  pressure, so the remap was pointless. Content-transparent: compressed
  output is unchanged.

### Changed
- `src/models/ppmd.cpp`: `mmap_to_disk` is now `true` by default on **both**
  Linux and Windows — the `mman_shim.h` shim passes CI round-trip tests on
  both the `-n` and dictionary paths. Still overridable at build time with
  `-DMMAP_TO_DISK_DEFAULT=0/1`; a full-scale enwik9 run on Windows remains
  recommended before any new submission.
- Windows CI: single native build (default is now the mmap-backed build)
  that runs both round-trip tests and uploads `cmix.exe` as the
  `fx2-cmix-windows-x64` artifact.
- `tools/build_windows.sh`: scripted native Windows release build (MSYS2
  MINGW64, clang, `MARCH=core-avx2`, submission SEED); used by CI and
  documented for local builds.
- `docs/WINDOWS_BUILD.md`: native Windows build + artifact guide, mmap
  defaults, and limitations.
- `makefile`: new `MARCH=<cpu>` override (e.g. `MARCH=core-avx2`) that
  replaces the `-march` chosen by the COREI7/ZEN2/native detection. CI uses
  `core-avx2` because `fxcmv1.cpp` requires AVX2 and a fixed march keeps
  compressed output reproducible across runners.
- `makefile`: new `DEBUG=1` build flag — adds `-g` and disables stripping
  (`-s`), for symbolicated debugging and CI crash backtraces.
- `src/models/mman_shim.h`: size the Win32 file mapping explicitly from
  the requested `length` (fixes `MapViewOfFile` ERROR_ACCESS_DENIED when
  the backing file's size lags the fd position) and report `GetLastError()`
  on failure. Windows `mmap_to_disk` round-trips now validate in CI.

## [1.0.0] — 2024-10-08 — Hutter Prize submission

The official fx2-cmix Hutter Prize submission. Awarded October 8, 2024;
improvement of 1.585% over the previous record holder (fx-cmix).
See `README.md` for the full submission description and results.

### Summary of the submission (from upstream)
- NLP: stemmer-based natural language processing (from paq8px(d)), reverse
  dictionary transform, single-pass Wikipedia transform, new article order
  (embeddings + t-SNE + k-means pipeline).
- Slimmer cmix core: removed indirect predictors, match predictors, and
  mixers; split main predictors across three ContextMaps; sparse match model.
- Result: 110,793,128-byte self-extracting archive (vs. 112,578,322 previous
  record); ~65 h decompression on the reference machine.

[Unreleased]: https://github.com/niko4244/fx2-cmix/compare/main...dev
[1.0.0]: https://github.com/niko4244/fx2-cmix/releases/tag/v1.0.0

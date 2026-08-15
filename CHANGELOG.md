# Changelog

All notable changes to fx2-cmix are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to the Hutter Prize rules of the [Prize](http://prize.hutter1.net/).

## [Unreleased] — dev branch

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
    runner (`-n prof_input/input`, min of 2 reps each), reporting the delta
    and uploading a `benchmark-report` artifact. Cross-run totals on shared
    runners are noise (117s vs 125s for byte-identical code), but an
    interleaved same-job comparison cancels drift, giving speedup/regression
    work a stable measurement channel. Informational — the output-identity
    job remains the correctness gate.

### Changed
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

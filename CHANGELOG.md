# Changelog

All notable changes to fx2-cmix are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to the Hutter Prize rules of the [Prize](http://prize.hutter1.net/).

## [Unreleased] — dev branch

### Added
- GitHub Actions CI (`.github/workflows/ci.yml`):
  - **Linux (clang-17)** job: builds `cmix` and `remap`, then verifies
    lossless round-trips on `prof_input/input` — both with no preprocessing
    (`-n`) and with the full preprocess + dictionary path (`-c`/`-d`).
  - **Windows (MinGW clang)** job: builds under MSYS2/MINGW64 and runs the
    same round-trip tests, then rebuilds with `mmap_to_disk` forced on to
    exercise the Win32 mmap shim end-to-end.
- `CHANGELOG.md` — this file.
- README: CI badge and a short Development section.

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
- `src/models/ppmd.cpp`: `mmap_to_disk` is now build-configurable via
  `-DMMAP_TO_DISK_DEFAULT=0/1`. Platform defaults are unchanged: `true` on
  Linux (submission default), `false` on Windows (pending full-scale
  validation of `mman_shim.h`).
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

# Native Windows release build

fx2-cmix builds and runs natively on Windows (64-bit) via MSYS2/MINGW64 with
clang. CI produces a downloadable `cmix.exe` artifact on every run; you can
also build locally with the provided script.

## Get the prebuilt binary (CI artifact)

Every CI run of the `dev` branch builds a native `cmix.exe` and uploads it as
an artifact named **`fx2-cmix-windows-x64`** (kept 14 days):

1. Open the run at https://github.com/niko4244/fx2-cmix/actions
2. Click the run → *Windows (MinGW clang) build + round-trip tests* job
3. Download the `fx2-cmix-windows-x64` artifact.

The binary in that artifact has passed lossless round-trip tests (both the
no-preprocess `-n` and the full preprocess + dictionary `-c`/`-d` paths).

## Build locally

1. Install [MSYS2](https://www.msys2.org), then from a **MINGW64** terminal:

   ```bash
   pacman -S --needed make diffutils coreutils \
     mingw-w64-x86_64-clang mingw-w64-x86_64-make \
     mingw-w64-x86_64-gcc-libs mingw-w64-x86_64-winpthreads
   ```

2. Build (same flags as CI and the official submission's SEED/MARCH):

   ```bash
   git clone https://github.com/niko4244/fx2-cmix.git
   cd fx2-cmix
   bash tools/build_windows.sh            # -> ./cmix.exe
   bash tools/build_windows.sh out/       # optional: copy to out/cmix.exe
   ```

3. Smoke-test the binary:

   ```bash
   ./cmix.exe -n prof_input/input out_n.bin
   ./cmix.exe -d out_n.bin back_n.bin
   cmp prof_input/input back_n.bin && echo "round-trip OK"
   ```

## Notes and limitations

- **mmap_to_disk** defaults to `true` on Windows: PPM backs its 14 GB heap
  with a memory-mapped temp file (`ppm.temp`) via the `mman_shim.h` shim,
  validated by the CI round-trip tests. Override at build time with
  `-DMMAP_TO_DISK_DEFAULT=0` (RAM-backed) if you prefer.
- **Not a submission build**: the Windows binary is not PGO-optimized and not
  upx-compressed (that is the Linux `build_and_construct_comp.sh` procedure),
  and it is not intended for Hutter Prize submissions. Use it for
  experiments, development, and correctness checks.
- **Full-scale validation** of the Windows build (a complete enwik9 run) has
  not been performed; the small-corpus round-trips in CI pass.
- The `-e` (enwik9 self-extracting archive) pipeline is Linux-oriented and
  not exercised on Windows; use `-c`/`-d`/`-n` for round-trips.

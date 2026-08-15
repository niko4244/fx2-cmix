#!/usr/bin/env bash
#
# Build a native Windows cmix.exe (MinGW, MSYS2 MINGW64 shell).
#
# Requirements:
#   MSYS2 (https://www.msys2.org), MINGW64 toolchain. Install once:
#     pacman -S --needed make diffutils coreutils \
#       mingw-w64-x86_64-clang mingw-w64-x86_64-make \
#       mingw-w64-x86_64-gcc-libs mingw-w64-x86_64-winpthreads
#
# Usage (inside an MSYS2 MINGW64 shell):
#   bash tools/build_windows.sh              # builds ./cmix.exe
#   bash tools/build_windows.sh out/         # ...and copies it to out/
#
# Notes:
#   * MARCH is fixed to core-avx2: fxcmv1.cpp uses AVX2 intrinsics.
#   * SEED/UPDATE_LIMIT match the official submission build (see
#     build_and_construct_comp.sh).
#   * The binary is not PGO-optimized and not upx-compressed; for a
#     submission-grade binary use the full Linux procedure in
#     build_and_construct_comp.sh.
#   * mmap_to_disk defaults to true on Windows (Win32 shim, validated by
#     CI round-trip tests); override with -DMMAP_TO_DISK_DEFAULT=0 if needed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-}"

for t in make clang++; do
  command -v "$t" >/dev/null \
    || { echo "ERROR: '$t' not found — run inside an MSYS2 MINGW64 shell and install the packages listed in the header"; exit 1; }
done

make clean >/dev/null 2>&1 || true
make cmix CC=clang++ MARCH=core-avx2 \
  CFLAGS_DEFINES="-DSEED=923 -DUPDATE_LIMIT=3000" -j"$(nproc 2>/dev/null || echo 4)"

BIN="$(ls cmix.exe 2>/dev/null || echo cmix)"
SIZE=$(stat -c%s "$BIN" 2>/dev/null || echo "?")
echo "Built: $ROOT/$BIN ($SIZE bytes)"

if [ -n "$OUT" ]; then
  mkdir -p "$OUT"
  cp "$BIN" "$OUT/cmix.exe"
  echo "Copied to: $OUT/cmix.exe"
fi

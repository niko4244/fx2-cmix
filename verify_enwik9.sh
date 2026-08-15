#!/usr/bin/env bash
#
# verify_enwik9.sh — full Hutter-Prize-scale validation of fx2-cmix.
#
# Downloads the canonical enwik9, builds the compressor with PGO (the same
# procedure as the official submission), runs the full "-e" pipeline
# (Wikipedia preprocess + compress + self-extracting archive), runs the
# archive to decompress, and verifies the restored file is byte-identical
# to canonical enwik9.
#
# Optionally cross-checks against the OFFICIAL fx2-cmix submission archive
# (the self-extracting decompressor) if you provide its path.
#
# Usage:
#   ./verify_enwik9.sh [work_dir] [official_archive9]
#
#   work_dir          where enwik9.zip/enwik9 are kept (default: ./enwik9_run)
#   official_archive9 path to the official submission's self-extracting
#                     decompressor, if you want the cross-check
#
# Requirements (Ubuntu 20.04/22.04):
#   clang++-17, llvm-profdata-17, upx-ucl, make, curl, unzip
#   (see install_tools/ for the first three)
#
# Resources:
#   Time: ~65 h compression + ~65 h decompression on a c2-standard-4-class
#         machine (faster on beefier hardware).
#   RAM:  16 GB recommended (PPM uses a 14 GB file-backed heap).
#   Disk: ~25 GB free recommended (enwik9 1 GB + transforms ~7 GB + PPM temp).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:-$ROOT/enwik9_run}"
OFFICIAL_ARCHIVE="${2:-}"

ENWIK9_ZIP_URL="https://mattmahoney.net/dc/enwik9.zip"
ENWIK9_ZIP_SIZE=322592222
ENWIK9_SIZE=1000000000
ENWIK9_MD5="e206c3450ac99950df65bf70ef61a12d"
ENWIK9_SHA1="2996e86fb978f93cca8f566cc56998923e7fe581"

say() { echo; echo "== $* =="; }

mkdir -p "$WORK"

say "[1/6] Acquiring canonical enwik9"
cd "$WORK"
if [ ! -f enwik9.zip ]; then
  echo "Downloading $ENWIK9_ZIP_URL ..."
  if ! curl -L --fail -o enwik9.zip "$ENWIK9_ZIP_URL" 2>/dev/null; then
    wget -O enwik9.zip "$ENWIK9_ZIP_URL"
  fi
fi
[ "$(stat -c%s enwik9.zip)" -eq "$ENWIK9_ZIP_SIZE" ] \
  || { echo "ERROR: enwik9.zip size mismatch (got $(stat -c%s enwik9.zip), expected $ENWIK9_ZIP_SIZE)"; exit 1; }
if [ ! -f enwik9 ]; then
  unzip -o enwik9.zip
fi
[ "$(stat -c%s enwik9)" -eq "$ENWIK9_SIZE" ] \
  || { echo "ERROR: enwik9 size mismatch (got $(stat -c%s enwik9), expected $ENWIK9_SIZE)"; exit 1; }
md5=$(md5sum enwik9 | awk '{print $1}')
sha1=$(sha1sum enwik9 | awk '{print $1}')
echo "enwik9 md5 : $md5"
echo "enwik9 sha1: $sha1"
{ [ "$md5" = "$ENWIK9_MD5" ] && [ "$sha1" = "$ENWIK9_SHA1" ]; } \
  || { echo "ERROR: enwik9 checksum mismatch"; exit 1; }
echo "enwik9 OK: 1,000,000,000 bytes, canonical checksums verified"

say "[2/6] Checking build tools"
for t in clang++-17 llvm-profdata-17 upx-ucl make; do
  command -v "$t" >/dev/null \
    || { echo "ERROR: $t not found (see install_tools/ and apt-get)"; exit 1; }
done

say "[3/6] Building with PGO (official submission procedure)"
cd "$ROOT"
make clean >/dev/null 2>&1 || true
rm -rf pgo_data run
bash build_and_construct_comp.sh
[ -x run/cmix ] || { echo "ERROR: build did not produce run/cmix"; exit 1; }
echo "Built run/cmix ($(stat -c%s run/cmix) bytes)"

say "[4/6] Compressing enwik9 (creates run/archive9; ~65 h on reference hardware)"
cd "$ROOT/run"
echo "Started: $(date -u)"
time ./cmix -e "$WORK/enwik9" enwik9.comp 2>&1 | tee "$WORK/compress.log"
[ -x archive9 ] || { echo "ERROR: archive9 was not created"; exit 1; }
echo "archive9 size: $(stat -c%s archive9) bytes"

say "[5/6] Decompressing (runs ./archive9; ~65 h on reference hardware)"
time ./archive9 2>&1 | tee "$WORK/decompress.log"
[ -f enwik9_uncompressed ] || { echo "ERROR: enwik9_uncompressed was not produced"; exit 1; }

say "[6/6] Verifying byte-identity"
if cmp -s "$WORK/enwik9" enwik9_uncompressed; then
  echo "ROUND-TRIP OK: decompressed output is byte-identical to canonical enwik9"
else
  echo "ROUND-TRIP FAILED: restored file differs from canonical enwik9"
  exit 1
fi
echo "restored md5 : $(md5sum enwik9_uncompressed | awk '{print $1}')  (canonical $ENWIK9_MD5)"
echo "restored sha1: $(sha1sum enwik9_uncompressed | awk '{print $1}')  (canonical $ENWIK9_SHA1)"
echo "restored sha256: $(sha256sum enwik9_uncompressed | awk '{print $1}')"

if [ -n "$OFFICIAL_ARCHIVE" ]; then
  say "Cross-check against official archive: $OFFICIAL_ARCHIVE"
  mkdir -p "$WORK/official"
  cp "$OFFICIAL_ARCHIVE" "$WORK/official/archive9"
  chmod +x "$WORK/official/archive9"
  cd "$WORK/official"
  ./archive9 2>&1 | tee "$WORK/official_decompress.log"
  if cmp -s "$WORK/enwik9" enwik9_uncompressed; then
    echo "OFFICIAL ARCHIVE OK: also reproduces canonical enwik9 byte-identically"
  else
    echo "OFFICIAL ARCHIVE FAILED: output differs from canonical enwik9"
    exit 1
  fi
fi

echo
echo "DONE."
echo "  our rebuilt archive9 : $(stat -c%s "$ROOT/run/archive9") bytes"
echo "  official submission  : 110,793,128 bytes (see README)"
echo "  restored file        : byte-identical to canonical enwik9"

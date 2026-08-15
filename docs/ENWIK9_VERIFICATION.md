# Full-scale enwik9 verification

This document describes how to run the complete Hutter-Prize-scale validation
of fx2-cmix on a big Linux machine: compress canonical `enwik9`, decompress
the self-extracting archive, and verify the result is **byte-identical** to
the official data (and optionally to the official submission archive).

CI (`ci.yml`) validates lossless round-trips on the small `prof_input/input`
corpus and that the PPMd fix is output-neutral, but the real submission-style
run is too long and memory-hungry for CI, so it is run manually.

## Canonical enwik9

| | value |
|---|---|
| source | https://mattmahoney.net/dc/enwik9.zip (322,592,222 bytes) |
| size | 1,000,000,000 bytes |
| MD5 | `e206c3450ac99950df65bf70ef61a12d` |
| SHA-1 | `2996e86fb978f93cca8f566cc56998923e7fe581` |

These are the values published by Matt Mahoney (https://mattmahoney.net/dc/textdata.html).
Any restored file that matches these checksums is byte-identical to the
canonical enwik9 used by the Hutter Prize.

## Resources

- **Time**: roughly 65 h compression + 65 h decompression on a
  c2-standard-4-class machine (Intel Xeon @ ~3.1 GHz, Geekbench ~1026).
  Faster hardware shortens this proportionally.
- **RAM**: 16 GB recommended (PPM uses a 14 GB file-backed heap; peak RSS on
  the reference machine was ~9.5 GB).
- **Disk**: ~25 GB free recommended (enwik9 1 GB, Wikipedia transforms ~7 GB,
  PPM temp file, archive output).

## Quick start

```bash
# 1. Install the toolchain (one-time):
./install_tools/install_clang-17.sh   # clang++-17, llvm-profdata-17
sudo apt-get install -y upx-ucl make curl unzip

# 2. Run the whole verification (downloads enwik9, PGO build, compress,
#    decompress, checksum comparison):
./verify_enwik9.sh
```

The script prints progress and a final summary; it fails loudly on any
checksum mismatch. Intermediate files live in `enwik9_run/`.

### Cross-checking against the official submission archive

The official fx2-cmix submission archive is a self-extracting decompressor
(110,793,128 bytes), linked from the README
(https://drive.google.com/file/d/14QillUEElT5vR0ttmayRAXlciXuPwDWm/).
Download it once and pass its path to cross-check that it also reproduces
canonical enwik9:

```bash
./verify_enwik9.sh enwik9_run /path/to/official_archive9
```

## Manual steps (equivalent to the script)

```bash
# build with PGO (same procedure as the submission)
bash build_and_construct_comp.sh          # produces run/cmix

# compress enwik9 -> archive9
cd run
./cmix -e /path/to/enwik9 enwik9.comp     # ~65 h; expected: 934220400 bytes -> 110111245 bytes

# decompress (no arguments)
./archive9                                # ~65 h; produces enwik9_uncompressed

# verify byte-identity
cmp /path/to/enwik9 enwik9_uncompressed
md5sum  enwik9_uncompressed   # expect e206c3450ac99950df65bf70ef61a12d
sha1sum enwik9_uncompressed   # expect 2996e86fb978f93cca8f566cc56998923e7fe581
```

Note: the decompressor writes the restored file as **`enwik9_uncompressed`**
(see `src/runner.cpp`).

## Expected outputs (from the official submission, README)

| metric | value |
|---|---|
| compressor executable size | 441,463 bytes |
| self-extracting archive size | 110,351,665 bytes (archive portion) |
| total submission size | 110,793,128 bytes |
| improvement over previous record | 1.585% |
| compression output line | `934220400 bytes -> 110111245 bytes in 228589.79 s.` |
| decompression output line | `110111245 bytes -> 934220400 bytes in 229670.44 s.` |
| decompression RAM max | 9,523,660 KiB |

A rebuilt binary will not necessarily produce a bit-identical *archive*
(build/toolchain differences), but it **must** decompress to the identical
1 GB `enwik9`. That is what this verification checks.

## Notes on the current dev-branch fixes

- The PPMd heap-remap removal (`src/models/ppmd.cpp`) is content-transparent:
  it only stops the unsafe periodic `munmap`/`mmap`, so compressed output must
  be unchanged. CI's "PPMd fix output-neutrality" job verifies this on the
  small corpus (identical compressed bytes before/after the fix).
- `mmap_to_disk` is `true` by default on both Linux and Windows (the Win32
  shim passes CI round-trips); a full-scale enwik9 run on Windows is still
  recommended before any new submission.

## What CI does and does not cover

Covered: builds (Linux clang-17, Windows MinGW clang), lossless round-trips
on `prof_input/input` (no-preprocess, preprocess, and dictionary paths),
Windows `mmap_to_disk` round-trip, and byte-neutrality of the PPMd fix.

Not covered: the full 1 GB enwik9 pipeline (too long/too much RAM for
standard runners). This document + `verify_enwik9.sh` are the manual
procedure for that.

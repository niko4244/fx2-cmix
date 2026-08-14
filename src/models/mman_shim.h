// Minimal mmap()/munmap() shim for MinGW/Windows, covering only the
// PROT_READ|PROT_WRITE, MAP_SHARED, fd-backed usage in ppmd.cpp. Not a
// general-purpose POSIX mman.h replacement.
#ifndef MMAN_SHIM_H
#define MMAN_SHIM_H

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <io.h>
#include <sys/types.h>
#include <cstddef>

#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define MAP_SHARED 0x01
#define MAP_FAILED ((void*)-1)

inline void* mmap(void* /*addr*/, size_t length, int prot, int /*flags*/,
                   int fd, off_t offset) {
  HANDLE file = (HANDLE)_get_osfhandle(fd);
  if (file == INVALID_HANDLE_VALUE) return MAP_FAILED;
  DWORD protect = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
  HANDLE mapping = CreateFileMappingA(file, NULL, protect, 0, 0, NULL);
  if (!mapping) return MAP_FAILED;
  DWORD access = (prot & PROT_WRITE) ? FILE_MAP_WRITE : FILE_MAP_READ;
  DWORD offset_hi = (DWORD)((unsigned long long)offset >> 32);
  DWORD offset_lo = (DWORD)((unsigned long long)offset & 0xFFFFFFFFu);
  void* view = MapViewOfFile(mapping, access, offset_hi, offset_lo, length);
  CloseHandle(mapping);  // the view keeps the underlying mapping alive
  if (!view) return MAP_FAILED;
  return view;
}

inline int munmap(void* addr, size_t /*length*/) {
  return UnmapViewOfFile(addr) ? 0 : -1;
}

#else
#include <sys/mman.h>
#endif  // _WIN32

#endif  // MMAN_SHIM_H

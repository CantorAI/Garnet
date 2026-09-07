#pragma once

#include <string>
#if defined(_WIN32)
#include <windows.h>
using NativeLibraryHandle = HMODULE;
inline NativeLibraryHandle OpenNativeLibrary(const char* path) { return LoadLibraryA(path); }
inline auto NativeLibrarySymbol(NativeLibraryHandle library, const char* name) {
    return GetProcAddress(library, name);
}
inline void CloseNativeLibrary(NativeLibraryHandle library) { FreeLibrary(library); }
inline std::string NativeLibraryError() { return std::to_string(GetLastError()); }
#else
#include <dlfcn.h>
using NativeLibraryHandle = void*;
inline NativeLibraryHandle OpenNativeLibrary(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
inline void* NativeLibrarySymbol(NativeLibraryHandle library, const char* name) { return dlsym(library, name); }
inline void CloseNativeLibrary(NativeLibraryHandle library) { dlclose(library); }
inline std::string NativeLibraryError() {
    const char* error = dlerror();
    return error ? error : "unknown dynamic loader error";
}
#endif

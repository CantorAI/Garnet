// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "xlang3/xlang3.h"


#if (WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "garnet.h"

static bool GetCurLibInfo(void* EntryFuncName, std::string& strFullPath,
	std::string& strFolderPath, std::string& strLibName)
{
#if (WIN32)
	HMODULE  hModule = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCSTR>(EntryFuncName), &hModule)) return false;
    char path[32768]{};
    const DWORD length = GetModuleFileNameA(hModule, path, sizeof(path));
    if (!length || length >= sizeof(path)) return false;
	std::string strPath(path, length);
	strFullPath = strPath;
	auto pos = strPath.rfind("\\");
	if (pos != std::string::npos)
	{
		strFolderPath = strPath.substr(0, pos);
		strLibName = strPath.substr(pos + 1);
	}
#else
	Dl_info dl_info;
	if (!dladdr((void*)EntryFuncName, &dl_info) || !dl_info.dli_fname) return false;
	std::string strPath = dl_info.dli_fname;
	strFullPath = strPath;
	auto pos = strPath.rfind("/");
	if (pos != std::string::npos)
	{
		strFolderPath = strPath.substr(0, pos);
		strLibName = strPath.substr(pos + 1);
	}
#endif
	//remove ext
	pos = strLibName.rfind(".");
	if (pos != std::string::npos)
	{
		strLibName = strLibName.substr(0, pos);
	}
	return true;
}

extern "C" XLANG3_PACKAGE_EXPORT const uint32_t xlang3_package_abi_version = X3_ABI_VERSION;

extern "C" XLANG3_PACKAGE_EXPORT X3Status Load(void* pHost, X3Value curModule)
{
    auto* host = static_cast<X3PackageHost*>(pHost);
    if (!host || host->abi_version != X3_ABI_VERSION) return X3_STATUS_ERROR;
    try {
        Garnet::GarnetAPI::BuildAPI();
        return Garnet::GarnetAPI::APISET().Create(host, "garnet", curModule);
    } catch (...) { return X3_STATUS_ERROR; }
}

void Garnet::GarnetAPI::OnPackageCreated(X::Package<GarnetAPI>* package)
{
    std::string fullPath, folder, name;
    if (!GetCurLibInfo(reinterpret_cast<void*>(Load), fullPath, folder, name))
        throw X::Error("cannot locate Garnet library");
    SetModule(package->CurrentModule());
    SetBaseFolder(folder);
}

// ndfake_provider.h - in-tree Network Direct SPI provider (test double)
//
// A user-mode ND provider that simulates an RNIC on regular host memory.
// The NdkRdmaTransport loads it through VVM_ND_PROVIDER_DLL (see
// ndk_transport.cpp) instead of a WSC-registered provider, so the NDKPI
// consumer can be exercised end-to-end without ND hardware.

#pragma once

#ifdef _WIN32

#include <guiddef.h>
typedef struct _GUID GUID;

#ifdef NDFAKE_BUILD
#define NDFAKE_API __declspec(dllexport)
#else
#define NDFAKE_API __declspec(dllimport)
#endif

// {5F0A3C11-2B4E-4C6F-9E1A-2B3C4D5E6F70}
DEFINE_GUID(NDFAKE_PROVIDER_CLSID,
    0x5F0A3C11, 0x2B4E, 0x4C6F, 0x9E, 0x1A, 0x2B, 0x3C, 0x4D, 0x5E, 0x6F, 0x70);

#ifdef __cplusplus
extern "C" {
#endif

// Returns the CLSID the transport must pass to DllGetClassObject.
NDFAKE_API HRESULT WINAPI NDFakeGetProviderClsid(CLSID* pOut);

// Standard COM entry point (declared in combaseapi.h; we implement it).
// NDFAKE_API HRESULT WINAPI DllGetClassObject(REFCLSID rclsid, REFIID riid, void** ppv);

#ifdef __cplusplus
}
#endif

#endif // _WIN32
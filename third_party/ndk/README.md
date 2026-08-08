# Network Direct (ND) headers

This directory vendors the user-mode **Network Direct SPI (NDSPI)** headers used by
the Windows RDMA transport (`src/network/ndk_transport.cpp`).

## Files

| File | Origin |
|------|--------|
| `ndspi.h` | microsoft/NetworkDirect `src/ndutil/ndspi.h` (MIT) |
| `nddef.h` | microsoft/NetworkDirect `src/ndutil/nddef.h` (MIT) |
| `ndstatus.h` | Generated from microsoft/NetworkDirect `src/ndutil/ndstatus.mc` with `mc.exe` (Windows SDK 10.0.22621.0, x64) |

## Regenerating ndstatus.h

The Microsoft NetworkDirect repo ships `ndstatus.mc` rather than the generated
header. To regenerate (e.g. after pulling a newer revision):

```
mc.exe ndstatus.mc   # run inside this directory, then keep only ndstatus.h
```

## Why these are vendored

`ndspi.h` is the application-facing counterpart of the kernel-mode **NDKPI**
(Network Direct Kernel Provider Interface). NDKPI itself is implemented by
RNIC miniport drivers (e.g. NVIDIA's mlx5nd2 provider) and is not callable from
user mode; user-mode applications consume the stack through these NDSPI COM
interfaces. The Windows SDK does not ship `ndspi.h`, so it is pinned here.

Both Microsoft headers are licensed under the MIT License; copyright notices
are preserved in each file.

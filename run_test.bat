@echo off
set "WindowsSdkDir=C:\Program Files (x86)\Windows Kits\10\"
set "WindowsSDKVersion=10.0.26100.0"
set "Path=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%Path%"
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "VVM_ND_PROVIDER_DLL=D:\VulkanVM\build_win\tests\ndfake_provider.dll"
D:\VulkanVM\build_win\tests\ndk_transport_test.exe
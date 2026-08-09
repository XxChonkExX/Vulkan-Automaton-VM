@echo off
set "WindowsSdkDir=C:\Program Files (x86)\Windows Kits\10\"
set "WindowsSDKVersion=10.0.26100.0"
set "Path=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%Path%"
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul

if exist D:\VulkanVM\build_win rmdir /s /q D:\VulkanVM\build_win

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DVVM_BUILD_TESTS=ON ^
    -DCMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION=10.0.26100.0 ^
    -DCMAKE_SYSTEM_VERSION=10.0.26100.0 ^
    -DWINDOWS_SDK_UM="C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/um" ^
    -DWINDOWS_SDK_SHARED="C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared" ^
    -DWINDOWS_SDK_UCRT="C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/ucrt" ^
    -DWINDOWS_SDK_LIB_PATH="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/x64" ^
    -DWINDOWS_SDK_UCRT_LIB_PATH="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64" ^
    -DMSVC_INCLUDE_PATH="C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/include" ^
    -DMSVC_LIBRARY_PATH="C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/lib/x64" ^
    D:\VulkanVM -B D:\VulkanVM\build_win

if errorlevel 1 exit /b 1

cmake --build D:\VulkanVM\build_win --config Release --target ndk_transport_test

if errorlevel 1 exit /b 1

echo BUILD SUCCESS
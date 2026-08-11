@echo off
set "WindowsSdkDir=C:\Program Files (x86)\Windows Kits\10\"
set "WindowsSDKVersion=10.0.26100.0"
set "Path=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%Path%"
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul

cmake --build D:\VulkanVM\build_ninja --target network_remote_test

if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

echo BUILD SUCCESS

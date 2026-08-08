@echo off
REM Android build script for VulkanVM (Windows)
REM Usage: build_android.bat [abi] [platform] [build_type]

setlocal enabledelayedexpansion

REM Default values
set ABI=%1
if "%ABI%"=="" set ABI=arm64-v8a

set PLATFORM=%2
if "%PLATFORM%"=="" set PLATFORM=android-34

set BUILD_TYPE=%3
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

set BUILD_DIR=build_android_%ABI%

REM Android SDK/NDK paths - adjust if needed
if "%ANDROID_SDK_ROOT%"=="" set ANDROID_SDK_ROOT=%LOCALAPPDATA%\Android\Sdk
if "%ANDROID_NDK_ROOT%"=="" set ANDROID_NDK_ROOT=%ANDROID_SDK_ROOT%\ndk\27.0.12077973

echo Building VulkanVM for Android
echo   ABI: %ABI%
echo   Platform: %PLATFORM%
echo   Build Type: %BUILD_TYPE%
echo   NDK: %ANDROID_NDK_ROOT%

REM Check NDK exists
if not exist "%ANDROID_NDK_ROOT%" (
    echo ERROR: Android NDK not found at %ANDROID_NDK_ROOT%
    echo Set ANDROID_NDK_ROOT environment variable
    exit /b 1
)

REM Configure
cmake -B %BUILD_DIR% ^
    -DCMAKE_TOOLCHAIN_FILE=cmake/android.toolchain.cmake ^
    -DANDROID_ABI=%ABI% ^
    -DANDROID_PLATFORM=%PLATFORM% ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DVVM_BUILD_TESTS=OFF ^
    -DVVM_BUILD_EXAMPLES=OFF ^
    -DVVM_BUILD_PYTORCH=OFF ^
    -DVVM_BUILD_ONNX=OFF ^
    -DVVM_BUILD_NETWORK=ON ^
    -DVVM_BUILD_TENSOR_TRANSPORT=ON ^
    -DVVM_BUILD_SHARED=ON ^
    -DVVM_ANDROID_HARDWARE_BUFFER=ON ^
    -DVVM_ANDROID_EXTERNAL_MEMORY=ON ^
    -DCMAKE_TOOLCHAIN_FILE=cmake/android.toolchain.cmake ^
    .

if errorlevel 1 (
    echo CMake configure failed
    exit /b 1
)

REM Build
cmake --build %BUILD_DIR% --config %BUILD_TYPE% -j

if errorlevel 1 (
    echo Build failed
    exit /b 1
)

echo Build complete! Output in %BUILD_DIR%
echo.
echo To install on device:
echo   adb push %BUILD_DIR%\libvulkan_vm.so /data/local/tmp/
echo.
echo Or create AAR for Android Studio integration

endlocal
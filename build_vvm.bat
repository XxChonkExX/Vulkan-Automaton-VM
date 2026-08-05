@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d D:\VulkanVM
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Debug -DVVM_BUILD_NETWORK=ON -DVVM_BUILD_EXAMPLES=ON -DVVM_BUILD_TESTS=ON
if errorlevel 1 exit /b 1
cmake --build build --config Debug
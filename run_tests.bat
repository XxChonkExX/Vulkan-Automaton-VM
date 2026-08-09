@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
ctest --test-dir build_ninja --output-on-failure -C Release
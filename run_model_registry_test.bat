@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;%PATH%
build_ninja\examples\model_registry_test.exe
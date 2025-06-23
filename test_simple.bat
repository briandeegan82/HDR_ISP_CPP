@echo off
echo === Simple Hybrid Backend Test ===

REM Check if Visual Studio is available
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo Visual Studio compiler not found. Please run from Developer Command Prompt.
    echo Or install Visual Studio 2022 with C++ Desktop Development workload.
    pause
    exit /b 1
)

echo Visual Studio compiler found!

REM Create test directory
if not exist "simple_test" mkdir simple_test
cd simple_test

REM Try to compile the quick test
echo Compiling quick test...
cl /EHsc /I"C:\vcpkg\installed\x64-windows\include" quick_test.cpp /link /LIBPATH:"C:\vcpkg\installed\x64-windows\lib" opencv_world480.lib

if %errorlevel% equ 0 (
    echo Compilation successful!
    echo Running test...
    quick_test.exe
) else (
    echo Compilation failed. Check OpenCV installation.
)

cd ..
pause 
# Digital Gain Halide Test Build Script
Write-Host "=== Digital Gain Halide Test Build ===" -ForegroundColor Green

# Create build directory
$build_dir = "test_digital_gain_build"
if (Test-Path $build_dir) {
    Write-Host "Cleaning existing build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $build_dir
}
New-Item -ItemType Directory -Path $build_dir | Out-Null

# Change to build directory
Set-Location $build_dir

# Configure with CMake
Write-Host "Configuring with CMake..." -ForegroundColor Cyan
cmake -G "Visual Studio 17 2022" -A x64 -DUSE_HYBRID_BACKEND=ON -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -f ../test_digital_gain_halide_CMakeLists.txt ..

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit 1
}

# Build the test
Write-Host "Building test executable..." -ForegroundColor Cyan
cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Run the test
Write-Host "Running Digital Gain Halide test..." -ForegroundColor Cyan
./Release/test_digital_gain_halide.exe

if ($LASTEXITCODE -ne 0) {
    Write-Host "Test execution failed!" -ForegroundColor Red
    exit 1
}

Write-Host "=== Test completed successfully! ===" -ForegroundColor Green

# Return to original directory
Set-Location .. 
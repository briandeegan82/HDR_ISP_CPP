# Build script for Black Level Correction Halide Test
Write-Host "Building Black Level Correction Halide Test..." -ForegroundColor Green

# Create build directory
if (!(Test-Path "test_build")) {
    New-Item -ItemType Directory -Path "test_build"
}

# Change to build directory
Set-Location "test_build"

# Copy CMakeLists.txt to build directory
Copy-Item "../test_black_level_correction_halide_CMakeLists.txt" "CMakeLists.txt"

# Configure with CMake using Visual Studio 2022 generator
Write-Host "Configuring with CMake..." -ForegroundColor Yellow
cmake -G "Visual Studio 17 2022" -A x64 -DUSE_HYBRID_BACKEND=ON -DCMAKE_BUILD_TYPE=Release ..

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit 1
}

# Build the project
Write-Host "Building project..." -ForegroundColor Yellow
cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Run the test
Write-Host "Running Black Level Correction Halide Test..." -ForegroundColor Green
./bin/test_black_level_correction_halide.exe

if ($LASTEXITCODE -ne 0) {
    Write-Host "Test failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Test completed successfully!" -ForegroundColor Green

# Return to original directory
Set-Location .. 
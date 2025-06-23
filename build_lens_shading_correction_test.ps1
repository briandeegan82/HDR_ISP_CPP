# Lens Shading Correction Halide Test Build Script
Write-Host "=== Building Lens Shading Correction Halide Test ===" -ForegroundColor Green

# Create build directory
$buildDir = "test_lens_shading_correction_build"
if (Test-Path $buildDir) {
    Write-Host "Cleaning existing build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $buildDir
}
New-Item -ItemType Directory -Path $buildDir | Out-Null

# Change to build directory
Set-Location $buildDir

try {
    # Configure with CMake
    Write-Host "Configuring with CMake..." -ForegroundColor Cyan
    cmake -G "Visual Studio 16 2019" -A x64 -DUSE_HYBRID_BACKEND=ON -DCMAKE_PREFIX_PATH="C:/vcpkg/installed/x64-windows" ../test_lens_shading_correction_halide_CMakeLists.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed!" -ForegroundColor Red
        exit 1
    }
    
    # Build the project
    Write-Host "Building project..." -ForegroundColor Cyan
    cmake --build . --config Release
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Build completed successfully!" -ForegroundColor Green
    
    # Run the test
    Write-Host "Running Lens Shading Correction Halide test..." -ForegroundColor Cyan
    $testExe = ".\bin\Release\lens_shading_correction_halide_test.exe"
    
    if (Test-Path $testExe) {
        & $testExe
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Test completed successfully!" -ForegroundColor Green
        } else {
            Write-Host "Test failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        }
    } else {
        Write-Host "Test executable not found: $testExe" -ForegroundColor Red
    }
    
} catch {
    Write-Host "Error during build: $_" -ForegroundColor Red
    exit 1
} finally {
    # Return to original directory
    Set-Location ..
}

Write-Host "=== Build Script Complete ===" -ForegroundColor Green 
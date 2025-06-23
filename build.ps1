# Create build directory if it doesn't exist
$buildDir = "build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir
}

# Change to build directory
Set-Location $buildDir

# Check if hybrid backend should be enabled
$enableHybrid = $false
if (Test-Path "C:\vcpkg\installed\x64-windows\share\halide") {
    Write-Host "Halide found - enabling hybrid backend" -ForegroundColor Green
    $enableHybrid = $true
} else {
    Write-Host "Halide not found - building with OpenCV OpenCL only" -ForegroundColor Yellow
}

# Configure with CMake using vcpkg toolchain
Write-Host "Configuring CMake..."
if ($enableHybrid) {
    cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release -DUSE_HYBRID_BACKEND=ON
} else {
    cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release -DUSE_HYBRID_BACKEND=OFF
}

# Build the project
Write-Host "Building project..."
cmake --build . --config Release

# Return to original directory
Set-Location $PSScriptRoot

Write-Host "Build complete!" -ForegroundColor Green 
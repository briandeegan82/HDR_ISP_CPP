# Create build directory if it doesn't exist
$buildDir = "build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir
}

# Change to build directory
Set-Location $buildDir

# Configure with CMake using vcpkg toolchain
Write-Host "Configuring CMake..."
cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release

# Build the project
Write-Host "Building project..."
cmake --build . --config Release

# Return to original directory
Set-Location $PSScriptRoot

Write-Host "Build complete!" 
# Test script for Bayer Noise Reduction Halide module
Write-Host "Building Bayer Noise Reduction Halide test..." -ForegroundColor Green

# Create test build directory
$test_build_dir = "test_bayer_noise_reduction_build"
if (Test-Path $test_build_dir) {
    Remove-Item -Recurse -Force $test_build_dir
}
New-Item -ItemType Directory -Path $test_build_dir | Out-Null

# Copy test CMakeLists.txt
Copy-Item "test_bayer_noise_reduction_halide_CMakeLists.txt" "$test_build_dir/CMakeLists.txt"

# Copy test source
Copy-Item "test_bayer_noise_reduction_halide.cpp" "$test_build_dir/"

# Copy necessary source files
$source_files = @(
    "src/modules/bayer_noise_reduction/bayer_noise_reduction_halide.cpp",
    "src/modules/bayer_noise_reduction/bayer_noise_reduction_halide.hpp",
    "src/common/eigen_utils.hpp",
    "src/common/eigen_utils.cpp"
)

foreach ($file in $source_files) {
    if (Test-Path $file) {
        $dest_dir = "$test_build_dir/src/modules/bayer_noise_reduction/"
        if (!(Test-Path $dest_dir)) {
            New-Item -ItemType Directory -Path $dest_dir -Force | Out-Null
        }
        Copy-Item $file $dest_dir
    } else {
        Write-Host "Warning: Source file not found: $file" -ForegroundColor Yellow
    }
}

# Copy common directory
if (Test-Path "src/common") {
    Copy-Item "src/common" "$test_build_dir/src/" -Recurse -Force
}

# Copy include directory
if (Test-Path "include") {
    Copy-Item "include" "$test_build_dir/" -Recurse -Force
}

# Change to test build directory
Set-Location $test_build_dir

# Configure with CMake
Write-Host "Configuring with CMake..." -ForegroundColor Yellow
cmake -DUSE_HYBRID_BACKEND=ON -DCMAKE_BUILD_TYPE=Release ..

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# Build
Write-Host "Building..." -ForegroundColor Yellow
cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# Run test
Write-Host "Running Bayer Noise Reduction Halide test..." -ForegroundColor Green
./test_bayer_noise_reduction_halide.exe

if ($LASTEXITCODE -ne 0) {
    Write-Host "Test failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host "Bayer Noise Reduction Halide test completed successfully!" -ForegroundColor Green
Set-Location .. 
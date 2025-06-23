# Create a directory for vcpkg if it doesn't exist
$vcpkgDir = "C:\vcpkg"
if (-not (Test-Path $vcpkgDir)) {
    Write-Host "Cloning vcpkg repository..."
    git clone https://github.com/Microsoft/vcpkg.git $vcpkgDir
}

# Change to vcpkg directory
Set-Location $vcpkgDir

# Bootstrap vcpkg if not already done
if (-not (Test-Path ".\vcpkg.exe")) {
    Write-Host "Bootstrapping vcpkg..."
    .\bootstrap-vcpkg.bat
}

# Install required packages with OpenCL support
Write-Host "Installing required packages with OpenCL support..."
.\vcpkg install opencv[opencl]:x64-windows
.\vcpkg install eigen3:x64-windows
.\vcpkg install libraw:x64-windows
.\vcpkg install yaml-cpp:x64-windows
.\vcpkg install fftw3:x64-windows
.\vcpkg install gsl:x64-windows
.\vcpkg install halide:x64-windows

# Integrate vcpkg with CMake
Write-Host "Integrating vcpkg with CMake..."
.\vcpkg integrate install

# Return to original directory
Set-Location $PSScriptRoot

Write-Host "vcpkg setup complete with OpenCL and Halide support!" 
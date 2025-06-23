# HDR ISP C++

A C++ implementation of an Image Signal Processor (ISP) pipeline for High Dynamic Range (HDR) image processing.

## Overview

This project implements a complete ISP pipeline for processing RAW images into high-quality HDR images. It includes various modules for different stages of image processing:

- Auto Exposure
- Auto White Balance
- Bayer Noise Reduction
- Black Level Correction
- Color Correction Matrix
- Color Space Conversion
- Demosaic
- HDR Durand
- Lens Shading Correction
- Local Dynamic Contrast Improvement (LDCI)
- Noise Reduction 2D
- OECF (Opto-Electronic Conversion Function)
- PWC Generation
- Scale
- Sharpen
- White Balance
- YUV Conversion Format

## Prerequisites

- Windows 10 or later
- Git
- CMake 3.10 or later
- Visual Studio 2022 or later with C++17 support
- vcpkg (will be installed automatically by the setup script)

## Dependencies

The project uses the following libraries:
- OpenCV
- Eigen3
- LibRaw
- yaml-cpp
- FFTW3
- GSL

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HDR_ISP_CPP.git
cd HDR_ISP_CPP
```

2. Run the installation scripts:
```powershell
# Install CMake (if not already installed)
.\install_cmake.ps1

# Set up vcpkg and install dependencies
.\setup_vcpkg.ps1

# Build the project
.\build.ps1
```

## Project Structure

```
HDR_ISP_CPP/
├── CMakeLists.txt              # Main CMake configuration
├── src/
│   ├── common/                 # Common utilities and base classes
│   └── modules/               # Individual ISP modules
│       ├── auto_exposure/
│       ├── auto_white_balance/
│       ├── bayer_noise_reduction/
│       └── ...
├── include/                    # Public headers
├── config/                     # Configuration files
└── data/                       # Test data and resources
```

## Building

The project uses CMake as its build system. The build process is automated through the provided PowerShell scripts:

1. `install_cmake.ps1`: Installs CMake if not present
2. `setup_vcpkg.ps1`: Sets up vcpkg and installs required dependencies
3. `build.ps1`: Configures and builds the project

To build manually:
```powershell
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Usage

[Usage instructions will be added as the project develops]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[License information to be added]

## Acknowledgments

- This project is a C++ port of a Python-based ISP implementation
- Thanks to all the open-source libraries used in this project 
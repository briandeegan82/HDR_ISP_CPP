# Hybrid Backend: OpenCV + Halide + OpenCL

This project now supports a hybrid approach combining OpenCV and Halide for maximum performance across different platforms while avoiding proprietary dependencies like CUDA.

## Overview

The hybrid backend automatically selects the best available acceleration method:
1. **Halide OpenCL** (fastest, when available)
2. **OpenCV OpenCL** (good performance, widely supported)
3. **Halide CPU** (fast CPU fallback)
4. **OpenCV CPU** (reliable fallback)

## Setup

### 1. Install Dependencies
```powershell
# Run the hybrid setup script
.\setup_hybrid.ps1
```

This will:
- Install OpenCV with OpenCL support
- Install Halide
- Configure the build system

### 2. Build the Project
```powershell
.\build.ps1
```

## Usage

### Basic Usage

```cpp
#include "hybrid_backend.hpp"

// Initialize the hybrid backend (auto-selects best available)
hdr_isp::g_backend->initialize();

// Use convenience macros to switch backends
USE_OPENCV_OPENCL();  // Force OpenCV OpenCL
USE_HALIDE_CPU();     // Force Halide CPU
USE_AUTO();           // Auto-select best backend

// Performance measurement
START_TIMER();
// ... your processing code ...
double elapsed_ms = END_TIMER();
```

### In Your ISP Modules

```cpp
#include "hybrid_backend.hpp"

class MyISPModule {
public:
    cv::Mat process(const cv::Mat& input) {
        // The backend automatically uses the best available method
        return hdr_isp::g_backend->processWithOpenCV(input, "gaussian_blur");
    }
    
    cv::Mat processWithHalide(const cv::Mat& input) {
        // Convert to Halide format
        auto halide_input = hdr_isp::g_backend->cvMatToHalide(input);
        
        // Process with Halide
        auto halide_output = hdr_isp::g_backend->processWithHalide(halide_input, "custom_operation");
        
        // Convert back to OpenCV
        return hdr_isp::g_backend->halideToCvMat(halide_output);
    }
};
```

### Performance Benchmarking

```cpp
#include "performance_benchmark.hpp"

// Run a quick benchmark
auto results = hdr_isp::quickBenchmark(test_image);

// Generate a detailed report
hdr_isp::PerformanceBenchmark benchmark;
auto full_results = benchmark.runFullBenchmark(test_image, 1000);
benchmark.generateReport(full_results, "benchmark_report.txt");
```

## Backend Selection Strategy

The system automatically selects the best backend based on:

1. **Hardware Detection**: Checks for OpenCL-capable devices
2. **Performance Testing**: Benchmarks available backends
3. **Fallback Strategy**: Ensures the pipeline always works

### Priority Order:
1. **Halide OpenCL** - Best performance for compute-intensive tasks
2. **OpenCV OpenCL** - Good performance, widely supported
3. **Halide CPU** - Fast CPU implementation with SIMD
4. **OpenCV CPU** - Reliable fallback

## Platform Support

| Platform | OpenCL | Halide | Notes |
|----------|--------|--------|-------|
| Windows (Intel) | ✅ | ✅ | Full support |
| Windows (AMD) | ✅ | ✅ | Full support |
| Windows (NVIDIA) | ✅ | ✅ | OpenCL support (no CUDA) |
| Linux | ✅ | ✅ | Full support |
| macOS | ✅ | ✅ | Full support |
| ARM | ✅ | ✅ | Limited OpenCL support |

## Performance Expectations

### Typical Speedups (compared to OpenCV CPU):

| Operation | OpenCV OpenCL | Halide CPU | Halide OpenCL |
|-----------|---------------|------------|---------------|
| Gaussian Blur | 2-4x | 3-5x | 5-10x |
| Color Conversion | 2-3x | 2-4x | 4-8x |
| Resize | 3-5x | 4-6x | 6-12x |
| Convolution | 2-4x | 3-5x | 5-10x |

*Results vary based on hardware, image size, and specific operations.*

## Troubleshooting

### OpenCL Not Available
- Ensure graphics drivers are up to date
- Check if your GPU supports OpenCL
- Verify OpenCL runtime is installed

### Halide Build Issues
- Ensure you have a C++17 compatible compiler
- Check that vcpkg is properly configured
- Verify Halide installation with `vcpkg list | findstr halide`

### Performance Issues
- Run benchmarks to identify bottlenecks
- Check which backend is being used
- Profile your specific workload

## Advanced Usage

### Custom Halide Operations

```cpp
// Define a custom Halide operation
Halide::Func customOperation(Halide::Func input) {
    Halide::Func output;
    Halide::Var x, y, c;
    
    output(x, y, c) = input(x, y, c) * 1.5f; // Example operation
    
    return output;
}

// Use in your pipeline
auto halide_input = g_backend->cvMatToHalide(input);
Halide::Buffer<float> output = customOperation(halide_input).realize({width, height, channels});
```

### Backend-Specific Optimizations

```cpp
// Optimize for specific hardware
if (g_backend->getCurrentBackend() == BackendType::HALIDE_OPENCL) {
    // Use Halide-specific optimizations
    // Schedule for GPU
} else if (g_backend->getCurrentBackend() == BackendType::OPENCV_OPENCL) {
    // Use OpenCV OpenCL optimizations
    // Enable specific OpenCL kernels
}
```

## Migration Guide

### From OpenCV-Only Code

1. **Include the hybrid backend header**:
   ```cpp
   #include "hybrid_backend.hpp"
   ```

2. **Initialize the backend**:
   ```cpp
   hdr_isp::g_backend->initialize();
   ```

3. **Replace direct OpenCV calls** (optional):
   ```cpp
   // Instead of: cv::GaussianBlur(input, output, cv::Size(5,5), 0);
   // Use: g_backend->processWithOpenCV(input, "gaussian_blur");
   ```

### From CUDA Code

1. **Replace CUDA kernels** with Halide operations
2. **Use OpenCL** instead of CUDA for GPU acceleration
3. **Maintain the same API** for seamless integration

## Contributing

When adding new operations:

1. **Implement in both OpenCV and Halide** when possible
2. **Add performance benchmarks** for new operations
3. **Test across different backends** and platforms
4. **Document any backend-specific optimizations**

## License

This hybrid backend approach maintains the same license as the original project while providing cross-platform, high-performance image processing capabilities. 
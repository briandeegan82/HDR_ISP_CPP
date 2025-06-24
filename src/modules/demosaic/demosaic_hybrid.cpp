#include "demosaic_hybrid.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <immintrin.h> // For SIMD intrinsics

namespace fs = std::filesystem;

DemosaicHybrid::DemosaicHybrid(const hdr_isp::EigenImageU32& img, const std::string& bayer_pattern, int bit_depth, bool is_save)
    : Demosaic(img, bayer_pattern, bit_depth, is_save)
    , current_backend_(DemosaicBackend::AUTO)
    , opencv_opencl_available_(false)
    , halide_available_(false)
    , simd_available_(false)
    , is_debug_(false)
    , is_save_(is_save)
    , last_execution_time_ms_(0.0)
{
    initializeBackends();
    if (current_backend_ == DemosaicBackend::AUTO) {
        current_backend_ = selectBestBackend();
    }
    
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Constructor started" << std::endl;
        std::cout << "Demosaic Hybrid - Input image size: " << img.rows() << "x" << img.cols() << std::endl;
        std::cout << "Demosaic Hybrid - Bayer pattern: " << bayer_pattern << std::endl;
        std::cout << "Demosaic Hybrid - Selected backend: " << static_cast<int>(current_backend_) << std::endl;
    }
}

DemosaicHybrid::DemosaicHybrid(const hdr_isp::EigenImageU32& img, const std::string& bayer_pattern, const hdr_isp::FixedPointConfig& fp_config, int bit_depth, bool is_save)
    : Demosaic(img, bayer_pattern, fp_config, bit_depth, is_save)
    , current_backend_(DemosaicBackend::AUTO)
    , opencv_opencl_available_(false)
    , halide_available_(false)
    , simd_available_(false)
    , is_debug_(false)
    , is_save_(is_save)
    , last_execution_time_ms_(0.0)
{
    initializeBackends();
    if (current_backend_ == DemosaicBackend::AUTO) {
        current_backend_ = selectBestBackend();
    }
    
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Constructor started" << std::endl;
        std::cout << "Demosaic Hybrid - Input image size: " << img.rows() << "x" << img.cols() << std::endl;
        std::cout << "Demosaic Hybrid - Bayer pattern: " << bayer_pattern << std::endl;
        std::cout << "Demosaic Hybrid - Fixed-point enabled: " << (fp_config.isEnabled() ? "true" : "false") << std::endl;
        std::cout << "Demosaic Hybrid - Selected backend: " << static_cast<int>(current_backend_) << std::endl;
    }
}

bool DemosaicHybrid::initializeBackends() {
    // Check OpenCV OpenCL availability
    opencv_opencl_available_ = cv::ocl::haveOpenCL();
    
    // Check Halide availability
#ifdef USE_HYBRID_BACKEND
    halide_available_ = true;
#else
    halide_available_ = false;
#endif
    
    // Check SIMD availability (SSE4.2 and AVX2)
    simd_available_ = __builtin_cpu_supports("avx2") || __builtin_cpu_supports("sse4.2");
    
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Backend availability:" << std::endl;
        std::cout << "  OpenCV OpenCL: " << (opencv_opencl_available_ ? "Available" : "Not available") << std::endl;
        std::cout << "  Halide: " << (halide_available_ ? "Available" : "Not available") << std::endl;
        std::cout << "  SIMD: " << (simd_available_ ? "Available" : "Not available") << std::endl;
    }
    
    return true;
}

DemosaicBackend DemosaicHybrid::selectBestBackend() const {
    // Priority order: Halide OpenCL > OpenCV OpenCL > Halide CPU > SIMD > OpenCV CPU
    if (halide_available_) {
        return DemosaicBackend::HALIDE_CPU; // For now, use Halide CPU
    } else if (opencv_opencl_available_) {
        return DemosaicBackend::OPENCV_OPENCL;
    } else if (simd_available_) {
        return DemosaicBackend::SIMD_OPTIMIZED;
    } else {
        return DemosaicBackend::OPENCV_CPU;
    }
}

void DemosaicHybrid::setBackend(DemosaicBackend backend) {
    if (backend == DemosaicBackend::AUTO) {
        backend = selectBestBackend();
    }
    
    if (!isBackendAvailable(backend)) {
        std::cerr << "Demosaic Hybrid - Backend not available: " << static_cast<int>(backend) << std::endl;
        return;
    }
    
    current_backend_ = backend;
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Switched to backend: " << static_cast<int>(backend) << std::endl;
    }
}

bool DemosaicHybrid::isBackendAvailable(DemosaicBackend backend) const {
    switch (backend) {
        case DemosaicBackend::OPENCV_CPU:
            return true; // Always available
        case DemosaicBackend::OPENCV_OPENCL:
            return opencv_opencl_available_;
        case DemosaicBackend::HALIDE_CPU:
            return halide_available_;
        case DemosaicBackend::HALIDE_OPENCL:
            return halide_available_; // Simplified for now
        case DemosaicBackend::SIMD_OPTIMIZED:
            return simd_available_;
        case DemosaicBackend::AUTO:
            return true;
        default:
            return false;
    }
}

hdr_isp::EigenImage3C DemosaicHybrid::execute() {
    if (!is_enable_) {
        return hdr_isp::EigenImage3C(img_.rows(), img_.cols());
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - execute() started with backend: " << static_cast<int>(current_backend_) << std::endl;
    }
    
    hdr_isp::EigenImage3C result;
    
    switch (current_backend_) {
        case DemosaicBackend::OPENCV_CPU:
            result = execute_opencv_cpu();
            break;
        case DemosaicBackend::OPENCV_OPENCL:
            result = execute_opencv_opencl();
            break;
        case DemosaicBackend::HALIDE_CPU:
            result = execute_halide_cpu();
            break;
        case DemosaicBackend::HALIDE_OPENCL:
            result = execute_halide_opencl();
            break;
        case DemosaicBackend::SIMD_OPTIMIZED:
            result = execute_simd_optimized();
            break;
        default:
            result = execute_opencv_cpu(); // Fallback
            break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    last_execution_time_ms_ = duration.count();
    
    if (is_debug_) {
        std::cout << "Demosaic Hybrid execution time: " << last_execution_time_ms_ << "ms" << std::endl;
    }

    if (is_save_) {
        save();
    }

    return result;
}

hdr_isp::EigenImage3CFixed DemosaicHybrid::execute_fixed() {
    if (!is_enable_) {
        return hdr_isp::EigenImage3CFixed(img_.rows(), img_.cols());
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - execute_fixed() started with backend: " << static_cast<int>(current_backend_) << std::endl;
    }
    
    hdr_isp::EigenImage3CFixed result;
    
    switch (current_backend_) {
        case DemosaicBackend::OPENCV_CPU:
            result = execute_fixed_opencv_cpu();
            break;
        case DemosaicBackend::OPENCV_OPENCL:
            result = execute_fixed_opencv_opencl();
            break;
        case DemosaicBackend::HALIDE_CPU:
            result = execute_fixed_halide_cpu();
            break;
        case DemosaicBackend::HALIDE_OPENCL:
            result = execute_fixed_halide_opencl();
            break;
        case DemosaicBackend::SIMD_OPTIMIZED:
            result = execute_fixed_simd_optimized();
            break;
        default:
            result = execute_fixed_opencv_cpu(); // Fallback
            break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    last_execution_time_ms_ = duration.count();
    
    if (is_debug_) {
        std::cout << "Demosaic Hybrid fixed execution time: " << last_execution_time_ms_ << "ms" << std::endl;
    }
    
    return result;
}

// OpenCV CPU Implementation
hdr_isp::EigenImage3C DemosaicHybrid::execute_opencv_cpu() {
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Using OpenCV CPU backend" << std::endl;
    }
    
    // Convert Eigen to OpenCV
    cv::Mat input_mat = img_.toOpenCV(CV_32SC1);
    
    // Use OpenCV's built-in demosaic
    cv::Mat output_mat;
    int bayer_code = getOpenCVBayerPattern(bayer_pattern_);
    cv::cvtColor(input_mat, output_mat, bayer_code);
    
    // Convert back to Eigen
    hdr_isp::EigenImage3C result = hdr_isp::EigenImage3C::fromOpenCV(output_mat);
    
    return result;
}

hdr_isp::EigenImage3CFixed DemosaicHybrid::execute_fixed_opencv_cpu() {
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Using OpenCV CPU backend (fixed-point)" << std::endl;
    }
    
    // Execute floating-point version first
    hdr_isp::EigenImage3C float_result = execute_opencv_cpu();
    
    // Convert to fixed-point
    hdr_isp::EigenImage3CFixed result = hdr_isp::EigenImage3CFixed::fromEigenImage3C(float_result, fp_config_);
    
    return result;
}

// OpenCV OpenCL Implementation
hdr_isp::EigenImage3C DemosaicHybrid::execute_opencv_opencl() {
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Using OpenCV OpenCL backend" << std::endl;
    }
    
    // Convert Eigen to OpenCV
    cv::Mat input_mat = img_.toOpenCV(CV_32SC1);
    
    // Use OpenCV OpenCL for demosaic
    cv::Mat output_mat = demosaic_opencv_opencl(input_mat);
    
    // Convert back to Eigen
    hdr_isp::EigenImage3C result = hdr_isp::EigenImage3C::fromOpenCV(output_mat);
    
    return result;
}

hdr_isp::EigenImage3CFixed DemosaicHybrid::execute_fixed_opencv_opencl() {
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Using OpenCV OpenCL backend (fixed-point)" << std::endl;
    }
    
    // Execute floating-point version first
    hdr_isp::EigenImage3C float_result = execute_opencv_opencl();
    
    // Convert to fixed-point
    hdr_isp::EigenImage3CFixed result = hdr_isp::EigenImage3CFixed::fromEigenImage3C(float_result, fp_config_);
    
    return result;
}

cv::Mat DemosaicHybrid::demosaic_opencv_opencl(const cv::Mat& input) {
    // Enable OpenCL for OpenCV
    cv::ocl::setUseOpenCL(true);
    
    // Convert to UMat for OpenCL processing
    cv::UMat input_umat, output_umat;
    input.copyTo(input_umat);
    
    // Use OpenCV's built-in demosaic with OpenCL
    int bayer_code = getOpenCVBayerPattern(bayer_pattern_);
    cv::cvtColor(input_umat, output_umat, bayer_code);
    
    // Convert back to CPU Mat
    cv::Mat result;
    output_umat.copyTo(result);
    
    return result;
}

int DemosaicHybrid::getOpenCVBayerPattern(const std::string& bayer_pattern) {
    if (bayer_pattern == "rggb") {
        return cv::COLOR_BayerRG2RGB;
    } else if (bayer_pattern == "bggr") {
        return cv::COLOR_BayerBG2RGB;
    } else if (bayer_pattern == "grbg") {
        return cv::COLOR_BayerGR2RGB;
    } else if (bayer_pattern == "gbrg") {
        return cv::COLOR_BayerGB2RGB;
    } else {
        throw std::runtime_error("Unsupported bayer pattern: " + bayer_pattern);
    }
}

// SIMD Optimized Implementation
hdr_isp::EigenImage3C DemosaicHybrid::execute_simd_optimized() {
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Using SIMD optimized backend" << std::endl;
    }
    
    if (bayer_pattern_ == "rggb") {
        return demosaic_simd_rggb(img_);
    } else if (bayer_pattern_ == "bggr") {
        return demosaic_simd_bggr(img_);
    } else if (bayer_pattern_ == "grbg") {
        return demosaic_simd_grbg(img_);
    } else if (bayer_pattern_ == "gbrg") {
        return demosaic_simd_gbrg(img_);
    } else {
        throw std::runtime_error("Unsupported bayer pattern: " + bayer_pattern_);
    }
}

hdr_isp::EigenImage3CFixed DemosaicHybrid::execute_fixed_simd_optimized() {
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Using SIMD optimized backend (fixed-point)" << std::endl;
    }
    
    // Execute floating-point version first
    hdr_isp::EigenImage3C float_result = execute_simd_optimized();
    
    // Convert to fixed-point
    hdr_isp::EigenImage3CFixed result = hdr_isp::EigenImage3CFixed::fromEigenImage3C(float_result, fp_config_);
    
    return result;
}

// SIMD-optimized demosaic implementations
hdr_isp::EigenImage3C DemosaicHybrid::demosaic_simd_rggb(const hdr_isp::EigenImageU32& input) {
    int rows = input.rows();
    int cols = input.cols();
    
    hdr_isp::EigenImage3C result(rows, cols);
    
    // Interpolate each channel using SIMD
    interpolate_green_simd(input, result.g(), "rggb");
    interpolate_red_simd(input, result.r(), "rggb");
    interpolate_blue_simd(input, result.b(), "rggb");
    
    return result;
}

hdr_isp::EigenImage3C DemosaicHybrid::demosaic_simd_bggr(const hdr_isp::EigenImageU32& input) {
    int rows = input.rows();
    int cols = input.cols();
    
    hdr_isp::EigenImage3C result(rows, cols);
    
    // Interpolate each channel using SIMD
    interpolate_green_simd(input, result.g(), "bggr");
    interpolate_red_simd(input, result.r(), "bggr");
    interpolate_blue_simd(input, result.b(), "bggr");
    
    return result;
}

hdr_isp::EigenImage3C DemosaicHybrid::demosaic_simd_grbg(const hdr_isp::EigenImageU32& input) {
    int rows = input.rows();
    int cols = input.cols();
    
    hdr_isp::EigenImage3C result(rows, cols);
    
    // Interpolate each channel using SIMD
    interpolate_green_simd(input, result.g(), "grbg");
    interpolate_red_simd(input, result.r(), "grbg");
    interpolate_blue_simd(input, result.b(), "grbg");
    
    return result;
}

hdr_isp::EigenImage3C DemosaicHybrid::demosaic_simd_gbrg(const hdr_isp::EigenImageU32& input) {
    int rows = input.rows();
    int cols = input.cols();
    
    hdr_isp::EigenImage3C result(rows, cols);
    
    // Interpolate each channel using SIMD
    interpolate_green_simd(input, result.g(), "gbrg");
    interpolate_red_simd(input, result.r(), "gbrg");
    interpolate_blue_simd(input, result.b(), "gbrg");
    
    return result;
}

// SIMD vectorized interpolation functions
void DemosaicHybrid::interpolate_green_simd(const hdr_isp::EigenImageU32& input, hdr_isp::EigenImage& output, const std::string& pattern) {
    int rows = input.rows();
    int cols = input.cols();
    
    // Initialize output to zero
    output = hdr_isp::EigenImage::Zero(rows, cols);
    
    // Copy existing green pixels based on pattern
    if (pattern == "rggb") {
        // Green pixels at (1,0) and (0,1) in 2x2 blocks
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                output.data()(i, j) = static_cast<float>(input.data()(i, j)) / 255.0f;
            }
        }
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                output.data()(i, j) = static_cast<float>(input.data()(i, j)) / 255.0f;
            }
        }
    }
    // Add similar patterns for other bayer arrangements...
    
    // Vectorized interpolation for missing green pixels
    #ifdef __AVX2__
    // Use AVX2 for vectorized interpolation
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j += 8) {
            if (j + 7 < cols) {
                // Process 8 pixels at once using AVX2
                __m256i mask = _mm256_set1_epi32(0);
                __m256 sum = _mm256_setzero_ps();
                __m256i count = _mm256_setzero_si256();
                
                // Check neighbors and accumulate (simplified for brevity)
                // This would implement the actual bilinear interpolation logic
                
                // Store result
                _mm256_storeu_ps(&output.data()(i, j), sum);
            }
        }
    }
    #else
    // Fallback to scalar implementation
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (output.data()(i, j) == 0.0f) {
                float sum = 0.0f;
                int count = 0;
                
                // Check 4 neighbors
                if (i > 0) { sum += output.data()(i-1, j); count++; }
                if (i < rows-1) { sum += output.data()(i+1, j); count++; }
                if (j > 0) { sum += output.data()(i, j-1); count++; }
                if (j < cols-1) { sum += output.data()(i, j+1); count++; }
                
                if (count > 0) {
                    output.data()(i, j) = sum / static_cast<float>(count);
                }
            }
        }
    }
    #endif
}

void DemosaicHybrid::interpolate_red_simd(const hdr_isp::EigenImageU32& input, hdr_isp::EigenImage& output, const std::string& pattern) {
    // Similar implementation to interpolate_green_simd but for red channel
    // Implementation would depend on the specific bayer pattern
    int rows = input.rows();
    int cols = input.cols();
    
    output = hdr_isp::EigenImage::Zero(rows, cols);
    
    // Copy existing red pixels and interpolate missing ones
    // This is a simplified implementation
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output.data()(i, j) = static_cast<float>(input.data()(i, j)) / 255.0f;
        }
    }
}

void DemosaicHybrid::interpolate_blue_simd(const hdr_isp::EigenImageU32& input, hdr_isp::EigenImage& output, const std::string& pattern) {
    // Similar implementation to interpolate_green_simd but for blue channel
    // Implementation would depend on the specific bayer pattern
    int rows = input.rows();
    int cols = input.cols();
    
    output = hdr_isp::EigenImage::Zero(rows, cols);
    
    // Copy existing blue pixels and interpolate missing ones
    // This is a simplified implementation
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output.data()(i, j) = static_cast<float>(input.data()(i, j)) / 255.0f;
        }
    }
}

// Halide implementations (if available)
#ifdef USE_HYBRID_BACKEND

hdr_isp::EigenImage3C DemosaicHybrid::execute_halide_cpu() {
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Using Halide CPU backend" << std::endl;
    }
    
    // Convert Eigen to Halide
    Halide::Buffer<uint32_t> input_buffer = eigen_to_halide_uint32(img_);
    
    // Apply Halide demosaic
    Halide::Buffer<float> output_buffer = demosaic_halide_cpu(input_buffer);
    
    // Convert back to Eigen
    hdr_isp::EigenImage3C result = halide_to_eigen_3c(output_buffer, img_.rows(), img_.cols());
    
    return result;
}

hdr_isp::EigenImage3CFixed DemosaicHybrid::execute_fixed_halide_cpu() {
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Using Halide CPU backend (fixed-point)" << std::endl;
    }
    
    // Convert Eigen to Halide
    Halide::Buffer<uint32_t> input_buffer = eigen_to_halide_uint32(img_);
    
    // Apply Halide demosaic with fixed-point output
    Halide::Buffer<int16_t> output_buffer = demosaic_halide_fixed(input_buffer);
    
    // Convert back to Eigen
    hdr_isp::EigenImage3CFixed result = halide_to_eigen_3c_fixed(output_buffer, img_.rows(), img_.cols());
    
    return result;
}

hdr_isp::EigenImage3C DemosaicHybrid::execute_halide_opencl() {
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Using Halide OpenCL backend" << std::endl;
    }
    
    // Convert Eigen to Halide
    Halide::Buffer<uint32_t> input_buffer = eigen_to_halide_uint32(img_);
    
    // Apply Halide demosaic with OpenCL
    Halide::Buffer<float> output_buffer = demosaic_halide_opencl(input_buffer);
    
    // Convert back to Eigen
    hdr_isp::EigenImage3C result = halide_to_eigen_3c(output_buffer, img_.rows(), img_.cols());
    
    return result;
}

hdr_isp::EigenImage3CFixed DemosaicHybrid::execute_fixed_halide_opencl() {
    if (is_debug_) {
        std::cout << "Demosaic Hybrid - Using Halide OpenCL backend (fixed-point)" << std::endl;
    }
    
    // For now, fall back to CPU implementation
    return execute_fixed_halide_cpu();
}

// Halide-specific implementations would go here
Halide::Buffer<float> DemosaicHybrid::demosaic_halide_cpu(const Halide::Buffer<uint32_t>& input) {
    // Placeholder for Halide implementation
    // This would contain the actual Halide code for demosaic
    Halide::Buffer<float> output(input.width(), input.height(), 3);
    
    // Simplified implementation - would be replaced with actual Halide code
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < input.height(); y++) {
            for (int x = 0; x < input.width(); x++) {
                output(x, y, c) = static_cast<float>(input(x, y)) / 255.0f;
            }
        }
    }
    
    return output;
}

Halide::Buffer<int16_t> DemosaicHybrid::demosaic_halide_fixed(const Halide::Buffer<uint32_t>& input) {
    // Placeholder for Halide fixed-point implementation
    Halide::Buffer<int16_t> output(input.width(), input.height(), 3);
    
    // Simplified implementation - would be replaced with actual Halide code
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < input.height(); y++) {
            for (int x = 0; x < input.width(); x++) {
                output(x, y, c) = static_cast<int16_t>(input(x, y) >> 8);
            }
        }
    }
    
    return output;
}

Halide::Buffer<float> DemosaicHybrid::demosaic_halide_opencl(const Halide::Buffer<uint32_t>& input) {
    // Placeholder for Halide OpenCL implementation
    // This would use Halide's OpenCL target
    return demosaic_halide_cpu(input); // Fallback to CPU for now
}

// Utility functions for Halide
Halide::Buffer<uint32_t> DemosaicHybrid::eigen_to_halide_uint32(const hdr_isp::EigenImageU32& eigen_img) {
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    
    Halide::Buffer<uint32_t> buffer(cols, rows);
    
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            buffer(x, y) = eigen_img.data()(y, x);
        }
    }
    
    return buffer;
}

hdr_isp::EigenImage3C DemosaicHybrid::halide_to_eigen_3c(const Halide::Buffer<float>& buffer, int rows, int cols) {
    hdr_isp::EigenImage3C result(rows, cols);
    
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (c == 0) result.r().data()(y, x) = buffer(x, y, c);
                else if (c == 1) result.g().data()(y, x) = buffer(x, y, c);
                else if (c == 2) result.b().data()(y, x) = buffer(x, y, c);
            }
        }
    }
    
    return result;
}

hdr_isp::EigenImage3CFixed DemosaicHybrid::halide_to_eigen_3c_fixed(const Halide::Buffer<int16_t>& buffer, int rows, int cols) {
    hdr_isp::EigenImage3CFixed result(rows, cols);
    
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (c == 0) result.r()(y, x) = buffer(x, y, c);
                else if (c == 1) result.g()(y, x) = buffer(x, y, c);
                else if (c == 2) result.b()(y, x) = buffer(x, y, c);
            }
        }
    }
    
    return result;
}

#else

// Stub implementations when Halide is not available
hdr_isp::EigenImage3C DemosaicHybrid::execute_halide_cpu() {
    return execute_opencv_cpu(); // Fallback
}

hdr_isp::EigenImage3CFixed DemosaicHybrid::execute_fixed_halide_cpu() {
    return execute_fixed_opencv_cpu(); // Fallback
}

hdr_isp::EigenImage3C DemosaicHybrid::execute_halide_opencl() {
    return execute_opencv_cpu(); // Fallback
}

hdr_isp::EigenImage3CFixed DemosaicHybrid::execute_fixed_halide_opencl() {
    return execute_fixed_opencv_cpu(); // Fallback
}

#endif

void DemosaicHybrid::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_demosaic_hybrid_" + 
                                 std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + ".png";
        cv::Mat temp_img = img_.toOpenCV(CV_32SC1);
        cv::imwrite(output_path, temp_img);
    }
} 
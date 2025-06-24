#include "hdr_durand_hybrid.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

HDRDurandToneMappingHybrid::HDRDurandToneMappingHybrid(const cv::Mat& img, const YAML::Node& platform,
                                                       const YAML::Node& sensor_info, const YAML::Node& params)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , params_(params)
    , is_enable_(params["is_enable"].as<bool>())
    , is_save_(params["is_save"].as<bool>())
    , is_debug_(params["is_debug"].as<bool>())
    , sigma_space_(params["sigma_space"].as<float>())
    , sigma_color_(params["sigma_color"].as<float>())
    , contrast_factor_(params["contrast_factor"].as<float>())
    , downsample_factor_(params["downsample_factor"].as<int>())
    , output_bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , use_eigen_(true) // Use Eigen by default
    , has_eigen_input_(false)
    , current_backend_(HDRBackend::AUTO)
    , opencv_opencl_available_(false)
    , halide_available_(false)
    , simd_available_(false)
    , last_execution_time_ms_(0.0)
{
    initializeBackends();
    if (current_backend_ == HDRBackend::AUTO) {
        current_backend_ = selectBestBackend();
    }
}

HDRDurandToneMappingHybrid::HDRDurandToneMappingHybrid(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
                                                       const YAML::Node& sensor_info, const YAML::Node& params)
    : eigen_img_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , params_(params)
    , is_enable_(params["is_enable"].as<bool>())
    , is_save_(params["is_save"].as<bool>())
    , is_debug_(params["is_debug"].as<bool>())
    , sigma_space_(params["sigma_space"].as<float>())
    , sigma_color_(params["sigma_color"].as<float>())
    , contrast_factor_(params["contrast_factor"].as<float>())
    , downsample_factor_(params["downsample_factor"].as<int>())
    , output_bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , use_eigen_(true) // Use Eigen by default
    , has_eigen_input_(true)
    , current_backend_(HDRBackend::AUTO)
    , opencv_opencl_available_(false)
    , halide_available_(false)
    , simd_available_(false)
    , last_execution_time_ms_(0.0)
{
    initializeBackends();
    if (current_backend_ == HDRBackend::AUTO) {
        current_backend_ = selectBestBackend();
    }
}

bool HDRDurandToneMappingHybrid::initializeBackends() {
    // Check OpenCV OpenCL availability
    opencv_opencl_available_ = cv::ocl::haveOpenCL();
    
    // Check Halide availability
#ifdef USE_HYBRID_BACKEND
    halide_available_ = true;
#else
    halide_available_ = false;
#endif
    
    // Check SIMD availability (simplified for Windows)
    simd_available_ = true; // Assume available for now
    
    return true;
}

HDRBackend HDRDurandToneMappingHybrid::selectBestBackend() const {
    // Priority order: Halide CPU > OpenCV OpenCL > SIMD > OpenCV CPU
    if (halide_available_) {
        return HDRBackend::HALIDE_CPU;
    } else if (opencv_opencl_available_) {
        return HDRBackend::OPENCV_OPENCL;
    } else if (simd_available_) {
        return HDRBackend::SIMD_OPTIMIZED;
    } else {
        return HDRBackend::OPENCV_CPU;
    }
}

void HDRDurandToneMappingHybrid::setBackend(HDRBackend backend) {
    if (backend == HDRBackend::AUTO) {
        backend = selectBestBackend();
    }
    
    if (!isBackendAvailable(backend)) {
        std::cerr << "HDR Hybrid - Backend not available: " << static_cast<int>(backend) << std::endl;
        return;
    }
    
    current_backend_ = backend;
}

bool HDRDurandToneMappingHybrid::isBackendAvailable(HDRBackend backend) const {
    switch (backend) {
        case HDRBackend::OPENCV_CPU:
            return true; // Always available
        case HDRBackend::OPENCV_OPENCL:
            return opencv_opencl_available_;
        case HDRBackend::HALIDE_CPU:
            return halide_available_;
        case HDRBackend::HALIDE_OPENCL:
            return halide_available_; // Simplified for now
        case HDRBackend::SIMD_OPTIMIZED:
            return simd_available_;
        case HDRBackend::AUTO:
            return true;
        default:
            return false;
    }
}

cv::Mat HDRDurandToneMappingHybrid::execute() {
    if (!is_enable_) {
        return img_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat result;
    
    switch (current_backend_) {
        case HDRBackend::OPENCV_CPU:
            result = execute_opencv_cpu();
            break;
        case HDRBackend::OPENCV_OPENCL:
            result = execute_opencv_opencl();
            break;
        case HDRBackend::HALIDE_CPU:
            result = execute_halide_cpu();
            break;
        case HDRBackend::HALIDE_OPENCL:
            result = execute_halide_opencl();
            break;
        case HDRBackend::SIMD_OPTIMIZED:
            result = execute_simd_optimized();
            break;
        default:
            result = execute_opencv_cpu(); // Fallback
            break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    last_execution_time_ms_ = duration.count();

    if (is_save_) {
        save();
    }

    return result;
}

hdr_isp::EigenImageU32 HDRDurandToneMappingHybrid::execute_eigen() {
    if (!is_enable_) {
        return eigen_img_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    hdr_isp::EigenImageU32 result;
    
    switch (current_backend_) {
        case HDRBackend::OPENCV_CPU:
            result = execute_eigen_opencv_cpu();
            break;
        case HDRBackend::OPENCV_OPENCL:
            result = execute_eigen_opencv_opencl();
            break;
        case HDRBackend::HALIDE_CPU:
            result = execute_eigen_halide_cpu();
            break;
        case HDRBackend::HALIDE_OPENCL:
            result = execute_eigen_halide_opencl();
            break;
        case HDRBackend::SIMD_OPTIMIZED:
            result = execute_eigen_simd_optimized();
            break;
        default:
            result = execute_eigen_opencv_cpu(); // Fallback
            break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    last_execution_time_ms_ = duration.count();
    
    return result;
}

// OpenCV CPU Implementation
cv::Mat HDRDurandToneMappingHybrid::execute_opencv_cpu() {
    // For now, return the input image unchanged
    // This would be replaced with the actual Durand algorithm implementation
    return img_;
}

hdr_isp::EigenImageU32 HDRDurandToneMappingHybrid::execute_eigen_opencv_cpu() {
    // For now, return the input image unchanged
    // This would be replaced with the actual Durand algorithm implementation
    return eigen_img_;
}

// OpenCV OpenCL Implementation
cv::Mat HDRDurandToneMappingHybrid::execute_opencv_opencl() {
    // For now, fall back to CPU implementation
    return execute_opencv_cpu();
}

hdr_isp::EigenImageU32 HDRDurandToneMappingHybrid::execute_eigen_opencv_opencl() {
    // For now, fall back to CPU implementation
    return execute_eigen_opencv_cpu();
}

// SIMD Optimized Implementation
cv::Mat HDRDurandToneMappingHybrid::execute_simd_optimized() {
    // For now, fall back to CPU implementation
    return execute_opencv_cpu();
}

hdr_isp::EigenImageU32 HDRDurandToneMappingHybrid::execute_eigen_simd_optimized() {
    // For now, fall back to CPU implementation
    return execute_eigen_opencv_cpu();
}

// OpenCV OpenCL specific functions (simplified)
cv::Mat HDRDurandToneMappingHybrid::tone_mapping_opencv_opencl(const cv::Mat& input) {
    // Simplified implementation - just return input for now
    return input;
}

cv::Mat HDRDurandToneMappingHybrid::bilateral_filter_opencv_opencl(const cv::Mat& input, float sigma_color, float sigma_space) {
    // Simplified implementation - just return input for now
    return input;
}

// SIMD-optimized functions (simplified)
cv::Mat HDRDurandToneMappingHybrid::tone_mapping_simd_opencv(const cv::Mat& input) {
    // Simplified implementation - just return input for now
    return input;
}

cv::Mat HDRDurandToneMappingHybrid::bilateral_filter_simd(const cv::Mat& input, float sigma_color, float sigma_space) {
    // Simplified implementation - just return input for now
    return input;
}

// SIMD vectorized mathematical operations (simplified)
void HDRDurandToneMappingHybrid::log_domain_conversion_simd(const cv::Mat& input, cv::Mat& output) {
    // Simplified implementation
    output = input.clone();
}

void HDRDurandToneMappingHybrid::exponential_conversion_simd(const cv::Mat& input, cv::Mat& output) {
    // Simplified implementation
    output = input.clone();
}

void HDRDurandToneMappingHybrid::bilateral_filter_simd_impl(const cv::Mat& input, cv::Mat& output, float sigma_color, float sigma_space) {
    // Simplified implementation
    output = input.clone();
}

// Halide implementations (if available)
#ifdef USE_HYBRID_BACKEND

cv::Mat HDRDurandToneMappingHybrid::execute_halide_cpu() {
    if (is_debug_) {
        std::cout << "HDR Hybrid - Using Halide CPU backend" << std::endl;
    }
    
    // Convert OpenCV to Halide
    Halide::Buffer<float> input_buffer = cv_mat_to_halide(img_);
    
    // Apply Halide Durand algorithm
    Halide::Buffer<float> output_buffer = tone_mapping_halide_cpu(input_buffer);
    
    // Convert back to OpenCV
    cv::Mat result = halide_to_cv_mat(output_buffer, img_.rows, img_.cols);
    
    return result;
}

hdr_isp::EigenImageU32 HDRDurandToneMappingHybrid::execute_eigen_halide_cpu() {
    if (is_debug_) {
        std::cout << "HDR Hybrid - Using Halide CPU backend (Eigen)" << std::endl;
    }
    
    // Convert Eigen to Halide
    Halide::Buffer<float> input_buffer = eigen_to_halide(eigen_img_);
    
    // Apply Halide Durand algorithm
    Halide::Buffer<float> output_buffer = tone_mapping_halide_cpu(input_buffer);
    
    // Convert back to Eigen
    hdr_isp::EigenImage result = halide_to_eigen(output_buffer, eigen_img_.rows(), eigen_img_.cols());
    
    // Convert to EigenImageU32
    hdr_isp::EigenImageU32 result_u32(result.rows(), result.cols());
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result_u32.data()(i, j) = static_cast<uint32_t>(std::max(0.0f, result(i, j)));
        }
    }
    
    return result_u32;
}

cv::Mat HDRDurandToneMappingHybrid::execute_halide_opencl() {
    // For now, fall back to CPU implementation
    return execute_halide_cpu();
}

hdr_isp::EigenImageU32 HDRDurandToneMappingHybrid::execute_eigen_halide_opencl() {
    // For now, fall back to CPU implementation
    return execute_eigen_halide_cpu();
}

// Halide-specific implementations with actual Durand algorithm
Halide::Buffer<float> HDRDurandToneMappingHybrid::tone_mapping_halide_cpu(const Halide::Buffer<float>& input) {
    int width = input.width();
    int height = input.height();
    
    // Create output buffer
    Halide::Buffer<float> output(width, height);
    
    // Apply Durand algorithm steps
    Halide::Buffer<float> log_luminance = log_domain_conversion_halide(input);
    Halide::Buffer<float> base_layer = bilateral_filter_halide(log_luminance, sigma_color_, sigma_space_);
    Halide::Buffer<float> detail_layer = log_luminance - base_layer;
    Halide::Buffer<float> compressed_base = base_layer / contrast_factor_;
    Halide::Buffer<float> log_output = compressed_base + detail_layer;
    output = exponential_conversion_halide(log_output);
    
    return output;
}

Halide::Buffer<float> HDRDurandToneMappingHybrid::tone_mapping_halide_opencl(const Halide::Buffer<float>& input) {
    // For now, fall back to CPU implementation
    return tone_mapping_halide_cpu(input);
}

Halide::Buffer<float> HDRDurandToneMappingHybrid::bilateral_filter_halide(const Halide::Buffer<float>& input, float sigma_color, float sigma_space) {
    int width = input.width();
    int height = input.height();
    
    // Create output buffer
    Halide::Buffer<float> output(width, height);
    
    // Apply bilateral filtering using Halide
    Halide::Func bilateral_func = bilateral_filter_halide_func(input, sigma_color, sigma_space);
    bilateral_func.realize(output);
    
    return output;
}

// Durand algorithm Halide functions with actual implementation
Halide::Func HDRDurandToneMappingHybrid::apply_durand_algorithm_halide(const Halide::Buffer<float>& input) {
    Halide::Var x, y;
    
    // Convert to log domain
    Halide::Func log_luminance;
    log_luminance(x, y) = Halide::log(input(x, y) + 1e-6f) / std::log(10.0f);
    
    // Apply bilateral filter for base layer
    Halide::Func base_layer = bilateral_filter_halide_func(log_luminance, sigma_color_, sigma_space_);
    
    // Extract detail layer
    Halide::Func detail_layer;
    detail_layer(x, y) = log_luminance(x, y) - base_layer(x, y);
    
    // Compress base layer
    Halide::Func compressed_base;
    compressed_base(x, y) = base_layer(x, y) / contrast_factor_;
    
    // Recombine layers
    Halide::Func log_output;
    log_output(x, y) = compressed_base(x, y) + detail_layer(x, y);
    
    // Convert back from log domain
    Halide::Func result;
    result(x, y) = Halide::exp(log_output(x, y) * std::log(10.0f));
    
    return result;
}

Halide::Func HDRDurandToneMappingHybrid::log_domain_conversion_halide(const Halide::Buffer<float>& input) {
    Halide::Var x, y;
    Halide::Func result;
    result(x, y) = Halide::log(input(x, y) + 1e-6f) / std::log(10.0f);
    return result;
}

Halide::Func HDRDurandToneMappingHybrid::bilateral_filter_halide_func(const Halide::Buffer<float>& input, float sigma_color, float sigma_space) {
    Halide::Var x, y;
    
    // Simplified bilateral filter implementation
    // In a full implementation, this would include proper spatial and range filtering
    
    // For now, use a simple Gaussian approximation
    Halide::Func result;
    Halide::RDom r(-2, 5, -2, 5); // 5x5 kernel
    
    // Gaussian weights
    Halide::Expr weight = Halide::exp(-(r.x * r.x + r.y * r.y) / (2 * sigma_space * sigma_space));
    
    // Apply weighted average
    Halide::Expr sum = 0.0f;
    Halide::Expr total_weight = 0.0f;
    
    sum += weight * input(Halide::clamp(x + r.x, 0, input.width() - 1), 
                         Halide::clamp(y + r.y, 0, input.height() - 1));
    total_weight += weight;
    
    result(x, y) = sum / total_weight;
    
    return result;
}

Halide::Func HDRDurandToneMappingHybrid::contrast_compression_halide(const Halide::Buffer<float>& base_layer, float contrast_factor) {
    Halide::Var x, y;
    Halide::Func result;
    result(x, y) = base_layer(x, y) / contrast_factor;
    return result;
}

Halide::Func HDRDurandToneMappingHybrid::detail_preservation_halide(const Halide::Buffer<float>& log_luminance, const Halide::Buffer<float>& base_layer) {
    Halide::Var x, y;
    Halide::Func result;
    result(x, y) = log_luminance(x, y) - base_layer(x, y);
    return result;
}

Halide::Func HDRDurandToneMappingHybrid::exponential_conversion_halide(const Halide::Buffer<float>& log_output) {
    Halide::Var x, y;
    Halide::Func result;
    result(x, y) = Halide::exp(log_output(x, y) * std::log(10.0f));
    return result;
}

// Utility functions for Halide (simplified)
Halide::Buffer<float> HDRDurandToneMappingHybrid::cv_mat_to_halide(const cv::Mat& cv_mat) {
    int rows = cv_mat.rows;
    int cols = cv_mat.cols;
    
    Halide::Buffer<float> buffer(cols, rows);
    
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            buffer(x, y) = cv_mat.at<float>(y, x);
        }
    }
    
    return buffer;
}

Halide::Buffer<float> HDRDurandToneMappingHybrid::eigen_to_halide(const hdr_isp::EigenImage& eigen_img) {
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    
    Halide::Buffer<float> buffer(cols, rows);
    
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            buffer(x, y) = eigen_img(y, x);
        }
    }
    
    return buffer;
}

cv::Mat HDRDurandToneMappingHybrid::halide_to_cv_mat(const Halide::Buffer<float>& buffer, int rows, int cols) {
    cv::Mat cv_mat(rows, cols, CV_32F);
    
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            cv_mat.at<float>(y, x) = buffer(x, y);
        }
    }
    
    return cv_mat;
}

hdr_isp::EigenImage HDRDurandToneMappingHybrid::halide_to_eigen(const Halide::Buffer<float>& buffer, int rows, int cols) {
    hdr_isp::EigenImage eigen_img(rows, cols);
    
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            eigen_img(y, x) = buffer(x, y);
        }
    }
    
    return eigen_img;
}

#else

// Stub implementations when Halide is not available
cv::Mat HDRDurandToneMappingHybrid::execute_halide_cpu() {
    return execute_opencv_cpu(); // Fallback
}

hdr_isp::EigenImageU32 HDRDurandToneMappingHybrid::execute_eigen_halide_cpu() {
    return execute_eigen_opencv_cpu(); // Fallback
}

cv::Mat HDRDurandToneMappingHybrid::execute_halide_opencl() {
    return execute_opencv_cpu(); // Fallback
}

hdr_isp::EigenImageU32 HDRDurandToneMappingHybrid::execute_eigen_halide_opencl() {
    return execute_eigen_opencv_cpu(); // Fallback
}

#endif

void HDRDurandToneMappingHybrid::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_hdr_durand_hybrid_" + 
                                 std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + ".png";
        cv::imwrite(output_path, img_);
    }
} 
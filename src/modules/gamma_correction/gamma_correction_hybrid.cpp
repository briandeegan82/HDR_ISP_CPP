#include "gamma_correction_hybrid.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

GammaCorrectionHybrid::GammaCorrectionHybrid(const hdr_isp::EigenImage3C& img, const YAML::Node& platform,
                                           const YAML::Node& sensor_info, const YAML::Node& parm_gmm)
    : img_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_gmm_(parm_gmm)
    , enable_(parm_gmm["is_enable"] ? parm_gmm["is_enable"].as<bool>() : false)
    , output_bit_depth_(sensor_info["output_bit_depth"] ? sensor_info["output_bit_depth"].as<int>() : 8)
    , is_save_(parm_gmm["is_save"] ? parm_gmm["is_save"].as<bool>() : false)
    , is_debug_(parm_gmm["is_debug"] ? parm_gmm["is_debug"].as<bool>() : false)
    , current_backend_(GammaBackend::AUTO)
    , opencv_opencl_available_(false)
    , halide_available_(false)
    , simd_available_(false)
    , last_execution_time_ms_(0.0)
{
    initializeBackends();
    if (current_backend_ == GammaBackend::AUTO) {
        current_backend_ = selectBestBackend();
    }
}

bool GammaCorrectionHybrid::initializeBackends() {
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

GammaBackend GammaCorrectionHybrid::selectBestBackend() const {
    // Priority order: Halide CPU > OpenCV OpenCL > SIMD > OpenCV CPU
    if (halide_available_) {
        return GammaBackend::HALIDE_CPU;
    } else if (opencv_opencl_available_) {
        return GammaBackend::OPENCV_OPENCL;
    } else if (simd_available_) {
        return GammaBackend::SIMD_OPTIMIZED;
    } else {
        return GammaBackend::OPENCV_CPU;
    }
}

void GammaCorrectionHybrid::setBackend(GammaBackend backend) {
    if (backend == GammaBackend::AUTO) {
        backend = selectBestBackend();
    }
    
    if (!isBackendAvailable(backend)) {
        std::cerr << "Gamma Hybrid - Backend not available: " << static_cast<int>(backend) << std::endl;
        return;
    }
    
    current_backend_ = backend;
}

bool GammaCorrectionHybrid::isBackendAvailable(GammaBackend backend) const {
    switch (backend) {
        case GammaBackend::OPENCV_CPU:
            return true; // Always available
        case GammaBackend::OPENCV_OPENCL:
            return opencv_opencl_available_;
        case GammaBackend::HALIDE_CPU:
            return halide_available_;
        case GammaBackend::HALIDE_OPENCL:
            return halide_available_; // Simplified for now
        case GammaBackend::SIMD_OPTIMIZED:
            return simd_available_;
        case GammaBackend::AUTO:
            return true;
        default:
            return false;
    }
}

hdr_isp::EigenImage3C GammaCorrectionHybrid::execute() {
    if (!enable_) {
        return img_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    hdr_isp::EigenImage3C result;
    
    switch (current_backend_) {
        case GammaBackend::OPENCV_CPU:
            result = execute_opencv_cpu();
            break;
        case GammaBackend::OPENCV_OPENCL:
            result = execute_opencv_opencl();
            break;
        case GammaBackend::HALIDE_CPU:
            result = execute_halide_cpu();
            break;
        case GammaBackend::HALIDE_OPENCL:
            result = execute_halide_opencl();
            break;
        case GammaBackend::SIMD_OPTIMIZED:
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

// OpenCV CPU Implementation
hdr_isp::EigenImage3C GammaCorrectionHybrid::execute_opencv_cpu() {
    if (is_debug_) {
        std::cout << "Gamma Hybrid - Using OpenCV CPU backend" << std::endl;
    }
    
    // Convert Eigen to OpenCV
    cv::Mat input_img = img_.toOpenCV(CV_32FC3);
    
    // Normalize input to expected range [0, max_val]
    int max_val = (1 << output_bit_depth_) - 1;
    double min_val, max_val_cv;
    cv::minMaxLoc(input_img, &min_val, &max_val_cv);
    
    if (is_debug_) {
        std::cout << "Gamma Hybrid - Normalizing from range [" << min_val << ", " << max_val_cv << "] to [0, " << max_val << "]" << std::endl;
    }
    
    // Apply normalization if needed
    if (max_val_cv > 0) {
        float scale_factor = static_cast<float>(max_val) / max_val_cv;
        input_img = input_img * scale_factor;
        
        if (is_debug_) {
            std::cout << "Gamma Hybrid - Applied scale factor: " << scale_factor << std::endl;
        }
    }
    
    // Generate gamma LUT
    auto lut = generate_gamma_lut(output_bit_depth_);
    
    // Apply gamma correction using LUT
    cv::Mat output_img = input_img.clone();
    
    // Apply LUT to each channel
    for (int i = 0; i < output_img.rows; ++i) {
        for (int j = 0; j < output_img.cols; ++j) {
            cv::Vec3f& pixel = output_img.at<cv::Vec3f>(i, j);
            
            // Apply to each channel
            for (int c = 0; c < 3; ++c) {
                int val = static_cast<int>(std::round(pixel[c]));
                val = std::max(0, std::min(max_val, val)); // Clamp to valid range
                if (val < static_cast<int>(lut.size())) {
                    pixel[c] = static_cast<float>(lut[val]);
                }
            }
        }
    }
    
    // Convert back to Eigen
    hdr_isp::EigenImage3C result = hdr_isp::EigenImage3C::fromOpenCV(output_img);
    
    return result;
}

// OpenCV OpenCL Implementation
hdr_isp::EigenImage3C GammaCorrectionHybrid::execute_opencv_opencl() {
    // For now, fall back to CPU implementation
    return execute_opencv_cpu();
}

// SIMD Optimized Implementation
hdr_isp::EigenImage3C GammaCorrectionHybrid::execute_simd_optimized() {
    // For now, fall back to CPU implementation
    return execute_opencv_cpu();
}

// SIMD-optimized functions (simplified)
hdr_isp::EigenImage3C GammaCorrectionHybrid::apply_gamma_simd_opencv(const hdr_isp::EigenImage3C& input) {
    // Simplified implementation - just return input for now
    return input;
}

void GammaCorrectionHybrid::apply_gamma_lut_simd(const hdr_isp::EigenImage3C& input, hdr_isp::EigenImage3C& output, const std::vector<uint32_t>& lut) {
    // Simplified implementation
    output = input;
}

// Halide implementations (if available)
#ifdef USE_HYBRID_BACKEND

hdr_isp::EigenImage3C GammaCorrectionHybrid::execute_halide_cpu() {
    if (is_debug_) {
        std::cout << "Gamma Hybrid - Using Halide CPU backend" << std::endl;
    }
    
    // Convert Eigen to Halide
    Halide::Buffer<float> input_buffer = eigen_to_halide_3c(img_);
    
    // Apply Halide gamma correction
    Halide::Buffer<float> output_buffer = apply_gamma_halide_cpu(input_buffer);
    
    // Convert back to Eigen
    hdr_isp::EigenImage3C result = halide_to_eigen_3c(output_buffer, img_.rows(), img_.cols());
    
    return result;
}

hdr_isp::EigenImage3C GammaCorrectionHybrid::execute_halide_opencl() {
    // For now, fall back to CPU implementation
    return execute_halide_cpu();
}

// Halide-specific implementations with actual gamma correction
Halide::Buffer<float> GammaCorrectionHybrid::apply_gamma_halide_cpu(const Halide::Buffer<float>& input) {
    int width = input.width();
    int height = input.height();
    int channels = input.channels();
    
    // Create output buffer
    Halide::Buffer<float> output(width, height, channels);
    
    // Generate gamma LUT
    Halide::Buffer<uint32_t> lut = generate_gamma_lut_halide(output_bit_depth_);
    
    // Apply gamma correction using Halide
    Halide::Func gamma_func = apply_gamma_lut_halide(input, lut);
    gamma_func.realize(output);
    
    return output;
}

Halide::Buffer<float> GammaCorrectionHybrid::apply_gamma_halide_opencl(const Halide::Buffer<float>& input) {
    // For now, fall back to CPU implementation
    return apply_gamma_halide_cpu(input);
}

// Gamma LUT generation and application with Halide
Halide::Buffer<uint32_t> GammaCorrectionHybrid::generate_gamma_lut_halide(int bit_depth) {
    // Safety check for reasonable bit depth
    if (bit_depth <= 0 || bit_depth > 16) {
        bit_depth = std::min(16, std::max(1, bit_depth));
    }
    
    int max_val = (1 << bit_depth) - 1;
    Halide::Buffer<uint32_t> lut(max_val + 1);
    
    // Generate LUT using Halide's vectorized operations
    Halide::Var i;
    Halide::Func lut_func;
    
    // Apply gamma correction: output = max_val * (input/max_val)^(1/2.2)
    Halide::Expr input_normalized = Halide::cast<float>(i) / max_val;
    Halide::Expr gamma_corrected = Halide::pow(input_normalized, 1.0f / 2.2f);
    Halide::Expr output = Halide::cast<uint32_t>(Halide::round(max_val * gamma_corrected));
    
    lut_func(i) = output;
    
    // Schedule for optimal performance
    lut_func.vectorize(i, 8);
    
    // Realize the LUT
    lut_func.realize(lut);
    
    return lut;
}

Halide::Func GammaCorrectionHybrid::apply_gamma_lut_halide(const Halide::Buffer<float>& input, const Halide::Buffer<uint32_t>& lut) {
    Halide::Var x, y, c;
    
    // Apply LUT to each channel
    Halide::Func result;
    
    // Convert input to integer index for LUT lookup
    Halide::Expr input_val = input(x, y, c);
    Halide::Expr index = Halide::cast<int>(Halide::clamp(input_val, 0.0f, static_cast<float>(lut.width() - 1)));
    
    // Look up gamma-corrected value
    Halide::Expr gamma_corrected = Halide::cast<float>(lut(index));
    
    result(x, y, c) = gamma_corrected;
    
    // Schedule for optimal performance
    result.vectorize(x, 8).parallel(y);
    
    return result;
}

// Utility functions for data conversion
Halide::Buffer<float> GammaCorrectionHybrid::eigen_to_halide_3c(const hdr_isp::EigenImage3C& eigen_img) {
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    
    // Create Halide buffer with interleaved RGB layout
    Halide::Buffer<float> buffer(cols, rows, 3);
    
    // Copy data in interleaved format for better vectorization
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            buffer(x, y, 0) = eigen_img.r().data()(y, x);  // R channel
            buffer(x, y, 1) = eigen_img.g().data()(y, x);  // G channel
            buffer(x, y, 2) = eigen_img.b().data()(y, x);  // B channel
        }
    }
    
    return buffer;
}

hdr_isp::EigenImage3C GammaCorrectionHybrid::halide_to_eigen_3c(const Halide::Buffer<float>& buffer, int rows, int cols) {
    // Create Eigen image
    hdr_isp::EigenImage3C result(rows, cols);
    
    // Copy data back from interleaved format
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            result.r().data()(y, x) = buffer(x, y, 0);  // R channel
            result.g().data()(y, x) = buffer(x, y, 1);  // G channel
            result.b().data()(y, x) = buffer(x, y, 2);  // B channel
        }
    }
    
    return result;
}

#else

// Stub implementations when Halide is not available
hdr_isp::EigenImage3C GammaCorrectionHybrid::execute_halide_cpu() {
    return execute_opencv_cpu(); // Fallback
}

hdr_isp::EigenImage3C GammaCorrectionHybrid::execute_halide_opencl() {
    return execute_opencv_cpu(); // Fallback
}

#endif

void GammaCorrectionHybrid::save() {
    if (is_save_) {
        fs::path output_dir = "out_frames";
        fs::create_directories(output_dir);
        
        // Try to get filename from platform config
        std::string in_file;
        if (platform_["in_file"]) {
            in_file = platform_["in_file"].as<std::string>();
        } else if (platform_["filename"]) {
            in_file = platform_["filename"].as<std::string>();
        } else {
            in_file = "unknown";
        }
        
        std::string bayer_pattern = sensor_info_["bayer_pattern"].as<std::string>();
        fs::path output_path = output_dir / ("Out_gamma_correction_hybrid_" + in_file + "_" + bayer_pattern + ".png");
        
        // Convert to OpenCV for saving
        cv::Mat opencv_img = img_.toOpenCV(CV_32FC3);
        cv::imwrite(output_path.string(), opencv_img);
    }
} 
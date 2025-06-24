#include "color_correction_matrix_hybrid.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

ColorCorrectionMatrixHybrid::ColorCorrectionMatrixHybrid(const hdr_isp::EigenImage3C& img, const YAML::Node& sensor_info, 
                                                         const YAML::Node& parm_ccm, const hdr_isp::FixedPointConfig& fp_config)
    : ColorCorrectionMatrix(img, sensor_info, parm_ccm, fp_config)
    , is_debug_(parm_ccm["is_debug"].as<bool>())
    , is_save_(parm_ccm["is_save"].as<bool>())
{
    if (is_debug_) {
        std::cout << "CCM Hybrid - Constructor started" << std::endl;
        std::cout << "CCM Hybrid - Input image size: " << img.rows() << "x" << img.cols() << std::endl;
        std::cout << "CCM Hybrid - Enable: " << (enable_ ? "true" : "false") << std::endl;
    }
}

ColorCorrectionMatrixHybrid::ColorCorrectionMatrixHybrid(const hdr_isp::EigenImage3CFixed& img, const YAML::Node& sensor_info, 
                                                         const YAML::Node& parm_ccm, const hdr_isp::FixedPointConfig& fp_config)
    : ColorCorrectionMatrix(img, sensor_info, parm_ccm, fp_config)
    , is_debug_(parm_ccm["is_debug"].as<bool>())
    , is_save_(parm_ccm["is_save"].as<bool>())
{
    if (is_debug_) {
        std::cout << "CCM Hybrid - Constructor started" << std::endl;
        std::cout << "CCM Hybrid - Input image size: " << img.rows() << "x" << img.cols() << std::endl;
        std::cout << "CCM Hybrid - Enable: " << (enable_ ? "true" : "false") << std::endl;
    }
}

Halide::Buffer<float> ColorCorrectionMatrixHybrid::apply_ccm_halide(const Halide::Buffer<float>& input) {
    int width = input.width();
    int height = input.height();
    
    if (is_debug_) {
        std::cout << "CCM Hybrid - apply_ccm_halide() started" << std::endl;
        std::cout << "CCM Hybrid - Input size: " << width << "x" << height << "x3" << std::endl;
    }
    
    Halide::Var x, y, c;  // x, y coordinates, c for RGB channels
    
    // Pre-computed CCM matrix as Halide constants
    Halide::Expr ccm[3][3] = {
        {ccm_mat_(0,0), ccm_mat_(0,1), ccm_mat_(0,2)},
        {ccm_mat_(1,0), ccm_mat_(1,1), ccm_mat_(1,2)},
        {ccm_mat_(2,0), ccm_mat_(2,1), ccm_mat_(2,2)}
    };
    
    // Vectorized matrix multiplication
    Halide::Func ccm_result;
    ccm_result(x, y, c) = 
        ccm[c][0] * input(x, y, 0) +  // R component
        ccm[c][1] * input(x, y, 1) +  // G component  
        ccm[c][2] * input(x, y, 2);   // B component
    
    // Optimized scheduling for floating-point
    ccm_result.vectorize(x, 8).parallel(y).reorder(c, x, y);
    
    // Realize the result
    Halide::Buffer<float> output = ccm_result.realize({width, height, 3});
    
    if (is_debug_) {
        float min_val = output.min();
        float max_val = output.max();
        std::cout << "CCM Hybrid - Output - Min: " << min_val << ", Max: " << max_val << std::endl;
    }
    
    return output;
}

Halide::Buffer<int16_t> ColorCorrectionMatrixHybrid::apply_ccm_fixed_halide(const Halide::Buffer<int16_t>& input) {
    int width = input.width();
    int height = input.height();
    
    if (is_debug_) {
        std::cout << "CCM Hybrid - apply_ccm_fixed_halide() started" << std::endl;
        std::cout << "CCM Hybrid - Input size: " << width << "x" << height << "x3" << std::endl;
        std::cout << "CCM Hybrid - Fractional bits: " << fractional_bits_ << std::endl;
    }
    
    Halide::Var x, y, c;
    
    // Pre-computed fixed-point CCM matrix
    Halide::Expr ccm_fixed[3][3] = {
        {ccm_mat_8bit_(0,0), ccm_mat_8bit_(0,1), ccm_mat_8bit_(0,2)},
        {ccm_mat_8bit_(1,0), ccm_mat_8bit_(1,1), ccm_mat_8bit_(1,2)},
        {ccm_mat_8bit_(2,0), ccm_mat_8bit_(2,1), ccm_mat_8bit_(2,2)}
    };
    
    // Fixed-point matrix multiplication with proper scaling
    Halide::Func ccm_fixed_result;
    ccm_fixed_result(x, y, c) = Halide::cast<int16_t>(
        (ccm_fixed[c][0] * input(x, y, 0) +
         ccm_fixed[c][1] * input(x, y, 1) +
         ccm_fixed[c][2] * input(x, y, 2) + half_scale_32bit_) >> fractional_bits_
    );
    
    // Optimized scheduling for fixed-point (higher vectorization width)
    ccm_fixed_result.vectorize(x, 16).parallel(y).reorder(c, x, y);
    
    // Realize the result
    Halide::Buffer<int16_t> output = ccm_fixed_result.realize({width, height, 3});
    
    if (is_debug_) {
        int16_t min_val = output.min();
        int16_t max_val = output.max();
        std::cout << "CCM Hybrid Fixed - Output - Min: " << min_val << ", Max: " << max_val << std::endl;
    }
    
    return output;
}

Halide::Buffer<float> ColorCorrectionMatrixHybrid::apply_ccm_vectorized_halide(const Halide::Buffer<float>& input) {
    // This is an alternative implementation with more aggressive vectorization
    int width = input.width();
    int height = input.height();
    
    if (is_debug_) {
        std::cout << "CCM Hybrid - apply_ccm_vectorized_halide() started" << std::endl;
        std::cout << "CCM Hybrid - Input size: " << width << "x" << height << "x3" << std::endl;
    }
    
    Halide::Var x, y, c;
    
    // Pre-computed CCM matrix
    Halide::Expr ccm[3][3] = {
        {ccm_mat_(0,0), ccm_mat_(0,1), ccm_mat_(0,2)},
        {ccm_mat_(1,0), ccm_mat_(1,1), ccm_mat_(1,2)},
        {ccm_mat_(2,0), ccm_mat_(2,1), ccm_mat_(2,2)}
    };
    
    // More aggressive vectorization with tiling
    Halide::Func ccm_vectorized;
    ccm_vectorized(x, y, c) = 
        ccm[c][0] * input(x, y, 0) + 
        ccm[c][1] * input(x, y, 1) + 
        ccm[c][2] * input(x, y, 2);
    
    // Advanced scheduling with tiling for better cache performance
    Halide::Var xo, yo, xi, yi;
    ccm_vectorized.tile(x, y, xo, yo, xi, yi, 32, 32)
                 .vectorize(xi, 8)
                 .parallel(yo)
                 .reorder(c, xi, yi, xo, yo);
    
    // Realize the result
    Halide::Buffer<float> output = ccm_vectorized.realize({width, height, 3});
    
    if (is_debug_) {
        float min_val = output.min();
        float max_val = output.max();
        std::cout << "CCM Hybrid Vectorized - Output - Min: " << min_val << ", Max: " << max_val << std::endl;
    }
    
    return output;
}

Halide::Buffer<float> ColorCorrectionMatrixHybrid::eigen_to_halide_float(const hdr_isp::EigenImage3C& eigen_img) {
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

hdr_isp::EigenImage3C ColorCorrectionMatrixHybrid::halide_to_eigen_float(const Halide::Buffer<float>& buffer, int rows, int cols) {
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

Halide::Buffer<int16_t> ColorCorrectionMatrixHybrid::eigen_to_halide_fixed(const hdr_isp::EigenImage3CFixed& eigen_img) {
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    
    // Create Halide buffer with interleaved RGB layout
    Halide::Buffer<int16_t> buffer(cols, rows, 3);
    
    // Copy data in interleaved format for better vectorization
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            buffer(x, y, 0) = eigen_img.r()(y, x);  // R channel
            buffer(x, y, 1) = eigen_img.g()(y, x);  // G channel
            buffer(x, y, 2) = eigen_img.b()(y, x);  // B channel
        }
    }
    
    return buffer;
}

hdr_isp::EigenImage3CFixed ColorCorrectionMatrixHybrid::halide_to_eigen_fixed(const Halide::Buffer<int16_t>& buffer, int rows, int cols) {
    // Create Eigen fixed-point image
    hdr_isp::EigenImage3CFixed result(rows, cols);
    
    // Copy data back from interleaved format
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            result.r()(y, x) = buffer(x, y, 0);  // R channel
            result.g()(y, x) = buffer(x, y, 1);  // G channel
            result.b()(y, x) = buffer(x, y, 2);  // B channel
        }
    }
    
    return result;
}

void ColorCorrectionMatrixHybrid::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_ccm_hybrid_" + 
                                 std::to_string(raw_.cols) + "x" + std::to_string(raw_.rows) + ".png";
        cv::Mat temp_img = raw_.toOpenCV(CV_32FC3);
        cv::imwrite(output_path, temp_img);
    }
}

hdr_isp::EigenImage3C ColorCorrectionMatrixHybrid::execute() {
    if (!enable_) {
        return raw_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    if (is_debug_) {
        std::cout << "CCM Hybrid - execute() started" << std::endl;
    }
    
    hdr_isp::EigenImage3C result;
    
    if (fp_config_.isEnabled()) {
        // Use fixed-point mode
        if (is_debug_) {
            std::cout << "CCM Hybrid - Using fixed-point mode" << std::endl;
        }
        
        // Convert Eigen to Halide fixed-point
        Halide::Buffer<int16_t> halide_input = eigen_to_halide_fixed(raw_fixed_);
        
        // Apply fixed-point CCM using Halide
        Halide::Buffer<int16_t> halide_output = apply_ccm_fixed_halide(halide_input);
        
        // Convert back to Eigen
        hdr_isp::EigenImage3CFixed result_fixed = halide_to_eigen_fixed(halide_output, raw_fixed_.rows(), raw_fixed_.cols());
        
        // Convert fixed-point result to floating-point
        result = hdr_isp::EigenImage3C::fromFixedPoint(result_fixed, fp_config_);
    } else {
        // Use floating-point mode
        if (is_debug_) {
            std::cout << "CCM Hybrid - Using floating-point mode" << std::endl;
        }
        
        // Convert Eigen to Halide float
        Halide::Buffer<float> halide_input = eigen_to_halide_float(raw_);
        
        // Apply CCM using Halide
        Halide::Buffer<float> halide_output = apply_ccm_halide(halide_input);
        
        // Convert back to Eigen
        result = halide_to_eigen_float(halide_output, raw_.rows(), raw_.cols());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "CCM Hybrid execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }

    // Save intermediate results if enabled
    if (is_save_) {
        save();
    }

    return result;
}

hdr_isp::EigenImage3CFixed ColorCorrectionMatrixHybrid::execute_fixed() {
    if (!enable_) {
        return raw_fixed_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    if (is_debug_) {
        std::cout << "CCM Hybrid - execute_fixed() started" << std::endl;
    }
    
    // Convert Eigen fixed-point to Halide
    Halide::Buffer<int16_t> halide_input = eigen_to_halide_fixed(raw_fixed_);
    
    // Apply fixed-point CCM using Halide
    Halide::Buffer<int16_t> halide_output = apply_ccm_fixed_halide(halide_input);
    
    // Convert back to Eigen fixed-point
    hdr_isp::EigenImage3CFixed result = halide_to_eigen_fixed(halide_output, raw_fixed_.rows(), raw_fixed_.cols());
    
    // Clip to valid range
    result = result.clip(-max_val_8bit_, max_val_8bit_);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "CCM Hybrid fixed execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }
    
    return result;
} 
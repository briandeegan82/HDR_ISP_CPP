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
}

ColorCorrectionMatrixHybrid::ColorCorrectionMatrixHybrid(const hdr_isp::EigenImage3CFixed& img, const YAML::Node& sensor_info, 
                                                         const YAML::Node& parm_ccm, const hdr_isp::FixedPointConfig& fp_config)
    : ColorCorrectionMatrix(img, sensor_info, parm_ccm, fp_config)
    , is_debug_(parm_ccm["is_debug"].as<bool>())
    , is_save_(parm_ccm["is_save"].as<bool>())
{
}

Halide::Buffer<float> ColorCorrectionMatrixHybrid::apply_ccm_halide(const Halide::Buffer<float>& input) {
    int width = input.width();
    int height = input.height();
    
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
    
    return output;
}

Halide::Buffer<int16_t> ColorCorrectionMatrixHybrid::apply_ccm_fixed_halide(const Halide::Buffer<int16_t>& input) {
    int width = input.width();
    int height = input.height();
    
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
    
    return output;
}

Halide::Buffer<float> ColorCorrectionMatrixHybrid::apply_ccm_vectorized_halide(const Halide::Buffer<float>& input) {
    // This is an alternative implementation with more aggressive vectorization
    int width = input.width();
    int height = input.height();
    
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
    
    return output;
}

// Utility functions for data conversion
Halide::Buffer<float> ColorCorrectionMatrixHybrid::eigen_to_halide_float(const hdr_isp::EigenImage3C& eigen_img) {
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    
    Halide::Buffer<float> buffer(cols, rows, 3);
    
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float value;
                switch (c) {
                    case 0: value = eigen_img.r()(i, j); break;
                    case 1: value = eigen_img.g()(i, j); break;
                    case 2: value = eigen_img.b()(i, j); break;
                }
                buffer(j, i, c) = value;
            }
        }
    }
    
    return buffer;
}

hdr_isp::EigenImage3C ColorCorrectionMatrixHybrid::halide_to_eigen_float(const Halide::Buffer<float>& buffer, int rows, int cols) {
    hdr_isp::EigenImage3C result(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.r()(i, j) = buffer(j, i, 0);
            result.g()(i, j) = buffer(j, i, 1);
            result.b()(i, j) = buffer(j, i, 2);
        }
    }
    
    return result;
}

Halide::Buffer<int16_t> ColorCorrectionMatrixHybrid::eigen_to_halide_fixed(const hdr_isp::EigenImage3CFixed& eigen_img) {
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    
    Halide::Buffer<int16_t> buffer(cols, rows, 3);
    
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int16_t value;
                switch (c) {
                    case 0: value = eigen_img.r()(i, j); break;
                    case 1: value = eigen_img.g()(i, j); break;
                    case 2: value = eigen_img.b()(i, j); break;
                }
                buffer(j, i, c) = value;
            }
        }
    }
    
    return buffer;
}

hdr_isp::EigenImage3CFixed ColorCorrectionMatrixHybrid::halide_to_eigen_fixed(const Halide::Buffer<int16_t>& buffer, int rows, int cols) {
    hdr_isp::EigenImage3CFixed result(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.r()(i, j) = buffer(j, i, 0);
            result.g()(i, j) = buffer(j, i, 1);
            result.b()(i, j) = buffer(j, i, 2);
        }
    }
    
    return result;
}

void ColorCorrectionMatrixHybrid::save() {
    if (!is_save_) return;
    
    // Save intermediate results if needed
    fs::path output_dir = fs::path(PROJECT_ROOT_DIR) / "out_frames" / "intermediate";
    fs::create_directories(output_dir);
    
    // Save floating-point result
    if (!use_fixed_input_) {
        fs::path output_path = output_dir / "color_correction_matrix_hybrid.png";
        cv::Mat opencv_img = raw_.toOpenCV(CV_32FC3);
        cv::imwrite(output_path.string(), opencv_img);
    }
}

hdr_isp::EigenImage3C ColorCorrectionMatrixHybrid::execute() {
    if (!enable_) {
        return raw_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    hdr_isp::EigenImage3C result;
    
    if (fp_config_.isEnabled()) {
        // Use fixed-point mode
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
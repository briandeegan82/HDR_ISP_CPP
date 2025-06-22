#include "color_correction_matrix.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <filesystem>
#ifdef _OPENMP
#include <omp.h>
#endif

ColorCorrectionMatrix::ColorCorrectionMatrix(const hdr_isp::EigenImage3C& img, const YAML::Node& sensor_info, const YAML::Node& parm_ccm, const hdr_isp::FixedPointConfig& fp_config)
    : raw_(img)
    , sensor_info_(sensor_info)
    , parm_ccm_(parm_ccm)
    , fp_config_(fp_config)
    , enable_(parm_ccm["is_enable"].as<bool>())
    , ccm_mat_(Eigen::Matrix3f::Identity())
    , is_save_(parm_ccm["is_save"].as<bool>())
    , use_fixed_input_(false)
    , matrices_initialized_(false)
    , fractional_bits_(fp_config.getFractionalBits())
{
    // Initialize CCM matrix
    std::vector<float> corrected_red = parm_ccm["corrected_red"].as<std::vector<float>>();
    std::vector<float> corrected_green = parm_ccm["corrected_green"].as<std::vector<float>>();
    std::vector<float> corrected_blue = parm_ccm["corrected_blue"].as<std::vector<float>>();

    ccm_mat_.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data());
    ccm_mat_.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data());
    ccm_mat_.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data());
    
    // Pre-compute fixed-point matrices and constants
    initialize_fixed_point_matrices();
}

ColorCorrectionMatrix::ColorCorrectionMatrix(const hdr_isp::EigenImage3CFixed& img, const YAML::Node& sensor_info, const YAML::Node& parm_ccm, const hdr_isp::FixedPointConfig& fp_config)
    : raw_fixed_(img)
    , sensor_info_(sensor_info)
    , parm_ccm_(parm_ccm)
    , fp_config_(fp_config)
    , enable_(parm_ccm["is_enable"].as<bool>())
    , ccm_mat_(Eigen::Matrix3f::Identity())
    , is_save_(parm_ccm["is_save"].as<bool>())
    , use_fixed_input_(true)
    , matrices_initialized_(false)
    , fractional_bits_(fp_config.getFractionalBits())
{
    std::cout << "CCM - Constructor started" << std::endl;
    std::cout << "CCM - Input image size: " << img.rows() << "x" << img.cols() << std::endl;
    std::cout << "CCM - Enable: " << (enable_ ? "true" : "false") << std::endl;
    
    // Initialize CCM matrix
    std::vector<float> corrected_red = parm_ccm["corrected_red"].as<std::vector<float>>();
    std::vector<float> corrected_green = parm_ccm["corrected_green"].as<std::vector<float>>();
    std::vector<float> corrected_blue = parm_ccm["corrected_blue"].as<std::vector<float>>();

    std::cout << "CCM - CCM matrix values:" << std::endl;
    std::cout << "CCM -   Red: [" << corrected_red[0] << ", " << corrected_red[1] << ", " << corrected_red[2] << "]" << std::endl;
    std::cout << "CCM -   Green: [" << corrected_green[0] << ", " << corrected_green[1] << ", " << corrected_green[2] << "]" << std::endl;
    std::cout << "CCM -   Blue: [" << corrected_blue[0] << ", " << corrected_blue[1] << ", " << corrected_blue[2] << "]" << std::endl;

    ccm_mat_.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data());
    ccm_mat_.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data());
    ccm_mat_.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data());
    
    // Pre-compute fixed-point matrices and constants
    initialize_fixed_point_matrices();
    
    std::cout << "CCM - Constructor completed" << std::endl;
}

hdr_isp::EigenImage3C ColorCorrectionMatrix::execute() {
    if (!enable_) {
        return raw_;
    }

    if (fp_config_.isEnabled()) {
        return apply_ccm_fixed_point();
    } else {
        return apply_ccm_eigen();
    }
}

hdr_isp::EigenImage3CFixed ColorCorrectionMatrix::execute_fixed() {
    std::cout << "CCM - execute_fixed() started" << std::endl;
    std::cout << "CCM - enable_: " << (enable_ ? "true" : "false") << std::endl;
    std::cout << "CCM - matrices_initialized_: " << (matrices_initialized_ ? "true" : "false") << std::endl;
    std::cout << "CCM - raw_fixed_.rows(): " << raw_fixed_.rows() << std::endl;
    std::cout << "CCM - raw_fixed_.cols(): " << raw_fixed_.cols() << std::endl;
    
    if (!enable_) {
        std::cout << "CCM - execute_fixed() - disabled, returning input" << std::endl;
        return raw_fixed_;
    }

    std::cout << "CCM - execute_fixed() - calling apply_ccm_fixed_point_input()" << std::endl;
    try {
        hdr_isp::EigenImage3CFixed result = apply_ccm_fixed_point_input();
        std::cout << "CCM - execute_fixed() - apply_ccm_fixed_point_input() completed successfully" << std::endl;
        std::cout << "CCM - execute_fixed() - result.rows(): " << result.rows() << std::endl;
        std::cout << "CCM - execute_fixed() - result.cols(): " << result.cols() << std::endl;
        return result;
    } catch (const std::exception& e) {
        std::cout << "CCM - execute_fixed() - Exception caught: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cout << "CCM - execute_fixed() - Unknown exception caught" << std::endl;
        throw;
    }
}

hdr_isp::EigenImage3C ColorCorrectionMatrix::apply_ccm_eigen() {
    // Get CCM parameters
    std::vector<float> corrected_red = parm_ccm_["corrected_red"].as<std::vector<float>>();
    std::vector<float> corrected_green = parm_ccm_["corrected_green"].as<std::vector<float>>();
    std::vector<float> corrected_blue = parm_ccm_["corrected_blue"].as<std::vector<float>>();

    // Create CCM matrix
    ccm_mat_.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data());
    ccm_mat_.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data());
    ccm_mat_.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data());

    // Apply CCM using floating-point arithmetic
    hdr_isp::EigenImage3C result = raw_ * ccm_mat_;

    return result;
}

hdr_isp::EigenImage3C ColorCorrectionMatrix::apply_ccm_fixed_point() {
    if (!matrices_initialized_) {
        initialize_fixed_point_matrices();
    }
    
    // Choose fixed-point type based on precision mode
    if (fp_config_.getPrecisionMode() == hdr_isp::FixedPointPrecision::FAST_8BIT) {
        // Use 8-bit fixed-point arithmetic with cached matrix
        hdr_isp::EigenImage3C result(raw_.rows(), raw_.cols());
        
        #pragma omp parallel for collapse(2) if(raw_.rows() * raw_.cols() > 10000)
        for (int i = 0; i < raw_.rows(); ++i) {
            for (int j = 0; j < raw_.cols(); ++j) {
                // Convert each channel to fixed-point using int32_t for intermediate calculations
                int32_t r_fixed = hdr_isp::FixedPointUtils::floatToFixed<int8_t>(raw_.r()(i, j), fractional_bits_);
                int32_t g_fixed = hdr_isp::FixedPointUtils::floatToFixed<int8_t>(raw_.g()(i, j), fractional_bits_);
                int32_t b_fixed = hdr_isp::FixedPointUtils::floatToFixed<int8_t>(raw_.b()(i, j), fractional_bits_);
                
                // Apply fixed-point matrix multiplication using cached matrix
                int32_t out_r = static_cast<int32_t>(ccm_mat_8bit_(0, 0)) * r_fixed + 
                               static_cast<int32_t>(ccm_mat_8bit_(0, 1)) * g_fixed + 
                               static_cast<int32_t>(ccm_mat_8bit_(0, 2)) * b_fixed;
                int32_t out_g = static_cast<int32_t>(ccm_mat_8bit_(1, 0)) * r_fixed + 
                               static_cast<int32_t>(ccm_mat_8bit_(1, 1)) * g_fixed + 
                               static_cast<int32_t>(ccm_mat_8bit_(1, 2)) * b_fixed;
                int32_t out_b = static_cast<int32_t>(ccm_mat_8bit_(2, 0)) * r_fixed + 
                               static_cast<int32_t>(ccm_mat_8bit_(2, 1)) * g_fixed + 
                               static_cast<int32_t>(ccm_mat_8bit_(2, 2)) * b_fixed;
                
                // Convert back to float using int32_t as the template parameter
                result.r()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int32_t>(out_r, fractional_bits_);
                result.g()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int32_t>(out_g, fractional_bits_);
                result.b()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int32_t>(out_b, fractional_bits_);
            }
        }
        
        return result;
    } else {
        // Use 16-bit fixed-point arithmetic with cached matrix
        hdr_isp::EigenImage3C result(raw_.rows(), raw_.cols());
        
        #pragma omp parallel for collapse(2) if(raw_.rows() * raw_.cols() > 10000)
        for (int i = 0; i < raw_.rows(); ++i) {
            for (int j = 0; j < raw_.cols(); ++j) {
                // Convert each channel to fixed-point using int64_t for intermediate calculations
                int64_t r_fixed = hdr_isp::FixedPointUtils::floatToFixed<int16_t>(raw_.r()(i, j), fractional_bits_);
                int64_t g_fixed = hdr_isp::FixedPointUtils::floatToFixed<int16_t>(raw_.g()(i, j), fractional_bits_);
                int64_t b_fixed = hdr_isp::FixedPointUtils::floatToFixed<int16_t>(raw_.b()(i, j), fractional_bits_);
                
                // Apply fixed-point matrix multiplication using cached matrix
                int64_t out_r = static_cast<int64_t>(ccm_mat_16bit_(0, 0)) * r_fixed + 
                               static_cast<int64_t>(ccm_mat_16bit_(0, 1)) * g_fixed + 
                               static_cast<int64_t>(ccm_mat_16bit_(0, 2)) * b_fixed;
                int64_t out_g = static_cast<int64_t>(ccm_mat_16bit_(1, 0)) * r_fixed + 
                               static_cast<int64_t>(ccm_mat_16bit_(1, 1)) * g_fixed + 
                               static_cast<int64_t>(ccm_mat_16bit_(1, 2)) * b_fixed;
                int64_t out_b = static_cast<int64_t>(ccm_mat_16bit_(2, 0)) * r_fixed + 
                               static_cast<int64_t>(ccm_mat_16bit_(2, 1)) * g_fixed + 
                               static_cast<int64_t>(ccm_mat_16bit_(2, 2)) * b_fixed;
                
                // Convert back to float using int64_t as the template parameter
                result.r()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int64_t>(out_r, fractional_bits_);
                result.g()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int64_t>(out_g, fractional_bits_);
                result.b()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int64_t>(out_b, fractional_bits_);
            }
        }
        
        return result;
    }
}

hdr_isp::EigenImage3CFixed ColorCorrectionMatrix::apply_ccm_fixed_point_input() {
    std::cout << "CCM - apply_ccm_fixed_point_input() started" << std::endl;
    
    if (!matrices_initialized_) {
        initialize_fixed_point_matrices();
    }
    
    // Create result matrix
    hdr_isp::EigenImage3CFixed result(raw_fixed_.rows(), raw_fixed_.cols());
    
    // Choose fixed-point type based on precision mode and use optimized functions
    if (fp_config_.getPrecisionMode() == hdr_isp::FixedPointPrecision::FAST_8BIT) {
        std::cout << "CCM - Using optimized 8-bit fixed-point arithmetic" << std::endl;
        apply_ccm_vectorized_8bit(result);
        
        // Clip to valid range using pre-computed max value
        result = result.clip(-max_val_8bit_, max_val_8bit_);
    } else {
        std::cout << "CCM - Using optimized 16-bit fixed-point arithmetic" << std::endl;
        apply_ccm_vectorized_16bit(result);
        
        // Clip to valid range using pre-computed max value
        result = result.clip(-max_val_16bit_, max_val_16bit_);
    }
    
    std::cout << "CCM - apply_ccm_fixed_point_input() completed" << std::endl;
    return result;
}

void ColorCorrectionMatrix::initialize_fixed_point_matrices() {
    std::cout << "CCM - initialize_fixed_point_matrices() started" << std::endl;
    
    if (matrices_initialized_) {
        std::cout << "CCM - initialize_fixed_point_matrices() - already initialized, returning" << std::endl;
        return;
    }
    
    std::cout << "CCM - initialize_fixed_point_matrices() - fractional_bits_: " << fractional_bits_ << std::endl;
    std::cout << "CCM - initialize_fixed_point_matrices() - ccm_mat_:" << std::endl;
    std::cout << ccm_mat_ << std::endl;
    
    try {
        // Pre-compute fixed-point matrices
        std::cout << "CCM - initialize_fixed_point_matrices() - computing 8-bit matrix..." << std::endl;
        ccm_mat_8bit_ = hdr_isp::FixedPointUtils::applyFixedPointScaling<int8_t>(ccm_mat_, fractional_bits_);
        std::cout << "CCM - initialize_fixed_point_matrices() - 8-bit matrix computed successfully" << std::endl;
        
        std::cout << "CCM - initialize_fixed_point_matrices() - computing 16-bit matrix..." << std::endl;
        ccm_mat_16bit_ = hdr_isp::FixedPointUtils::applyFixedPointScaling<int16_t>(ccm_mat_, fractional_bits_);
        std::cout << "CCM - initialize_fixed_point_matrices() - 16-bit matrix computed successfully" << std::endl;
        
        // Pre-compute constants
        std::cout << "CCM - initialize_fixed_point_matrices() - computing constants..." << std::endl;
        max_val_8bit_ = (1 << (8 - fractional_bits_)) - 1;
        max_val_16bit_ = (1 << (16 - fractional_bits_)) - 1;
        half_scale_32bit_ = 1 << (fractional_bits_ - 1);
        half_scale_64bit_ = 1LL << (fractional_bits_ - 1);
        
        std::cout << "CCM - initialize_fixed_point_matrices() - constants computed:" << std::endl;
        std::cout << "CCM -   max_val_8bit_: " << max_val_8bit_ << std::endl;
        std::cout << "CCM -   max_val_16bit_: " << max_val_16bit_ << std::endl;
        std::cout << "CCM -   half_scale_32bit_: " << half_scale_32bit_ << std::endl;
        std::cout << "CCM -   half_scale_64bit_: " << half_scale_64bit_ << std::endl;
        
        matrices_initialized_ = true;
        std::cout << "CCM - initialize_fixed_point_matrices() - completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CCM - initialize_fixed_point_matrices() - Exception caught: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cout << "CCM - initialize_fixed_point_matrices() - Unknown exception caught" << std::endl;
        throw;
    }
}

void ColorCorrectionMatrix::apply_ccm_vectorized_8bit(hdr_isp::EigenImage3CFixed& result) {
    std::cout << "CCM - apply_ccm_vectorized_8bit() started" << std::endl;
    
    const int rows = raw_fixed_.rows();
    const int cols = raw_fixed_.cols();
    
    std::cout << "CCM - apply_ccm_vectorized_8bit() - rows: " << rows << ", cols: " << cols << std::endl;
    std::cout << "CCM - apply_ccm_vectorized_8bit() - result.rows(): " << result.rows() << ", result.cols(): " << result.cols() << std::endl;
    
    try {
        // Use SIMD-friendly loop structure for better vectorization
        #pragma omp parallel for collapse(2) if(rows * cols > 10000)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // Load pixel values
                const int16_t r_in = raw_fixed_.r()(i, j);
                const int16_t g_in = raw_fixed_.g()(i, j);
                const int16_t b_in = raw_fixed_.b()(i, j);
                
                // Matrix multiplication with pre-computed constants
                const int32_t out_r = static_cast<int32_t>(ccm_mat_8bit_(0, 0)) * r_in + 
                                     static_cast<int32_t>(ccm_mat_8bit_(0, 1)) * g_in + 
                                     static_cast<int32_t>(ccm_mat_8bit_(0, 2)) * b_in;
                const int32_t out_g = static_cast<int32_t>(ccm_mat_8bit_(1, 0)) * r_in + 
                                     static_cast<int32_t>(ccm_mat_8bit_(1, 1)) * g_in + 
                                     static_cast<int32_t>(ccm_mat_8bit_(1, 2)) * b_in;
                const int32_t out_b = static_cast<int32_t>(ccm_mat_8bit_(2, 0)) * r_in + 
                                     static_cast<int32_t>(ccm_mat_8bit_(2, 1)) * g_in + 
                                     static_cast<int32_t>(ccm_mat_8bit_(2, 2)) * b_in;
                
                // Apply rounding and store results
                result.r()(i, j) = static_cast<int16_t>((out_r + half_scale_32bit_) >> fractional_bits_);
                result.g()(i, j) = static_cast<int16_t>((out_g + half_scale_32bit_) >> fractional_bits_);
                result.b()(i, j) = static_cast<int16_t>((out_b + half_scale_32bit_) >> fractional_bits_);
            }
        }
        
        std::cout << "CCM - apply_ccm_vectorized_8bit() - completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CCM - apply_ccm_vectorized_8bit() - Exception caught: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cout << "CCM - apply_ccm_vectorized_8bit() - Unknown exception caught" << std::endl;
        throw;
    }
}

void ColorCorrectionMatrix::apply_ccm_vectorized_16bit(hdr_isp::EigenImage3CFixed& result) {
    std::cout << "CCM - apply_ccm_vectorized_16bit() started" << std::endl;
    
    const int rows = raw_fixed_.rows();
    const int cols = raw_fixed_.cols();
    
    std::cout << "CCM - apply_ccm_vectorized_16bit() - rows: " << rows << ", cols: " << cols << std::endl;
    std::cout << "CCM - apply_ccm_vectorized_16bit() - result.rows(): " << result.rows() << ", result.cols(): " << result.cols() << std::endl;
    
    try {
        // Use SIMD-friendly loop structure for better vectorization
        #pragma omp parallel for collapse(2) if(rows * cols > 10000)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // Load pixel values
                const int16_t r_in = raw_fixed_.r()(i, j);
                const int16_t g_in = raw_fixed_.g()(i, j);
                const int16_t b_in = raw_fixed_.b()(i, j);
                
                // Matrix multiplication with pre-computed constants
                const int64_t out_r = static_cast<int64_t>(ccm_mat_16bit_(0, 0)) * r_in + 
                                     static_cast<int64_t>(ccm_mat_16bit_(0, 1)) * g_in + 
                                     static_cast<int64_t>(ccm_mat_16bit_(0, 2)) * b_in;
                const int64_t out_g = static_cast<int64_t>(ccm_mat_16bit_(1, 0)) * r_in + 
                                     static_cast<int64_t>(ccm_mat_16bit_(1, 1)) * g_in + 
                                     static_cast<int64_t>(ccm_mat_16bit_(1, 2)) * b_in;
                const int64_t out_b = static_cast<int64_t>(ccm_mat_16bit_(2, 0)) * r_in + 
                                     static_cast<int64_t>(ccm_mat_16bit_(2, 1)) * g_in + 
                                     static_cast<int64_t>(ccm_mat_16bit_(2, 2)) * b_in;
                
                // Apply rounding and store results
                result.r()(i, j) = static_cast<int16_t>((out_r + half_scale_64bit_) >> fractional_bits_);
                result.g()(i, j) = static_cast<int16_t>((out_g + half_scale_64bit_) >> fractional_bits_);
                result.b()(i, j) = static_cast<int16_t>((out_b + half_scale_64bit_) >> fractional_bits_);
            }
        }
        
        std::cout << "CCM - apply_ccm_vectorized_16bit() - completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CCM - apply_ccm_vectorized_16bit() - Exception caught: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cout << "CCM - apply_ccm_vectorized_16bit() - Unknown exception caught" << std::endl;
        throw;
    }
} 
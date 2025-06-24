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
    if (!enable_) {
        return raw_fixed_;
    }

    try {
        hdr_isp::EigenImage3CFixed result = apply_ccm_fixed_point_input();
        return result;
    } catch (const std::exception& e) {
        throw;
    } catch (...) {
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
    // Convert fixed-point input to floating-point for processing
    hdr_isp::EigenImage3C float_input = raw_fixed_.toEigenImage3C(fractional_bits_);
    
    // Get CCM parameters
    std::vector<float> corrected_red = parm_ccm_["corrected_red"].as<std::vector<float>>();
    std::vector<float> corrected_green = parm_ccm_["corrected_green"].as<std::vector<float>>();
    std::vector<float> corrected_blue = parm_ccm_["corrected_blue"].as<std::vector<float>>();

    // Create CCM matrix
    ccm_mat_.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data());
    ccm_mat_.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data());
    ccm_mat_.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data());

    // Apply CCM using floating-point arithmetic
    hdr_isp::EigenImage3C result = float_input * ccm_mat_;

    return result;
}

hdr_isp::EigenImage3CFixed ColorCorrectionMatrix::apply_ccm_fixed_point_input() {
    if (!matrices_initialized_) {
        initialize_fixed_point_matrices();
    }
    
    // Create result matrix
    hdr_isp::EigenImage3CFixed result(raw_fixed_.rows(), raw_fixed_.cols());
    
    // Choose fixed-point type based on precision mode and use optimized functions
    if (fp_config_.getPrecisionMode() == hdr_isp::FixedPointPrecision::FAST_8BIT) {
        apply_ccm_vectorized_8bit(result);
        
        // Clip to valid range using pre-computed max value
        result = result.clip(-max_val_8bit_, max_val_8bit_);
    } else {
        apply_ccm_vectorized_16bit(result);
        
        // Clip to valid range using pre-computed max value
        result = result.clip(-max_val_16bit_, max_val_16bit_);
    }
    
    return result;
}

void ColorCorrectionMatrix::initialize_fixed_point_matrices() {
    if (matrices_initialized_) {
        return;
    }
    
    try {
        // Pre-compute fixed-point matrices
        ccm_mat_8bit_ = hdr_isp::FixedPointUtils::applyFixedPointScaling<int8_t>(ccm_mat_, fractional_bits_);
        ccm_mat_16bit_ = hdr_isp::FixedPointUtils::applyFixedPointScaling<int16_t>(ccm_mat_, fractional_bits_);
        
        // Pre-compute constants
        max_val_8bit_ = (1 << (8 - fractional_bits_)) - 1;
        max_val_16bit_ = (1 << (16 - fractional_bits_)) - 1;
        half_scale_32bit_ = 1 << (fractional_bits_ - 1);
        half_scale_64bit_ = 1LL << (fractional_bits_ - 1);
        
        matrices_initialized_ = true;
    } catch (const std::exception& e) {
        throw;
    } catch (...) {
        throw;
    }
}

void ColorCorrectionMatrix::apply_ccm_vectorized_8bit(hdr_isp::EigenImage3CFixed& result) {
    const int rows = raw_fixed_.rows();
    const int cols = raw_fixed_.cols();
    
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
    } catch (const std::exception& e) {
        throw;
    } catch (...) {
        throw;
    }
}

void ColorCorrectionMatrix::apply_ccm_vectorized_16bit(hdr_isp::EigenImage3CFixed& result) {
    const int rows = raw_fixed_.rows();
    const int cols = raw_fixed_.cols();
    
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
    } catch (const std::exception& e) {
        throw;
    } catch (...) {
        throw;
    }
} 
#include "color_correction_matrix.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <filesystem>

ColorCorrectionMatrix::ColorCorrectionMatrix(const hdr_isp::EigenImage3C& img, const YAML::Node& sensor_info, const YAML::Node& parm_ccm, const hdr_isp::FixedPointConfig& fp_config)
    : raw_(img)
    , sensor_info_(sensor_info)
    , parm_ccm_(parm_ccm)
    , fp_config_(fp_config)
    , enable_(parm_ccm["is_enable"].as<bool>())
    , output_bit_depth_(parm_ccm["output_bit_depth"].as<int>())
    , ccm_mat_(Eigen::Matrix3f::Identity())
    , is_save_(parm_ccm["is_save"].as<bool>())
    , use_fixed_input_(false)
{
    // Initialize CCM matrix
    std::vector<float> corrected_red = parm_ccm["corrected_red"].as<std::vector<float>>();
    std::vector<float> corrected_green = parm_ccm["corrected_green"].as<std::vector<float>>();
    std::vector<float> corrected_blue = parm_ccm["corrected_blue"].as<std::vector<float>>();

    ccm_mat_.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data());
    ccm_mat_.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data());
    ccm_mat_.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data());
}

ColorCorrectionMatrix::ColorCorrectionMatrix(const hdr_isp::EigenImage3CFixed& img, const YAML::Node& sensor_info, const YAML::Node& parm_ccm, const hdr_isp::FixedPointConfig& fp_config)
    : raw_fixed_(img)
    , sensor_info_(sensor_info)
    , parm_ccm_(parm_ccm)
    , fp_config_(fp_config)
    , enable_(parm_ccm["is_enable"].as<bool>())
    , output_bit_depth_(parm_ccm["output_bit_depth"].as<int>())
    , ccm_mat_(Eigen::Matrix3f::Identity())
    , is_save_(parm_ccm["is_save"].as<bool>())
    , use_fixed_input_(true)
{
    // Initialize CCM matrix
    std::vector<float> corrected_red = parm_ccm["corrected_red"].as<std::vector<float>>();
    std::vector<float> corrected_green = parm_ccm["corrected_green"].as<std::vector<float>>();
    std::vector<float> corrected_blue = parm_ccm["corrected_blue"].as<std::vector<float>>();

    ccm_mat_.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data());
    ccm_mat_.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data());
    ccm_mat_.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data());
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

    return apply_ccm_fixed_point_input();
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
    // Get CCM parameters
    std::vector<float> corrected_red = parm_ccm_["corrected_red"].as<std::vector<float>>();
    std::vector<float> corrected_green = parm_ccm_["corrected_green"].as<std::vector<float>>();
    std::vector<float> corrected_blue = parm_ccm_["corrected_blue"].as<std::vector<float>>();

    // Create floating-point CCM matrix
    ccm_mat_.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data());
    ccm_mat_.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data());
    ccm_mat_.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data());

    int fractional_bits = fp_config_.getFractionalBits();
    
    // Choose fixed-point type based on precision mode
    if (fp_config_.getPrecisionMode() == hdr_isp::FixedPointPrecision::FAST_8BIT) {
        // Use 8-bit fixed-point arithmetic
        auto ccm_mat_fixed = hdr_isp::FixedPointUtils::applyFixedPointScaling<int8_t>(ccm_mat_, fractional_bits);
        
        // Convert input to fixed-point
        hdr_isp::EigenImage3C result(raw_.rows(), raw_.cols());
        
        for (int i = 0; i < raw_.rows(); ++i) {
            for (int j = 0; j < raw_.cols(); ++j) {
                // Convert each channel to fixed-point using int32_t for intermediate calculations
                int32_t r_fixed = hdr_isp::FixedPointUtils::floatToFixed<int8_t>(raw_.r()(i, j), fractional_bits);
                int32_t g_fixed = hdr_isp::FixedPointUtils::floatToFixed<int8_t>(raw_.g()(i, j), fractional_bits);
                int32_t b_fixed = hdr_isp::FixedPointUtils::floatToFixed<int8_t>(raw_.b()(i, j), fractional_bits);
                
                // Apply fixed-point matrix multiplication using int32_t for intermediate calculations
                int32_t out_r = static_cast<int32_t>(ccm_mat_fixed(0, 0)) * r_fixed + 
                               static_cast<int32_t>(ccm_mat_fixed(0, 1)) * g_fixed + 
                               static_cast<int32_t>(ccm_mat_fixed(0, 2)) * b_fixed;
                int32_t out_g = static_cast<int32_t>(ccm_mat_fixed(1, 0)) * r_fixed + 
                               static_cast<int32_t>(ccm_mat_fixed(1, 1)) * g_fixed + 
                               static_cast<int32_t>(ccm_mat_fixed(1, 2)) * b_fixed;
                int32_t out_b = static_cast<int32_t>(ccm_mat_fixed(2, 0)) * r_fixed + 
                               static_cast<int32_t>(ccm_mat_fixed(2, 1)) * g_fixed + 
                               static_cast<int32_t>(ccm_mat_fixed(2, 2)) * b_fixed;
                
                // Convert back to float using int32_t as the template parameter
                result.r()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int32_t>(out_r, fractional_bits);
                result.g()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int32_t>(out_g, fractional_bits);
                result.b()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int32_t>(out_b, fractional_bits);
            }
        }
        
        return result;
    } else {
        // Use 16-bit fixed-point arithmetic
        auto ccm_mat_fixed = hdr_isp::FixedPointUtils::applyFixedPointScaling<int16_t>(ccm_mat_, fractional_bits);
        
        // Convert input to fixed-point
        hdr_isp::EigenImage3C result(raw_.rows(), raw_.cols());
        
        for (int i = 0; i < raw_.rows(); ++i) {
            for (int j = 0; j < raw_.cols(); ++j) {
                // Convert each channel to fixed-point using int64_t for intermediate calculations
                int64_t r_fixed = hdr_isp::FixedPointUtils::floatToFixed<int16_t>(raw_.r()(i, j), fractional_bits);
                int64_t g_fixed = hdr_isp::FixedPointUtils::floatToFixed<int16_t>(raw_.g()(i, j), fractional_bits);
                int64_t b_fixed = hdr_isp::FixedPointUtils::floatToFixed<int16_t>(raw_.b()(i, j), fractional_bits);
                
                // Apply fixed-point matrix multiplication using int64_t for intermediate calculations
                int64_t out_r = static_cast<int64_t>(ccm_mat_fixed(0, 0)) * r_fixed + 
                               static_cast<int64_t>(ccm_mat_fixed(0, 1)) * g_fixed + 
                               static_cast<int64_t>(ccm_mat_fixed(0, 2)) * b_fixed;
                int64_t out_g = static_cast<int64_t>(ccm_mat_fixed(1, 0)) * r_fixed + 
                               static_cast<int64_t>(ccm_mat_fixed(1, 1)) * g_fixed + 
                               static_cast<int64_t>(ccm_mat_fixed(1, 2)) * b_fixed;
                int64_t out_b = static_cast<int64_t>(ccm_mat_fixed(2, 0)) * r_fixed + 
                               static_cast<int64_t>(ccm_mat_fixed(2, 1)) * g_fixed + 
                               static_cast<int64_t>(ccm_mat_fixed(2, 2)) * b_fixed;
                
                // Convert back to float using int64_t as the template parameter
                result.r()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int64_t>(out_r, fractional_bits);
                result.g()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int64_t>(out_g, fractional_bits);
                result.b()(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int64_t>(out_b, fractional_bits);
            }
        }
        
        return result;
    }
}

hdr_isp::EigenImage3CFixed ColorCorrectionMatrix::apply_ccm_fixed_point_input() {
    // Get CCM parameters
    std::vector<float> corrected_red = parm_ccm_["corrected_red"].as<std::vector<float>>();
    std::vector<float> corrected_green = parm_ccm_["corrected_green"].as<std::vector<float>>();
    std::vector<float> corrected_blue = parm_ccm_["corrected_blue"].as<std::vector<float>>();

    // Create floating-point CCM matrix
    ccm_mat_.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data());
    ccm_mat_.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data());
    ccm_mat_.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data());

    int fractional_bits = fp_config_.getFractionalBits();
    
    // Choose fixed-point type based on precision mode
    if (fp_config_.getPrecisionMode() == hdr_isp::FixedPointPrecision::FAST_8BIT) {
        // Use 8-bit fixed-point arithmetic
        Eigen::Matrix<int8_t, 3, 3> ccm_mat_8bit = hdr_isp::FixedPointUtils::applyFixedPointScaling<int8_t>(ccm_mat_, fractional_bits);
        
        std::cout << "CCM - Using 8-bit fixed-point arithmetic (fractional bits: " << fractional_bits << ")" << std::endl;
        
        // Cast to Eigen::Matrix3i for compatibility with EigenImage3CFixed operator*
        Eigen::Matrix3i ccm_mat_int = ccm_mat_8bit.cast<int>();
        
        // Apply fixed-point matrix multiplication
        hdr_isp::EigenImage3CFixed result = raw_fixed_ * ccm_mat_int;
        
        // Clip to valid range
        int16_t max_val = (1 << (8 - fractional_bits)) - 1;
        result = result.clip(-max_val, max_val);
        
        return result;
    } else {
        // Use 16-bit fixed-point arithmetic
        Eigen::Matrix<int16_t, 3, 3> ccm_mat_16bit = hdr_isp::FixedPointUtils::applyFixedPointScaling<int16_t>(ccm_mat_, fractional_bits);
        
        std::cout << "CCM - Using 16-bit fixed-point arithmetic (fractional bits: " << fractional_bits << ")" << std::endl;
        
        // Cast to Eigen::Matrix3i for compatibility with EigenImage3CFixed operator*
        Eigen::Matrix3i ccm_mat_int = ccm_mat_16bit.cast<int>();
        
        // Apply fixed-point matrix multiplication
        hdr_isp::EigenImage3CFixed result = raw_fixed_ * ccm_mat_int;
        
        // Clip to valid range
        int16_t max_val = (1 << (16 - fractional_bits)) - 1;
        result = result.clip(-max_val, max_val);
        
        return result;
    }
} 
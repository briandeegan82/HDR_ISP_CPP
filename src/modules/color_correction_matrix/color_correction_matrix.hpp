#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"
#include "../../common/fixed_point_utils.hpp"

class ColorCorrectionMatrix {
public:
    ColorCorrectionMatrix(const hdr_isp::EigenImage3C& img, const YAML::Node& sensor_info, const YAML::Node& parm_ccm, const hdr_isp::FixedPointConfig& fp_config);
    ColorCorrectionMatrix(const hdr_isp::EigenImage3CFixed& img, const YAML::Node& sensor_info, const YAML::Node& parm_ccm, const hdr_isp::FixedPointConfig& fp_config);
    
    hdr_isp::EigenImage3C execute();
    hdr_isp::EigenImage3CFixed execute_fixed();

private:
    hdr_isp::EigenImage3C apply_ccm_eigen();
    hdr_isp::EigenImage3C apply_ccm_fixed_point();
    hdr_isp::EigenImage3CFixed apply_ccm_fixed_point_input();
    
    // Performance optimization: Cache fixed-point matrices
    void initialize_fixed_point_matrices();
    void apply_ccm_vectorized_8bit(hdr_isp::EigenImage3CFixed& result);
    void apply_ccm_vectorized_16bit(hdr_isp::EigenImage3CFixed& result);

    hdr_isp::EigenImage3C raw_;
    hdr_isp::EigenImage3CFixed raw_fixed_;
    YAML::Node sensor_info_;
    YAML::Node parm_ccm_;
    hdr_isp::FixedPointConfig fp_config_;
    bool enable_;
    Eigen::Matrix3f ccm_mat_;
    bool is_save_;
    bool use_fixed_input_;
    
    // Cached fixed-point matrices for performance
    Eigen::Matrix<int8_t, 3, 3> ccm_mat_8bit_;
    Eigen::Matrix<int16_t, 3, 3> ccm_mat_16bit_;
    bool matrices_initialized_;
    int fractional_bits_;
    int16_t max_val_8bit_;
    int16_t max_val_16bit_;
    int32_t half_scale_32bit_;
    int64_t half_scale_64bit_;
}; 
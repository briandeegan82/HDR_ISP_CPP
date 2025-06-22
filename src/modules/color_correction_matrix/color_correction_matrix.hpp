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

    hdr_isp::EigenImage3C raw_;
    hdr_isp::EigenImage3CFixed raw_fixed_;
    YAML::Node sensor_info_;
    YAML::Node parm_ccm_;
    hdr_isp::FixedPointConfig fp_config_;
    bool enable_;
    int output_bit_depth_;
    Eigen::Matrix3f ccm_mat_;
    bool is_save_;
    bool use_fixed_input_;
}; 
#pragma once

#include <string>
#include <array>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class AutoWhiteBalance {
public:
    AutoWhiteBalance(const hdr_isp::EigenImageU32& raw, const YAML::Node& sensor_info, const YAML::Node& parm_awb);
    std::array<double, 2> execute();

private:
    hdr_isp::EigenImageU32 raw_;
    YAML::Node sensor_info_;
    YAML::Node parm_awb_;
    bool enable_;
    int bit_depth_;
    bool is_debug_;
    float underexposed_percentage_;
    float overexposed_percentage_;
    Eigen::MatrixXf flatten_img_;
    std::string bayer_;
    std::string algorithm_;

    std::tuple<double, double> determine_white_balance_gain();
    std::tuple<double, double> apply_gray_world();
    std::tuple<double, double> apply_norm_gray_world();
    std::tuple<double, double> apply_pca_illuminant_estimation();
}; 
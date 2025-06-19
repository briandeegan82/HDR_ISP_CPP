#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <array>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class AutoWhiteBalance {
public:
    AutoWhiteBalance(const cv::Mat& raw, const YAML::Node& sensor_info, const YAML::Node& parm_awb);
    std::array<double, 2> execute();

private:
    cv::Mat raw_;
    YAML::Node sensor_info_;
    YAML::Node parm_awb_;
    bool enable_;
    int bit_depth_;
    bool is_debug_;
    float underexposed_percentage_;
    float overexposed_percentage_;
    cv::Mat flatten_img_;
    std::string bayer_;
    std::string algorithm_;
    bool use_eigen_; // Use Eigen by default

    std::tuple<double, double> determine_white_balance_gain();
    std::tuple<double, double> determine_white_balance_gain_eigen();
    std::tuple<double, double> apply_gray_world();
    std::tuple<double, double> apply_gray_world_eigen();
    std::tuple<double, double> apply_norm_gray_world();
    std::tuple<double, double> apply_norm_gray_world_eigen();
    std::tuple<double, double> apply_pca_illuminant_estimation();
    std::tuple<double, double> apply_pca_illuminant_estimation_eigen();
}; 
#pragma once

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include "../../common/eigen_utils.hpp"

class GammaCorrection {
public:
    GammaCorrection(const cv::Mat& img, const YAML::Node& platform, 
                   const YAML::Node& sensor_info, const YAML::Node& parm_gmm);
    cv::Mat execute();

private:
    std::vector<uint32_t> generate_gamma_lut(int bit_depth);
    cv::Mat apply_gamma_opencv();
    cv::Mat apply_gamma_eigen();
    void save();

    cv::Mat img_;
    bool enable_;
    YAML::Node sensor_info_;
    int output_bit_depth_;
    YAML::Node parm_gmm_;
    bool is_save_;
    bool is_debug_;
    YAML::Node platform_;
    bool use_eigen_;
}; 
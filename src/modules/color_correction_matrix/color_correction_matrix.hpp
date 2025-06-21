#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class ColorCorrectionMatrix {
public:
    ColorCorrectionMatrix(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_ccm);
    
    cv::Mat execute();

private:
    cv::Mat apply_ccm_opencv();
    cv::Mat apply_ccm_eigen();

    cv::Mat raw_;
    YAML::Node sensor_info_;
    YAML::Node parm_ccm_;
    bool enable_;
    int output_bit_depth_;
    cv::Mat ccm_mat_;
    bool is_save_;
    bool use_eigen_;
}; 
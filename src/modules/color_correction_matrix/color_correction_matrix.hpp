#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

class ColorCorrectionMatrix {
public:
    ColorCorrectionMatrix(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_ccm);
    
    cv::Mat execute();

private:
    cv::Mat apply_ccm();

    cv::Mat raw_;
    YAML::Node sensor_info_;
    YAML::Node parm_ccm_;
    bool enable_;
    int output_bit_depth_;
    cv::Mat ccm_mat_;
    bool is_save_;
}; 
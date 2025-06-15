#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

class BlackLevelCorrection {
public:
    BlackLevelCorrection(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_blc);
    
    cv::Mat execute();

private:
    cv::Mat apply_blc_parameters();

    cv::Mat raw_;
    YAML::Node sensor_info_;
    YAML::Node parm_blc_;
    bool enable_;
    bool is_linearize_;
    int bit_depth_;
    std::string bayer_pattern_;
    bool is_save_;
}; 
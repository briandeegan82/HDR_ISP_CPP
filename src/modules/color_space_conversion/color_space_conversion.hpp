#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

class ColorSpaceConversion {
public:
    ColorSpaceConversion(const cv::Mat& img, const YAML::Node& sensor_info, 
                        const YAML::Node& parm_csc, const YAML::Node& parm_cse);
    
    cv::Mat execute();

private:
    cv::Mat rgb_to_yuv_8bit();

    cv::Mat raw_;
    YAML::Node sensor_info_;
    YAML::Node parm_csc_;
    YAML::Node parm_cse_;
    int bit_depth_;
    int conv_std_;
    cv::Mat rgb2yuv_mat_;
    bool is_save_;
}; 
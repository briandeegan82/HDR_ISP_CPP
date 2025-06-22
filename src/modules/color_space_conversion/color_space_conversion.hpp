#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class ColorSpaceConversion {
public:
    ColorSpaceConversion(const cv::Mat& img, const YAML::Node& sensor_info, 
                        const YAML::Node& parm_csc, const YAML::Node& parm_cse);
    ColorSpaceConversion(const hdr_isp::EigenImage3C& img, const YAML::Node& sensor_info, 
                        const YAML::Node& parm_csc, const YAML::Node& parm_cse);
    
    cv::Mat execute();
    hdr_isp::EigenImage3C execute_eigen();

private:
    cv::Mat rgb_to_yuv_8bit();
    hdr_isp::EigenImage3C rgb_to_yuv_8bit_eigen();

    cv::Mat raw_;
    hdr_isp::EigenImage3C eigen_raw_;
    YAML::Node sensor_info_;
    YAML::Node parm_csc_;
    YAML::Node parm_cse_;
    int bit_depth_;
    int conv_std_;
    cv::Mat rgb2yuv_mat_;
    bool is_save_;
    bool use_eigen_; // Use Eigen by default
    bool has_eigen_input_; // Flag to indicate if input is Eigen
}; 
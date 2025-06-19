#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class RGBConversion {
public:
    RGBConversion(cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info, 
                 const YAML::Node& parm_rgb, const YAML::Node& parm_csc);

    cv::Mat execute();

private:
    cv::Mat yuv_to_rgb_opencv();
    hdr_isp::EigenImage yuv_to_rgb_eigen();
    void save();

    cv::Mat img_;
    const YAML::Node& platform_;
    const YAML::Node& sensor_info_;
    const YAML::Node& parm_rgb_;
    const YAML::Node& parm_csc_;
    bool enable_;
    bool is_save_;
    int bit_depth_;
    int conv_std_;
    cv::Mat yuv_img_;
    cv::Mat yuv2rgb_mat_;
    cv::Vec3i offset_;
    bool use_eigen_;
}; 
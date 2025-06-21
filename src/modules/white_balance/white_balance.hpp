#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class WhiteBalance {
public:
    WhiteBalance(const cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                 const YAML::Node& parm_wbc);

    cv::Mat execute();

private:
    cv::Mat apply_wb_parameters_opencv();
    hdr_isp::EigenImage32 apply_wb_parameters_eigen();
    void save();

    cv::Mat img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_wbc_;

    bool is_enable_;
    bool is_save_;
    bool is_auto_;
    bool is_debug_;
    std::string bayer_;
    int bpp_;
    cv::Mat raw_;
    bool use_eigen_;
}; 
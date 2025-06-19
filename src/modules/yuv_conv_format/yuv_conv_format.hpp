#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <regex>
#include "../../common/eigen_utils.hpp"

class YUVConvFormat {
public:
    YUVConvFormat(const cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                  const YAML::Node& parm_yuv);

    cv::Mat execute();

private:
    cv::Mat convert2yuv_format_opencv();
    hdr_isp::EigenImage convert2yuv_format_eigen();
    void save();

    cv::Mat img_;
    cv::Size shape_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_yuv_;
    std::string in_file_;

    bool is_enable_;
    bool is_save_;
    bool use_eigen_;
}; 
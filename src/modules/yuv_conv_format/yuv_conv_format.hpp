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
    YUVConvFormat(const hdr_isp::EigenImage3C& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                  const YAML::Node& parm_yuv);

    cv::Mat execute();
    hdr_isp::EigenImage3C execute_eigen();

private:
    cv::Mat convert2yuv_format_opencv();
    hdr_isp::EigenImage convert2yuv_format_eigen();
    hdr_isp::EigenImage3C convert2yuv_format_eigen_3c();
    void save();

    cv::Mat img_;
    hdr_isp::EigenImage3C eigen_img_;
    cv::Size shape_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_yuv_;
    std::string in_file_;

    bool is_enable_;
    bool is_save_;
    bool is_debug_;
    bool use_eigen_;
    bool has_eigen_input_;
}; 
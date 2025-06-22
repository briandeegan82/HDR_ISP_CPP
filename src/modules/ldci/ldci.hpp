#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class LDCI {
public:
    LDCI(const cv::Mat& img, const YAML::Node& platform,
         const YAML::Node& sensor_info, const YAML::Node& params);
    LDCI(const hdr_isp::EigenImage3C& img, const YAML::Node& platform,
         const YAML::Node& sensor_info, const YAML::Node& params);

    cv::Mat execute();
    hdr_isp::EigenImage3C execute_eigen();

private:
    cv::Mat apply_ldci_opencv();
    hdr_isp::EigenImage3C apply_ldci_eigen();
    cv::Mat apply_ldci_multi_channel();
    cv::Mat calculate_local_contrast_opencv(const cv::Mat& img);
    hdr_isp::EigenImage calculate_local_contrast_eigen(const hdr_isp::EigenImage& img);
    cv::Mat enhance_contrast_opencv(const cv::Mat& img, const cv::Mat& local_contrast);
    hdr_isp::EigenImage enhance_contrast_eigen(const hdr_isp::EigenImage& img, const hdr_isp::EigenImage& local_contrast);
    void save();

    cv::Mat img_;
    hdr_isp::EigenImage3C eigen_img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node params_;
    bool is_enable_;
    bool is_save_;
    bool is_debug_;
    float strength_;
    int window_size_;
    int output_bit_depth_;
    bool use_eigen_;
    bool has_eigen_input_;
}; 
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class NoiseReduction2D {
public:
    NoiseReduction2D(const cv::Mat& img, const YAML::Node& platform,
                     const YAML::Node& sensor_info, const YAML::Node& params);
    NoiseReduction2D(const hdr_isp::EigenImage3C& img, const YAML::Node& platform,
                     const YAML::Node& sensor_info, const YAML::Node& params);

    cv::Mat execute();
    hdr_isp::EigenImage3C execute_eigen();

private:
    cv::Mat apply_noise_reduction();
    hdr_isp::EigenImage apply_noise_reduction_eigen();
    hdr_isp::EigenImage3C apply_noise_reduction_eigen_3c();
    cv::Mat apply_bilateral_filter(const cv::Mat& img);
    hdr_isp::EigenImage apply_bilateral_filter_eigen(const hdr_isp::EigenImage& img);
    void save();

    cv::Mat img_;
    hdr_isp::EigenImage3C eigen_img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node params_;
    bool is_enable_;
    bool is_save_;
    bool is_debug_;
    float sigma_space_;
    float sigma_color_;
    int window_size_;
    int output_bit_depth_;
    bool use_eigen_; // Use Eigen by default
    bool has_eigen_input_;
}; 
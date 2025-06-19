#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class HDRDurandToneMapping {
public:
    HDRDurandToneMapping(const cv::Mat& img, const YAML::Node& platform,
                         const YAML::Node& sensor_info, const YAML::Node& params);

    cv::Mat execute();

private:
    cv::Mat normalize(const cv::Mat& image);
    hdr_isp::EigenImage normalize_eigen(const hdr_isp::EigenImage& image);
    cv::Mat fast_bilateral_filter(const cv::Mat& image);
    hdr_isp::EigenImage fast_bilateral_filter_eigen(const hdr_isp::EigenImage& image);
    cv::Mat bilateral_filter(const cv::Mat& image, float sigma_color, float sigma_space);
    hdr_isp::EigenImage bilateral_filter_eigen(const hdr_isp::EigenImage& image, float sigma_color, float sigma_space);
    cv::Mat apply_tone_mapping();
    hdr_isp::EigenImage apply_tone_mapping_eigen();
    void save();

    cv::Mat img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node params_;
    bool is_enable_;
    bool is_save_;
    bool is_debug_;
    float sigma_space_;
    float sigma_color_;
    float contrast_factor_;
    int downsample_factor_;
    int output_bit_depth_;
    bool use_eigen_; // Use Eigen by default
}; 
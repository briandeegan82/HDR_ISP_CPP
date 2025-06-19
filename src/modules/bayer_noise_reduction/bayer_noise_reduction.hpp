#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class BayerNoiseReduction {
public:
    BayerNoiseReduction(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_bnr);
    
    cv::Mat execute();

private:
    cv::Mat apply_bnr();
    hdr_isp::EigenImage apply_bnr_eigen();
    void extract_channels(const cv::Mat& img, cv::Mat& r_channel, cv::Mat& b_channel);
    void extract_channels_eigen(const hdr_isp::EigenImage& img, hdr_isp::EigenImage& r_channel, hdr_isp::EigenImage& b_channel);
    void combine_channels(const cv::Mat& r_channel, const cv::Mat& g_channel, const cv::Mat& b_channel, cv::Mat& output);
    void combine_channels_eigen(const hdr_isp::EigenImage& r_channel, const hdr_isp::EigenImage& g_channel, const hdr_isp::EigenImage& b_channel, hdr_isp::EigenImage& output);
    cv::Mat interpolate_green_channel(const cv::Mat& img);
    hdr_isp::EigenImage interpolate_green_channel_eigen(const hdr_isp::EigenImage& img);
    cv::Mat bilateral_filter(const cv::Mat& src, int d, double sigmaColor, double sigmaSpace);
    hdr_isp::EigenImage bilateral_filter_eigen(const hdr_isp::EigenImage& src, int d, double sigmaColor, double sigmaSpace);

    cv::Mat raw_;
    YAML::Node sensor_info_;
    YAML::Node parm_bnr_;
    bool enable_;
    int bit_depth_;
    std::string bayer_pattern_;
    int width_;
    int height_;
    bool is_save_;
    bool use_eigen_; // Use Eigen by default
}; 
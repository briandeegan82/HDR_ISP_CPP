#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

class BayerNoiseReduction {
public:
    BayerNoiseReduction(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_bnr);
    
    cv::Mat execute();

private:
    cv::Mat apply_bnr();
    void extract_channels(const cv::Mat& img, cv::Mat& r_channel, cv::Mat& b_channel);
    void combine_channels(const cv::Mat& r_channel, const cv::Mat& g_channel, const cv::Mat& b_channel, cv::Mat& output);
    cv::Mat interpolate_green_channel(const cv::Mat& img);
    cv::Mat bilateral_filter(const cv::Mat& src, int d, double sigmaColor, double sigmaSpace);

    cv::Mat raw_;
    YAML::Node sensor_info_;
    YAML::Node parm_bnr_;
    bool enable_;
    int bit_depth_;
    std::string bayer_pattern_;
    int width_;
    int height_;
    bool is_save_;
}; 
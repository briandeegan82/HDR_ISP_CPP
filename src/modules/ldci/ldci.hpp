#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

class LDCI {
public:
    LDCI(const cv::Mat& img, const YAML::Node& platform,
         const YAML::Node& sensor_info, const YAML::Node& params);

    cv::Mat execute();

private:
    cv::Mat apply_ldci();
    cv::Mat calculate_local_contrast(const cv::Mat& img);
    cv::Mat enhance_contrast(const cv::Mat& img, const cv::Mat& local_contrast);
    void save();

    cv::Mat img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node params_;
    bool is_enable_;
    bool is_save_;
    bool is_debug_;
    float strength_;
    int window_size_;
    int output_bit_depth_;
}; 
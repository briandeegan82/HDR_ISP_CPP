#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include <tuple>

class AutoExposure {
public:
    AutoExposure(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_ae);
    
    int execute();
    int get_exposure_feedback();
    int determine_exposure(double skewness);
    double get_luminance_histogram_skewness(const cv::Mat& hist);
    std::tuple<cv::Mat, double> get_greyscale_image(const cv::Mat& img);

private:
    cv::Mat img_;
    YAML::Node sensor_info_;
    YAML::Node param_ae_;
    bool enable_;
    bool is_debug_;
    float center_illuminance_;
    float histogram_skewness_range_;
    int bit_depth_;
}; 
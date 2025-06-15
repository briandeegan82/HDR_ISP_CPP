#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

class DeadPixelCorrection {
public:
    DeadPixelCorrection(const cv::Mat& img, const YAML::Node& platform,
                        const YAML::Node& sensor_info, const YAML::Node& parm_dpc);

    cv::Mat execute();

private:
    cv::Mat correct_dead_pixels();
    void save(const std::string& filename_tag);

    cv::Mat img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_dpc_;
    bool enable_;
    bool is_debug_;
    bool is_save_;
    int bit_depth_;
    std::string bayer_pattern_;
}; 
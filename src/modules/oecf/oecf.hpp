#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

class OECF {
public:
    OECF(cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info, const YAML::Node& parm_oecf);

    cv::Mat execute();

private:
    cv::Mat apply_oecf();
    void save();

    cv::Mat& img_;
    const YAML::Node& platform_;
    const YAML::Node& sensor_info_;
    const YAML::Node& parm_oecf_;
    bool enable_;
    bool is_save_;
}; 
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

class LensShadingCorrection {
public:
    LensShadingCorrection(const cv::Mat& img, const YAML::Node& platform,
                          const YAML::Node& sensor_info, const YAML::Node& parm_lsc);

    cv::Mat execute();

private:
    cv::Mat img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_lsc_;
    bool enable_;
}; 
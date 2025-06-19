#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class LensShadingCorrection {
public:
    LensShadingCorrection(const cv::Mat& img, const YAML::Node& platform,
                          const YAML::Node& sensor_info, const YAML::Node& parm_lsc);

    cv::Mat execute();

private:
    cv::Mat apply_lsc_opencv();
    hdr_isp::EigenImage apply_lsc_eigen();

    cv::Mat img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_lsc_;
    bool enable_;
    bool use_eigen_;
}; 
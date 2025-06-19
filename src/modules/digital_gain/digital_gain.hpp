#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class DigitalGain {
public:
    DigitalGain(const cv::Mat& img, const YAML::Node& platform,
                const YAML::Node& sensor_info, const YAML::Node& parm_dga);

    std::pair<cv::Mat, int> execute();

private:
    cv::Mat apply_digital_gain_opencv();
    hdr_isp::EigenImage apply_digital_gain_eigen();
    void save();

    cv::Mat img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_dga_;
    bool is_save_;
    bool is_debug_;
    bool is_auto_;
    std::vector<float> gains_array_;
    int current_gain_;
    float ae_feedback_;
    bool use_eigen_;
}; 
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

class Scale {
public:
    Scale(cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info, 
          const YAML::Node& parm_sca, int conv_std);

    cv::Mat execute();

private:
    cv::Mat apply_scaling();
    void get_scaling_params();
    void save();

    cv::Mat& img_;
    const YAML::Node& platform_;
    const YAML::Node& sensor_info_;
    YAML::Node parm_sca_;  // Non-const copy to allow modification
    bool enable_;
    bool is_save_;
    int conv_std_;
    bool is_debug_;
    std::pair<int, int> old_size_;
    std::pair<int, int> new_size_;
}; 
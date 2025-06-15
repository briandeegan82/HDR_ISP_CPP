#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

class Scale2D {
public:
    Scale2D(const cv::Mat& single_channel, const YAML::Node& sensor_info, const YAML::Node& parm_sca);

    cv::Mat execute();

private:
    cv::Mat resize_by_non_int_fact(const std::pair<std::pair<int, int>, std::pair<int, int>>& red_fact,
                                  const std::pair<std::string, std::string>& method);
    cv::Mat hardware_dep_scaling();
    cv::Mat hardware_indp_scaling();
    std::vector<std::vector<std::pair<int, int>>> validate_input_output();
    cv::Mat apply_algo(const std::vector<std::vector<std::pair<int, int>>>& scale_info,
                      const std::pair<std::string, std::string>& method);
    void get_scaling_params();

    cv::Mat single_channel_;
    const YAML::Node& sensor_info_;
    const YAML::Node& parm_sca_;
    std::pair<int, int> old_size_;
    std::pair<int, int> new_size_;
    bool is_debug_;
    bool is_hardware_;
    std::string algo_;
    std::string upscale_method_;
    std::string downscale_method_;
}; 
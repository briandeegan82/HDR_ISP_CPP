#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class Sharpen {
public:
    Sharpen(const cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info, 
            const YAML::Node& parm_shp, const std::string& conv_std);

    cv::Mat execute();

private:
    cv::Mat apply_sharpen_opencv();
    hdr_isp::EigenImage apply_sharpen_eigen();
    void get_sharpen_params();
    void save(const std::string& filename);

    cv::Mat img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_shp_;
    std::string conv_std_;

    bool is_enable_;
    bool is_save_;
    bool is_debug_;
    float strength_;
    int kernel_size_;
    int output_bit_depth_;
    bool use_eigen_;
}; 
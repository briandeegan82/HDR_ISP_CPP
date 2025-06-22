#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class Sharpen {
public:
    Sharpen(const cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info, 
            const YAML::Node& parm_shp, const std::string& conv_std);
    Sharpen(const hdr_isp::EigenImage3C& img, const YAML::Node& platform, const YAML::Node& sensor_info, 
            const YAML::Node& parm_shp, const std::string& conv_std);

    cv::Mat execute();
    hdr_isp::EigenImage3C execute_eigen();

private:
    cv::Mat apply_sharpen_opencv();
    cv::Mat apply_sharpen_eigen_opencv();
    hdr_isp::EigenImage3C apply_sharpen_eigen();
    void get_sharpen_params();
    void save(const std::string& filename);

    cv::Mat img_;
    hdr_isp::EigenImage3C eigen_img_;
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
    bool has_eigen_input_;
}; 
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class BlackLevelCorrection {
public:
    BlackLevelCorrection(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_blc);
    
    cv::Mat execute();

private:
    cv::Mat apply_blc_parameters_opencv();
    hdr_isp::EigenImage apply_blc_parameters_eigen(const hdr_isp::EigenImage& img);
    void apply_blc_bayer_eigen(hdr_isp::EigenImage& img, double r_offset, double gr_offset, double gb_offset, double b_offset, double r_sat, double gr_sat, double gb_sat, double b_sat);
    cv::Mat raw_;
    YAML::Node sensor_info_;
    YAML::Node parm_blc_;
    bool enable_;
    bool is_linearize_;
    int bit_depth_;
    std::string bayer_pattern_;
    bool is_save_;
    bool use_eigen_; // Flag to choose between Eigen and OpenCV implementation
}; 
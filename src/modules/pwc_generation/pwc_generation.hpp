#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class PiecewiseCurve {
public:
    PiecewiseCurve(cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info, const YAML::Node& parm_cmpd);

    cv::Mat execute();

private:
    static std::vector<double> generate_decompanding_lut(
        const std::vector<int>& companded_pin,
        const std::vector<int>& companded_pout,
        int max_input_value = 4095
    );
    void save();

    cv::Mat execute_opencv();
    hdr_isp::EigenImage32 execute_eigen();

    cv::Mat& img_;
    const YAML::Node& platform_;
    const YAML::Node& sensor_info_;
    const YAML::Node& parm_cmpd_;
    bool enable_;
    int bit_depth_;
    std::vector<int> companded_pin_;
    std::vector<int> companded_pout_;
    bool is_save_;
    bool is_debug_;
    bool use_eigen_; // Flag to choose between Eigen and OpenCV implementation
}; 
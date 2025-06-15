#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class GammaCorrection {
public:
    GammaCorrection(const cv::Mat& img, const YAML::Node& platform, 
                   const YAML::Node& sensor_info, const YAML::Node& parm_gmm);
    cv::Mat execute();

private:
    cv::Mat img_;
    bool enable_;
    YAML::Node sensor_info_;
    int output_bit_depth_;
    YAML::Node parm_gmm_;
    bool is_save_;
    YAML::Node platform_;

    std::vector<uint32_t> generate_gamma_lut(int bit_depth);
    cv::Mat apply_gamma();
    void save();
}; 
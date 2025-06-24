#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Halide.h>
#include "../../common/eigen_utils.hpp"
#include "scale.hpp"

class ScaleHybrid : public Scale {
public:
    ScaleHybrid(cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info, 
                const YAML::Node& parm_sca, int conv_std);
    ScaleHybrid(const hdr_isp::EigenImage3C& img, const YAML::Node& platform, const YAML::Node& sensor_info, 
                const YAML::Node& parm_sca, int conv_std);

    cv::Mat execute() override;
    hdr_isp::EigenImage3C execute_eigen() override;

private:
    // Halide-optimized scaling functions
    Halide::Buffer<float> apply_nearest_neighbor_halide(const Halide::Buffer<float>& input, int new_width, int new_height);
    Halide::Buffer<float> apply_bilinear_halide(const Halide::Buffer<float>& input, int new_width, int new_height);
    Halide::Buffer<float> apply_bicubic_halide(const Halide::Buffer<float>& input, int new_width, int new_height);
    
    // Utility functions
    Halide::Buffer<float> opencv_to_halide(const cv::Mat& img);
    cv::Mat halide_to_opencv(const Halide::Buffer<float>& buffer);
    Halide::Buffer<float> eigen_to_halide(const hdr_isp::EigenImage& eigen_img);
    hdr_isp::EigenImage halide_to_eigen(const Halide::Buffer<float>& buffer, int rows, int cols);
    
    // Helper functions
    std::string get_algorithm();
    void save();
    
    // Member variables
    std::string algorithm_;
    bool is_hardware_;
}; 
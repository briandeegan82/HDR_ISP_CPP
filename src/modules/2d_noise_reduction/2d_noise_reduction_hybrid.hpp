#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include <Halide.h>
#include "../../common/eigen_utils.hpp"

class NoiseReduction2DHybrid : public NoiseReduction2D {
public:
    NoiseReduction2DHybrid(const cv::Mat& img, const YAML::Node& platform,
                           const YAML::Node& sensor_info, const YAML::Node& params);
    NoiseReduction2DHybrid(const hdr_isp::EigenImage3C& img, const YAML::Node& platform,
                           const YAML::Node& sensor_info, const YAML::Node& params);

    cv::Mat execute() override;
    hdr_isp::EigenImage3C execute_eigen() override;

private:
    Halide::Buffer<float> apply_bilateral_filter_halide(const Halide::Buffer<float>& input);
    Halide::Func create_bilateral_kernel();
    Halide::Func apply_spatial_filter(Halide::Buffer<float> input, Halide::Func kernel);
    Halide::Func apply_range_filter(Halide::Buffer<float> input, float sigma_color);
    Halide::Buffer<float> opencv_to_halide(const cv::Mat& img);
    cv::Mat halide_to_opencv(const Halide::Buffer<float>& buffer);
    void save();
}; 
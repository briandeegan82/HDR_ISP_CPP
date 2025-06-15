#include "lens_shading_correction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

LensShadingCorrection::LensShadingCorrection(const cv::Mat& img, const YAML::Node& platform,
                                            const YAML::Node& sensor_info, const YAML::Node& parm_lsc)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_lsc_(parm_lsc)
    , enable_(parm_lsc["is_enable"].as<bool>())
{
}

cv::Mat LensShadingCorrection::execute() {
    // Currently, the module just returns the input image
    // This is a placeholder for future implementation of lens shading correction
    return img_;
} 
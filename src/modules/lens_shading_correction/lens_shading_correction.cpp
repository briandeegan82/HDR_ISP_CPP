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
    , use_eigen_(true) // Use Eigen by default
{
}

cv::Mat LensShadingCorrection::apply_lsc_opencv() {
    // Placeholder: just return the input image
    return img_;
}

hdr_isp::EigenImage LensShadingCorrection::apply_lsc_eigen() {
    // Placeholder: just return the input image as Eigen
    return hdr_isp::EigenImage::fromOpenCV(img_);
}

cv::Mat LensShadingCorrection::execute() {
    if (enable_) {
        if (use_eigen_) {
            hdr_isp::EigenImage result = apply_lsc_eigen();
            return result.toOpenCV(img_.type());
        } else {
            return apply_lsc_opencv();
        }
    }
    return img_;
} 
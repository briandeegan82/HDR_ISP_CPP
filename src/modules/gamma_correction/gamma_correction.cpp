#include "gamma_correction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

GammaCorrection::GammaCorrection(const cv::Mat& img, const YAML::Node& platform,
                               const YAML::Node& sensor_info, const YAML::Node& parm_gmm)
    : img_(img)
    , enable_(parm_gmm["is_enable"].as<bool>())
    , sensor_info_(sensor_info)
    , output_bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , parm_gmm_(parm_gmm)
    , is_save_(parm_gmm["is_save"].as<bool>())
    , platform_(platform)
    , use_eigen_(true) // Use Eigen by default
{
}

std::vector<uint32_t> GammaCorrection::generate_gamma_lut(int bit_depth) {
    int max_val = (1 << bit_depth) - 1;
    std::vector<uint32_t> lut(max_val + 1);
    for (int i = 0; i <= max_val; ++i) {
        lut[i] = static_cast<uint32_t>(std::round(max_val * std::pow(static_cast<double>(i) / max_val, 1.0 / 2.2)));
    }
    return lut;
}

cv::Mat GammaCorrection::apply_gamma_opencv() {
    auto lut = generate_gamma_lut(output_bit_depth_);
    cv::Mat gamma_img = img_.clone();
    std::vector<cv::Mat> channels;
    cv::split(gamma_img, channels);
    for (auto& channel : channels) {
        cv::MatIterator_<uint16_t> it = channel.begin<uint16_t>();
        cv::MatIterator_<uint16_t> end = channel.end<uint16_t>();
        for (; it != end; ++it) {
            *it = static_cast<uint16_t>(lut[*it]);
        }
    }
    cv::merge(channels, gamma_img);
    return gamma_img;
}

hdr_isp::EigenImage GammaCorrection::apply_gamma_eigen() {
    auto lut = generate_gamma_lut(output_bit_depth_);
    hdr_isp::EigenImage eigen_img = hdr_isp::EigenImage::fromOpenCV(img_);
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    // Apply LUT to each pixel
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int val = static_cast<int>(eigen_img.data()(i, j));
            if (val >= 0 && val < static_cast<int>(lut.size()))
                eigen_img.data()(i, j) = static_cast<float>(lut[val]);
        }
    }
    return eigen_img;
}

void GammaCorrection::save() {
    if (is_save_) {
        fs::path output_dir = "out_frames";
        fs::create_directories(output_dir);
        std::string in_file = platform_["in_file"].as<std::string>();
        std::string bayer_pattern = sensor_info_["bayer_pattern"].as<std::string>();
        fs::path output_path = output_dir / ("Out_gamma_correction_" + in_file + "_" + bayer_pattern + ".png");
        cv::imwrite(output_path.string(), img_);
    }
}

cv::Mat GammaCorrection::execute() {
    if (enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        if (use_eigen_) {
            hdr_isp::EigenImage result = apply_gamma_eigen();
            img_ = result.toOpenCV(img_.type());
        } else {
            img_ = apply_gamma_opencv();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "  Execution time: " << elapsed.count() << "s" << std::endl;
    }
    save();
    return img_;
} 
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
    , is_debug_(parm_gmm["is_debug"].as<bool>())
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

cv::Mat GammaCorrection::apply_gamma_eigen() {
    auto lut = generate_gamma_lut(output_bit_depth_);
    
    // Check if image is multi-channel (RGB after demosaicing)
    if (img_.channels() == 3) {
        // Use EigenImage3C for 3-channel RGB image
        hdr_isp::EigenImage3C eigen_img = hdr_isp::EigenImage3C::fromOpenCV(img_);
        int rows = eigen_img.rows();
        int cols = eigen_img.cols();
        
        // Apply LUT to each channel
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // Apply to R channel
                int r_val = static_cast<int>(eigen_img.r().data()(i, j));
                if (r_val >= 0 && r_val < static_cast<int>(lut.size()))
                    eigen_img.r().data()(i, j) = static_cast<float>(lut[r_val]);
                
                // Apply to G channel
                int g_val = static_cast<int>(eigen_img.g().data()(i, j));
                if (g_val >= 0 && g_val < static_cast<int>(lut.size()))
                    eigen_img.g().data()(i, j) = static_cast<float>(lut[g_val]);
                
                // Apply to B channel
                int b_val = static_cast<int>(eigen_img.b().data()(i, j));
                if (b_val >= 0 && b_val < static_cast<int>(lut.size()))
                    eigen_img.b().data()(i, j) = static_cast<float>(lut[b_val]);
            }
        }
        
        // Convert back to OpenCV Mat
        return eigen_img.toOpenCV(img_.type());
    } else {
        // Single-channel image (before demosaicing)
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
        
        // Convert back to OpenCV Mat
        return eigen_img.toOpenCV(img_.type());
    }
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
            img_ = apply_gamma_eigen();
        } else {
            img_ = apply_gamma_opencv();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    return img_;
} 
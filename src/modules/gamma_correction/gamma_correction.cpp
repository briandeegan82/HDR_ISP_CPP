#include "gamma_correction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

GammaCorrection::GammaCorrection(const hdr_isp::EigenImage3C& img, const YAML::Node& platform,
                               const YAML::Node& sensor_info, const YAML::Node& parm_gmm)
    : img_(img)
    , enable_(parm_gmm["is_enable"].as<bool>())
    , sensor_info_(sensor_info)
    , output_bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , parm_gmm_(parm_gmm)
    , is_save_(parm_gmm["is_save"].as<bool>())
    , is_debug_(parm_gmm["is_debug"].as<bool>())
    , platform_(platform)
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

hdr_isp::EigenImage3C GammaCorrection::apply_gamma() {
    auto lut = generate_gamma_lut(output_bit_depth_);
    
    // Create a copy of the input image
    hdr_isp::EigenImage3C result = img_.clone();
    int rows = result.rows();
    int cols = result.cols();
    
    // Apply LUT to each channel
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Apply to R channel
            int r_val = static_cast<int>(result.r().data()(i, j));
            if (r_val >= 0 && r_val < static_cast<int>(lut.size()))
                result.r().data()(i, j) = static_cast<float>(lut[r_val]);
            
            // Apply to G channel
            int g_val = static_cast<int>(result.g().data()(i, j));
            if (g_val >= 0 && g_val < static_cast<int>(lut.size()))
                result.g().data()(i, j) = static_cast<float>(lut[g_val]);
            
            // Apply to B channel
            int b_val = static_cast<int>(result.b().data()(i, j));
            if (b_val >= 0 && b_val < static_cast<int>(lut.size()))
                result.b().data()(i, j) = static_cast<float>(lut[b_val]);
        }
    }
    
    return result;
}

void GammaCorrection::save() {
    if (is_save_) {
        fs::path output_dir = "out_frames";
        fs::create_directories(output_dir);
        std::string in_file = platform_["in_file"].as<std::string>();
        std::string bayer_pattern = sensor_info_["bayer_pattern"].as<std::string>();
        fs::path output_path = output_dir / ("Out_gamma_correction_" + in_file + "_" + bayer_pattern + ".png");
        
        // Convert to OpenCV for saving
        cv::Mat opencv_img = img_.toOpenCV(CV_32FC3);
        cv::imwrite(output_path.string(), opencv_img);
    }
}

hdr_isp::EigenImage3C GammaCorrection::execute() {
    if (enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        img_ = apply_gamma();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    return img_;
} 
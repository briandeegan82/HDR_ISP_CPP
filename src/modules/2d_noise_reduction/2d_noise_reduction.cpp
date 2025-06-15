#include "2d_noise_reduction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

NoiseReduction2D::NoiseReduction2D(const cv::Mat& img, const YAML::Node& platform,
                                  const YAML::Node& sensor_info, const YAML::Node& params)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , params_(params)
    , is_enable_(params["is_enable"].as<bool>())
    , is_save_(params["is_save"].as<bool>())
    , is_debug_(params["is_debug"].as<bool>())
    , sigma_space_(params["sigma_space"].as<float>())
    , sigma_color_(params["sigma_color"].as<float>())
    , window_size_(params["window_size"].as<int>())
    , output_bit_depth_(sensor_info["output_bit_depth"].as<int>())
{
}

cv::Mat NoiseReduction2D::apply_bilateral_filter(const cv::Mat& img) {
    cv::Mat filtered;
    cv::bilateralFilter(img, filtered, window_size_, sigma_color_, sigma_space_);
    return filtered;
}

cv::Mat NoiseReduction2D::apply_noise_reduction() {
    // Convert to float for processing
    cv::Mat float_img;
    img_.convertTo(float_img, CV_32F);
    
    // Apply bilateral filter
    cv::Mat filtered = apply_bilateral_filter(float_img);
    
    // Convert back to original bit depth
    cv::Mat result;
    if (output_bit_depth_ == 8) {
        filtered.convertTo(result, CV_8U, 255.0);
    }
    else if (output_bit_depth_ == 16) {
        filtered.convertTo(result, CV_16U, 65535.0);
    }
    else if (output_bit_depth_ == 32) {
        filtered.convertTo(result, CV_32F);
    }
    else {
        throw std::runtime_error("Unsupported output bit depth. Use 8, 16, or 32.");
    }
    
    return result;
}

void NoiseReduction2D::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_2d_noise_reduction_" + 
                                 std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + ".png";
        cv::imwrite(output_path, img_);
    }
}

cv::Mat NoiseReduction2D::execute() {
    if (is_enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        img_ = apply_noise_reduction();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    save();
    return img_;
} 
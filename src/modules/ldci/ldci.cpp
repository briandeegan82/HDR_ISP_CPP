#include "ldci.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

LDCI::LDCI(const cv::Mat& img, const YAML::Node& platform,
           const YAML::Node& sensor_info, const YAML::Node& params)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , params_(params)
    , is_enable_(params["is_enable"].as<bool>())
    , is_save_(params["is_save"].as<bool>())
    , is_debug_(params["is_debug"].as<bool>())
    , strength_(params["strength"].as<float>())
    , window_size_(params["window_size"].as<int>())
    , output_bit_depth_(sensor_info["output_bit_depth"].as<int>())
{
}

cv::Mat LDCI::calculate_local_contrast(const cv::Mat& img) {
    cv::Mat local_mean;
    cv::boxFilter(img, local_mean, CV_32F, cv::Size(window_size_, window_size_));
    
    cv::Mat local_std;
    cv::Mat img_squared;
    cv::multiply(img, img, img_squared);
    cv::boxFilter(img_squared, local_std, CV_32F, cv::Size(window_size_, window_size_));
    
    cv::Mat local_mean_squared;
    cv::multiply(local_mean, local_mean, local_mean_squared);
    local_std = local_std - local_mean_squared;
    cv::sqrt(local_std, local_std);
    
    return local_std;
}

cv::Mat LDCI::enhance_contrast(const cv::Mat& img, const cv::Mat& local_contrast) {
    cv::Mat enhanced;
    cv::Mat local_mean;
    cv::boxFilter(img, local_mean, CV_32F, cv::Size(window_size_, window_size_));
    
    // Calculate enhancement factor based on local contrast
    cv::Mat enhancement_factor;
    cv::divide(local_contrast, local_mean + 1e-6, enhancement_factor);
    enhancement_factor = 1.0 + strength_ * enhancement_factor;
    
    // Apply enhancement
    cv::multiply(img - local_mean, enhancement_factor, enhanced);
    enhanced += local_mean;
    
    return enhanced;
}

cv::Mat LDCI::apply_ldci() {
    // Convert to float for processing
    cv::Mat float_img;
    img_.convertTo(float_img, CV_32F);
    
    // Calculate local contrast
    cv::Mat local_contrast = calculate_local_contrast(float_img);
    
    // Enhance contrast
    cv::Mat enhanced = enhance_contrast(float_img, local_contrast);
    
    // Convert back to original bit depth
    cv::Mat result;
    if (output_bit_depth_ == 8) {
        enhanced.convertTo(result, CV_8U, 255.0);
    }
    else if (output_bit_depth_ == 16) {
        enhanced.convertTo(result, CV_16U, 65535.0);
    }
    else if (output_bit_depth_ == 32) {
        enhanced.convertTo(result, CV_32F);
    }
    else {
        throw std::runtime_error("Unsupported output bit depth. Use 8, 16, or 32.");
    }
    
    return result;
}

void LDCI::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_ldci_" + 
                                 std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + ".png";
        cv::imwrite(output_path, img_);
    }
}

cv::Mat LDCI::execute() {
    if (is_enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        img_ = apply_ldci();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    save();
    return img_;
} 
#include "ldci.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <algorithm>

namespace fs = std::filesystem;

LDCI::LDCI(const hdr_isp::EigenImage3C& img, const YAML::Node& platform,
           const YAML::Node& sensor_info, const YAML::Node& params)
    : eigen_img_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , params_(params)
    , is_enable_(params["is_enable"].as<bool>())
    , is_save_(params["is_save"].as<bool>())
    , is_debug_(params["is_debug"].as<bool>())
    , strength_(params["clip_limit"].as<float>())
    , window_size_(params["wind"].as<int>())
    , output_bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , fp_config_(params)  // Initialize with params
    , use_fixed_point_(false)
    , use_fixed_input_(false)
{
    std::cout << "LDCI - Constructor started (floating-point)" << std::endl;
    std::cout << "LDCI - Input image size: " << img.rows() << "x" << img.cols() << std::endl;
    std::cout << "LDCI - Enable: " << (is_enable_ ? "true" : "false") << std::endl;
    std::cout << "LDCI - Strength (clip limit): " << strength_ << std::endl;
    std::cout << "LDCI - Window size (tile size): " << window_size_ << std::endl;
    std::cout << "LDCI - Output bit depth: " << output_bit_depth_ << std::endl;
    
    // Print input image statistics
    float min_val = std::min({img.r().min(), img.g().min(), img.b().min()});
    float max_val = std::max({img.r().max(), img.g().max(), img.b().max()});
    float mean_val = (img.r().mean() + img.g().mean() + img.b().mean()) / 3.0f;
    std::cout << "LDCI - Input image - Mean: " << mean_val << ", Min: " << min_val << ", Max: " << max_val << std::endl;
    
    std::cout << "LDCI - Constructor completed" << std::endl;
}

LDCI::LDCI(const hdr_isp::EigenImage3CFixed& img, const YAML::Node& platform,
           const YAML::Node& sensor_info, const YAML::Node& params,
           const hdr_isp::FixedPointConfig& fp_config)
    : eigen_img_fixed_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , params_(params)
    , is_enable_(params["is_enable"].as<bool>())
    , is_save_(params["is_save"].as<bool>())
    , is_debug_(params["is_debug"].as<bool>())
    , strength_(params["clip_limit"].as<float>())
    , window_size_(params["wind"].as<int>())
    , output_bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , fp_config_(fp_config)
    , use_fixed_point_(fp_config.isEnabled())
    , use_fixed_input_(true)
{
    std::cout << "LDCI - Constructor started (fixed-point)" << std::endl;
    std::cout << "LDCI - Input image size: " << img.rows() << "x" << img.cols() << std::endl;
    std::cout << "LDCI - Enable: " << (is_enable_ ? "true" : "false") << std::endl;
    std::cout << "LDCI - Strength (clip limit): " << strength_ << std::endl;
    std::cout << "LDCI - Window size (tile size): " << window_size_ << std::endl;
    std::cout << "LDCI - Output bit depth: " << output_bit_depth_ << std::endl;
    std::cout << "LDCI - Fixed-point enabled: " << (use_fixed_point_ ? "true" : "false") << std::endl;
    std::cout << "LDCI - Fractional bits: " << fp_config_.getFractionalBits() << std::endl;
    
    // Print input image statistics
    int16_t min_val = std::min({img.r().min(), img.g().min(), img.b().min()});
    int16_t max_val = std::max({img.r().max(), img.g().max(), img.b().max()});
    float mean_val = (img.r().mean() + img.g().mean() + img.b().mean()) / 3.0f;
    std::cout << "LDCI - Input image - Mean: " << mean_val << ", Min: " << min_val << ", Max: " << max_val << std::endl;
    
    std::cout << "LDCI - Constructor completed" << std::endl;
}

hdr_isp::EigenImage3C LDCI::apply_ldci_opencv() {
    std::cout << "LDCI - apply_ldci_opencv() started" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Convert Eigen to OpenCV
    cv::Mat opencv_img = eigen_img_.toOpenCV(CV_32FC3);
    
    // Find the maximum value in the input image for proper scaling
    double min_val, max_val;
    cv::minMaxLoc(opencv_img, &min_val, &max_val);
    
    // Normalize to 0-255 range for CLAHE processing
    cv::Mat normalized;
    if (max_val > 0) {
        cv::normalize(opencv_img, normalized, 0, 255, cv::NORM_MINMAX);
    } else {
        normalized = opencv_img.clone();
    }
    normalized.convertTo(normalized, CV_8UC3);
    
    // Convert to LAB color space for better contrast enhancement
    cv::Mat lab;
    cv::cvtColor(normalized, lab, cv::COLOR_BGR2Lab);
    
    // Split channels
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab, lab_planes);
    
    // Apply CLAHE to L channel (lightness) only
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(strength_, cv::Size(window_size_, window_size_));
    clahe->apply(lab_planes[0], lab_planes[0]);
    
    // Merge channels back
    cv::merge(lab_planes, lab);
    
    // Convert back to BGR
    cv::Mat result;
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    
    // Convert back to float
    cv::Mat result_float;
    result.convertTo(result_float, CV_32F);
    
    // Scale back to original range
    if (max_val > 0) {
        result_float = result_float * (max_val / 255.0f);
    }
    
    // Convert back to Eigen
    hdr_isp::EigenImage3C eigen_result = hdr_isp::EigenImage3C::fromOpenCV(result_float);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "LDCI - apply_ldci_opencv() completed in " << duration.count() / 1000.0 << " ms" << std::endl;
    return eigen_result;
}

hdr_isp::EigenImage3CFixed LDCI::apply_ldci_fixed() {
    std::cout << "LDCI - apply_ldci_fixed() started" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Convert fixed-point to floating-point for OpenCV processing
    hdr_isp::EigenImage3C float_img = eigen_img_fixed_.toEigenImage3C(fp_config_.getFractionalBits());
    
    // Store original image for processing
    eigen_img_ = float_img;
    
    // Apply CLAHE using floating-point version
    hdr_isp::EigenImage3C enhanced_float = apply_ldci_opencv();
    
    // Convert back to fixed-point
    hdr_isp::EigenImage3CFixed result = hdr_isp::EigenImage3CFixed::fromEigenImage3C(enhanced_float, fp_config_.getFractionalBits());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "LDCI - apply_ldci_fixed() completed in " << duration.count() / 1000.0 << " ms" << std::endl;
    return result;
}

hdr_isp::EigenImage3C LDCI::execute() {
    if (!is_enable_) {
        std::cout << "LDCI - Module disabled, returning input image" << std::endl;
        return eigen_img_;
    }
    
    std::cout << "LDCI - execute() started" << std::endl;
    
    hdr_isp::EigenImage3C result = apply_ldci_opencv();
    
    if (is_save_) {
        save();
    }
    
    std::cout << "LDCI - execute() completed" << std::endl;
    return result;
}

hdr_isp::EigenImage3CFixed LDCI::execute_fixed() {
    if (!is_enable_) {
        std::cout << "LDCI - Module disabled, returning input image" << std::endl;
        return eigen_img_fixed_;
    }
    
    std::cout << "LDCI - execute_fixed() started" << std::endl;
    
    hdr_isp::EigenImage3CFixed result = apply_ldci_fixed();
    
    if (is_save_) {
        save_fixed();
    }
    
    std::cout << "LDCI - execute_fixed() completed" << std::endl;
    return result;
}

void LDCI::save() {
    if (is_save_) {
        std::cout << "LDCI - save() started" << std::endl;
        
        fs::path output_dir = "out_frames";
        fs::create_directories(output_dir);
        
        // Try to get filename from platform config
        std::string in_file;
        if (platform_["filename"]) {
            in_file = platform_["filename"].as<std::string>();
        } else {
            in_file = "unknown";
        }
        
        std::string bayer_pattern = sensor_info_["bayer_pattern"].as<std::string>();
        
        // Create output filename
        fs::path in_path(in_file);
        std::string base_name = in_path.stem().string();
        std::string output_filename = base_name + "_ldci.png";
        fs::path output_path = output_dir / output_filename;
        
        // Convert to OpenCV format for saving
        cv::Mat opencv_img = eigen_img_.toOpenCV(CV_8UC3);
        
        // Save image
        bool success = cv::imwrite(output_path.string(), opencv_img);
        if (success) {
            std::cout << "LDCI - Saved output image: " << output_path.string() << std::endl;
        } else {
            std::cout << "LDCI - Failed to save output image: " << output_path.string() << std::endl;
        }
        
        std::cout << "LDCI - save() completed" << std::endl;
    }
}

void LDCI::save_fixed() {
    if (is_save_) {
        std::cout << "LDCI - save_fixed() started" << std::endl;
        
        fs::path output_dir = "out_frames";
        fs::create_directories(output_dir);
        
        // Try to get filename from platform config
        std::string in_file;
        if (platform_["filename"]) {
            in_file = platform_["filename"].as<std::string>();
        } else {
            in_file = "unknown";
        }
        
        std::string bayer_pattern = sensor_info_["bayer_pattern"].as<std::string>();
        
        // Create output filename
        fs::path in_path(in_file);
        std::string base_name = in_path.stem().string();
        std::string output_filename = base_name + "_ldci_fixed.png";
        fs::path output_path = output_dir / output_filename;
        
        // Convert fixed-point to floating-point for saving
        hdr_isp::EigenImage3C float_img = eigen_img_fixed_.toEigenImage3C(fp_config_.getFractionalBits());
        
        // Convert to OpenCV format for saving
        cv::Mat opencv_img = float_img.toOpenCV(CV_8UC3);
        
        // Save image
        bool success = cv::imwrite(output_path.string(), opencv_img);
        if (success) {
            std::cout << "LDCI - Saved fixed-point output image: " << output_path.string() << std::endl;
        } else {
            std::cout << "LDCI - Failed to save fixed-point output image: " << output_path.string() << std::endl;
        }
        
        std::cout << "LDCI - save_fixed() completed" << std::endl;
    }
} 
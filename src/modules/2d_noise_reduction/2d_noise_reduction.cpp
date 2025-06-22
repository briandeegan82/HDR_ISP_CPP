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
    , is_enable_(params["is_enable"].IsDefined() ? params["is_enable"].as<bool>() : false)
    , is_save_(params["is_save"].IsDefined() ? params["is_save"].as<bool>() : false)
    , is_debug_(params["is_debug"].IsDefined() ? params["is_debug"].as<bool>() : false)
    , sigma_space_(params["sigma_space"].IsDefined() ? params["sigma_space"].as<float>() : 1.0f)
    , sigma_color_(params["sigma_color"].IsDefined() ? params["sigma_color"].as<float>() : 1.0f)
    , window_size_(params["window_size"].IsDefined() ? params["window_size"].as<int>() : 3)
    , output_bit_depth_(sensor_info["output_bit_depth"].IsDefined() ? sensor_info["output_bit_depth"].as<int>() : 16)
    , use_eigen_(true) // Use Eigen by default
{
}

NoiseReduction2D::NoiseReduction2D(const hdr_isp::EigenImage3C& img, const YAML::Node& platform,
                                  const YAML::Node& sensor_info, const YAML::Node& params)
    : eigen_img_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , params_(params)
    , is_enable_(params["is_enable"].IsDefined() ? params["is_enable"].as<bool>() : false)
    , is_save_(params["is_save"].IsDefined() ? params["is_save"].as<bool>() : false)
    , is_debug_(params["is_debug"].IsDefined() ? params["is_debug"].as<bool>() : false)
    , sigma_space_(params["sigma_space"].IsDefined() ? params["sigma_space"].as<float>() : 1.0f)
    , sigma_color_(params["sigma_color"].IsDefined() ? params["sigma_color"].as<float>() : 1.0f)
    , window_size_(params["window_size"].IsDefined() ? params["window_size"].as<int>() : 3)
    , output_bit_depth_(sensor_info["output_bit_depth"].IsDefined() ? sensor_info["output_bit_depth"].as<int>() : 16)
    , use_eigen_(true)
    , has_eigen_input_(true)
{
}

cv::Mat NoiseReduction2D::apply_bilateral_filter(const cv::Mat& img) {
    cv::Mat filtered;
    cv::bilateralFilter(img, filtered, window_size_, sigma_color_, sigma_space_);
    return filtered;
}

hdr_isp::EigenImage NoiseReduction2D::apply_bilateral_filter_eigen(const hdr_isp::EigenImage& img) {
    // Simplified bilateral filter using Eigen
    // In a full implementation, you'd implement proper bilateral filtering
    int rows = img.rows();
    int cols = img.cols();
    
    // Simple Gaussian blur as approximation
    hdr_isp::EigenImage filtered = hdr_isp::EigenImage::Zero(rows, cols);
    
    // Simple 3x3 Gaussian kernel
    Eigen::Matrix3f kernel;
    kernel << 1, 2, 1,
              2, 4, 2,
              1, 2, 1;
    kernel = kernel / 16.0f;
    
    // Apply convolution
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            float sum = 0.0f;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += img.data()(i + ki, j + kj) * kernel(ki + 1, kj + 1);
                }
            }
            filtered.data()(i, j) = sum;
        }
    }
    
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

hdr_isp::EigenImage NoiseReduction2D::apply_noise_reduction_eigen() {
    // Check if image is single-channel or multi-channel
    if (img_.channels() == 1) {
        // Single-channel processing
        hdr_isp::EigenImage eigen_img = hdr_isp::opencv_to_eigen(img_);
        
        // Apply bilateral filter
        hdr_isp::EigenImage filtered = apply_bilateral_filter_eigen(eigen_img);
        
        // Apply bit depth scaling
        if (output_bit_depth_ == 8) {
            filtered = filtered * 255.0f;
            filtered = filtered.cwiseMax(0.0f).cwiseMin(255.0f);
        }
        else if (output_bit_depth_ == 16) {
            filtered = filtered * 65535.0f;
            filtered = filtered.cwiseMax(0.0f).cwiseMin(65535.0f);
        }
        else if (output_bit_depth_ == 32) {
            // Keep as float
        }
        else {
            throw std::runtime_error("Unsupported output bit depth. Use 8, 16, or 32.");
        }
        
        return filtered;
    } else if (img_.channels() == 3) {
        // Multi-channel processing - convert to grayscale for noise reduction
        cv::Mat gray_img;
        cv::cvtColor(img_, gray_img, cv::COLOR_BGR2GRAY);
        
        hdr_isp::EigenImage eigen_img = hdr_isp::opencv_to_eigen(gray_img);
        
        // Apply bilateral filter
        hdr_isp::EigenImage filtered = apply_bilateral_filter_eigen(eigen_img);
        
        // Apply bit depth scaling
        if (output_bit_depth_ == 8) {
            filtered = filtered * 255.0f;
            filtered = filtered.cwiseMax(0.0f).cwiseMin(255.0f);
        }
        else if (output_bit_depth_ == 16) {
            filtered = filtered * 65535.0f;
            filtered = filtered.cwiseMax(0.0f).cwiseMin(65535.0f);
        }
        else if (output_bit_depth_ == 32) {
            // Keep as float
        }
        else {
            throw std::runtime_error("Unsupported output bit depth. Use 8, 16, or 32.");
        }
        
        return filtered;
    } else {
        throw std::runtime_error("Unsupported number of channels. Use 1 or 3 channels.");
    }
}

hdr_isp::EigenImage3C NoiseReduction2D::apply_noise_reduction_eigen_3c() {
    // Use the appropriate input (Eigen or converted from OpenCV)
    hdr_isp::EigenImage3C eigen_img;
    if (has_eigen_input_) {
        eigen_img = eigen_img_;
    } else {
        eigen_img = hdr_isp::EigenImage3C::fromOpenCV(img_);
    }
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    // Process each channel independently
    hdr_isp::EigenImage3C result(rows, cols);
    result.r() = apply_bilateral_filter_eigen(eigen_img.r());
    result.g() = apply_bilateral_filter_eigen(eigen_img.g());
    result.b() = apply_bilateral_filter_eigen(eigen_img.b());
    // Apply bit depth scaling
    if (output_bit_depth_ == 8) {
        result.r() = result.r() * 255.0f;
        result.g() = result.g() * 255.0f;
        result.b() = result.b() * 255.0f;
        result.r() = result.r().cwiseMax(0.0f).cwiseMin(255.0f);
        result.g() = result.g().cwiseMax(0.0f).cwiseMin(255.0f);
        result.b() = result.b().cwiseMax(0.0f).cwiseMin(255.0f);
    } else if (output_bit_depth_ == 16) {
        result.r() = result.r() * 65535.0f;
        result.g() = result.g() * 65535.0f;
        result.b() = result.b() * 65535.0f;
        result.r() = result.r().cwiseMax(0.0f).cwiseMin(65535.0f);
        result.g() = result.g().cwiseMax(0.0f).cwiseMin(65535.0f);
        result.b() = result.b().cwiseMax(0.0f).cwiseMin(65535.0f);
    }
    // For 32-bit, keep as float
    return result;
}

hdr_isp::EigenImage3C NoiseReduction2D::execute_eigen() {
    if (is_enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        hdr_isp::EigenImage3C result = apply_noise_reduction_eigen_3c();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (is_debug_) {
            std::cout << "  Eigen execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
        return result;
    }
    if (has_eigen_input_) {
        return eigen_img_;
    } else {
        return hdr_isp::EigenImage3C::fromOpenCV(img_);
    }
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
        
        if (use_eigen_) {
            hdr_isp::EigenImage result = apply_noise_reduction_eigen();
            img_ = hdr_isp::eigen_to_opencv(result);
        } else {
            img_ = apply_noise_reduction();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    return img_;
} 
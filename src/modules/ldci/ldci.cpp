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
    , strength_(params["clip_limit"].as<float>())
    , window_size_(params["wind"].as<int>())
    , output_bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , use_eigen_(true) // Use Eigen by default
    , has_eigen_input_(false)
{
}

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
    , use_eigen_(true) // Use Eigen by default
    , has_eigen_input_(true)
{
}

cv::Mat LDCI::calculate_local_contrast_opencv(const cv::Mat& img) {
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

hdr_isp::EigenImage LDCI::calculate_local_contrast_eigen(const hdr_isp::EigenImage& img) {
    int rows = img.rows();
    int cols = img.cols();
    int pad = window_size_ / 2;
    
    hdr_isp::EigenImage local_mean = hdr_isp::EigenImage::Zero(rows, cols);
    hdr_isp::EigenImage local_std = hdr_isp::EigenImage::Zero(rows, cols);
    
    // Calculate local mean and variance using sliding window
    for (int i = pad; i < rows - pad; i++) {
        for (int j = pad; j < cols - pad; j++) {
            float sum = 0.0f;
            float sum_sq = 0.0f;
            int count = 0;
            
            for (int ki = -pad; ki <= pad; ki++) {
                for (int kj = -pad; kj <= pad; kj++) {
                    float val = img(i + ki, j + kj);
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }
            
            local_mean(i, j) = sum / count;
            float variance = (sum_sq / count) - (local_mean(i, j) * local_mean(i, j));
            local_std(i, j) = std::sqrt(std::max(0.0f, variance));
        }
    }
    
    return local_std;
}

cv::Mat LDCI::enhance_contrast_opencv(const cv::Mat& img, const cv::Mat& local_contrast) {
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

hdr_isp::EigenImage LDCI::enhance_contrast_eigen(const hdr_isp::EigenImage& img, const hdr_isp::EigenImage& local_contrast) {
    int rows = img.rows();
    int cols = img.cols();
    int pad = window_size_ / 2;
    
    hdr_isp::EigenImage local_mean = hdr_isp::EigenImage::Zero(rows, cols);
    hdr_isp::EigenImage enhanced = hdr_isp::EigenImage::Zero(rows, cols);
    
    // Calculate local mean
    for (int i = pad; i < rows - pad; i++) {
        for (int j = pad; j < cols - pad; j++) {
            float sum = 0.0f;
            int count = 0;
            
            for (int ki = -pad; ki <= pad; ki++) {
                for (int kj = -pad; kj <= pad; kj++) {
                    sum += img(i + ki, j + kj);
                    count++;
                }
            }
            
            local_mean(i, j) = sum / count;
        }
    }
    
    // Calculate enhancement factor and apply enhancement
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float enhancement_factor = 1.0f + strength_ * (local_contrast(i, j) / (local_mean(i, j) + 1e-6f));
            enhanced(i, j) = local_mean(i, j) + enhancement_factor * (img(i, j) - local_mean(i, j));
        }
    }
    
    return enhanced;
}

cv::Mat LDCI::apply_ldci_opencv() {
    // Convert to float for processing
    cv::Mat float_img;
    img_.convertTo(float_img, CV_32F);
    
    // Calculate local contrast
    cv::Mat local_contrast = calculate_local_contrast_opencv(float_img);
    
    // Enhance contrast
    cv::Mat enhanced = enhance_contrast_opencv(float_img, local_contrast);
    
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

hdr_isp::EigenImage3C LDCI::apply_ldci_eigen() {
    // Use the appropriate input (Eigen or converted from OpenCV)
    hdr_isp::EigenImage3C eigen_img;
    if (has_eigen_input_) {
        eigen_img = eigen_img_;
    } else {
        eigen_img = hdr_isp::EigenImage3C::fromOpenCV(img_);
    }
    
    // Process each channel separately
    hdr_isp::EigenImage3C result;
    result.r() = hdr_isp::EigenImage::Zero(eigen_img.rows(), eigen_img.cols());
    result.g() = hdr_isp::EigenImage::Zero(eigen_img.rows(), eigen_img.cols());
    result.b() = hdr_isp::EigenImage::Zero(eigen_img.rows(), eigen_img.cols());
    
    // Process R channel
    hdr_isp::EigenImage local_contrast_r = calculate_local_contrast_eigen(eigen_img.r());
    result.r() = enhance_contrast_eigen(eigen_img.r(), local_contrast_r);
    
    // Process G channel
    hdr_isp::EigenImage local_contrast_g = calculate_local_contrast_eigen(eigen_img.g());
    result.g() = enhance_contrast_eigen(eigen_img.g(), local_contrast_g);
    
    // Process B channel
    hdr_isp::EigenImage local_contrast_b = calculate_local_contrast_eigen(eigen_img.b());
    result.b() = enhance_contrast_eigen(eigen_img.b(), local_contrast_b);
    
    // Apply bit depth conversion
    if (output_bit_depth_ == 8) {
        result.r() = result.r().cwiseMax(0.0f).cwiseMin(255.0f);
        result.g() = result.g().cwiseMax(0.0f).cwiseMin(255.0f);
        result.b() = result.b().cwiseMax(0.0f).cwiseMin(255.0f);
    } else if (output_bit_depth_ == 16) {
        result.r() = result.r().cwiseMax(0.0f).cwiseMin(65535.0f);
        result.g() = result.g().cwiseMax(0.0f).cwiseMin(65535.0f);
        result.b() = result.b().cwiseMax(0.0f).cwiseMin(65535.0f);
    }
    // For 32-bit, no clipping needed
    
    return result;
}

cv::Mat LDCI::apply_ldci_multi_channel() {
    // Split the image into channels
    std::vector<cv::Mat> channels;
    cv::split(img_, channels);
    
    // Process each channel separately
    std::vector<cv::Mat> processed_channels;
    for (const auto& channel : channels) {
        // Create a temporary LDCI instance for single channel processing
        cv::Mat single_channel = channel.clone();
        
        // Convert to float for processing
        cv::Mat float_img;
        single_channel.convertTo(float_img, CV_32F);
        
        // Calculate local contrast
        cv::Mat local_contrast = calculate_local_contrast_opencv(float_img);
        
        // Enhance contrast
        cv::Mat enhanced = enhance_contrast_opencv(float_img, local_contrast);
        
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
        
        processed_channels.push_back(result);
    }
    
    // Merge the processed channels back
    cv::Mat result;
    cv::merge(processed_channels, result);
    
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
        
        // Check if the image is multi-channel
        if (img_.channels() > 1) {
            // Use multi-channel processing
            img_ = apply_ldci_multi_channel();
        } else {
            // Use single-channel processing
            if (use_eigen_) {
                hdr_isp::EigenImage3C eigen_result = apply_ldci_eigen();
                img_ = eigen_result.toOpenCV(CV_32FC3);
            } else {
                img_ = apply_ldci_opencv();
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    return img_;
}

hdr_isp::EigenImage3C LDCI::execute_eigen() {
    if (is_enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        hdr_isp::EigenImage3C result = apply_ldci_eigen();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Eigen execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
        
        return result;
    }

    // Return original image if not enabled
    if (has_eigen_input_) {
        return eigen_img_;
    } else {
        return hdr_isp::EigenImage3C::fromOpenCV(img_);
    }
} 
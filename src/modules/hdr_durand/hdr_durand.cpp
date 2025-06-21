#include "hdr_durand.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

HDRDurandToneMapping::HDRDurandToneMapping(const cv::Mat& img, const YAML::Node& platform,
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
    , contrast_factor_(params["contrast_factor"].as<float>())
    , downsample_factor_(params["downsample_factor"].as<int>())
    , output_bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , use_eigen_(true) // Use Eigen by default
{
}

cv::Mat HDRDurandToneMapping::normalize(const cv::Mat& image) {
    double min_val, max_val;
    cv::minMaxLoc(image, &min_val, &max_val);
    cv::Mat normalized;
    image.convertTo(normalized, CV_32F);
    normalized = (normalized - min_val) / (max_val - min_val);
    return normalized;
}

hdr_isp::EigenImage HDRDurandToneMapping::normalize_eigen(const hdr_isp::EigenImage& image) {
    float min_val = image.min();
    float max_val = image.max();
    hdr_isp::EigenImage normalized = (image - min_val) / (max_val - min_val);
    return normalized;
}

cv::Mat HDRDurandToneMapping::bilateral_filter(const cv::Mat& image, float sigma_color, float sigma_space) {
    cv::Mat spatial_filtered;
    cv::GaussianBlur(image, spatial_filtered, cv::Size(0, 0), sigma_space);

    cv::Mat intensity_diff = image - spatial_filtered;
    cv::Mat range_kernel;
    cv::exp(-0.5 * (intensity_diff / sigma_color).mul(intensity_diff / sigma_color), range_kernel);

    return spatial_filtered + range_kernel.mul(intensity_diff);
}

hdr_isp::EigenImage HDRDurandToneMapping::bilateral_filter_eigen(const hdr_isp::EigenImage& image, float sigma_color, float sigma_space) {
    // Simplified bilateral filter using Eigen
    // In a full implementation, you'd implement proper bilateral filtering
    int rows = image.rows();
    int cols = image.cols();
    
    // Simple Gaussian blur as approximation
    hdr_isp::EigenImage spatial_filtered = hdr_isp::EigenImage::Zero(rows, cols);
    
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
                    sum += image(i + ki, j + kj) * kernel(ki + 1, kj + 1);
                }
            }
            spatial_filtered(i, j) = sum;
        }
    }
    
    // Apply range filter
    hdr_isp::EigenImage intensity_diff = image - spatial_filtered;
    hdr_isp::EigenImage range_kernel = hdr_isp::EigenImage::Zero(rows, cols);
    range_kernel.data() = (-intensity_diff.data().array().square() / (2 * sigma_color * sigma_color)).exp();
    
    return spatial_filtered + range_kernel.cwiseProduct(intensity_diff);
}

cv::Mat HDRDurandToneMapping::fast_bilateral_filter(const cv::Mat& image) {
    // Downsample
    cv::Mat small_img;
    cv::resize(image, small_img, cv::Size(), 1.0/downsample_factor_, 1.0/downsample_factor_, cv::INTER_LINEAR);

    // Apply bilateral filter on downsampled image
    cv::Mat small_filtered = bilateral_filter(small_img, sigma_color_, sigma_space_);

    // Upsample
    cv::Mat result;
    cv::resize(small_filtered, result, image.size(), 0, 0, cv::INTER_LINEAR);
    return result;
}

hdr_isp::EigenImage HDRDurandToneMapping::fast_bilateral_filter_eigen(const hdr_isp::EigenImage& image) {
    // Downsample
    int small_rows = image.rows() / downsample_factor_;
    int small_cols = image.cols() / downsample_factor_;
    hdr_isp::EigenImage small_img = hdr_isp::EigenImage::Zero(small_rows, small_cols);
    
    // Simple downsampling by taking every nth pixel
    for (int i = 0; i < small_rows; i++) {
        for (int j = 0; j < small_cols; j++) {
            small_img(i, j) = image(i * downsample_factor_, j * downsample_factor_);
        }
    }

    // Apply bilateral filter on downsampled image
    hdr_isp::EigenImage small_filtered = bilateral_filter_eigen(small_img, sigma_color_, sigma_space_);

    // Upsample (simple nearest neighbor)
    hdr_isp::EigenImage result = hdr_isp::EigenImage::Zero(image.rows(), image.cols());
    for (int i = 0; i < image.rows(); i++) {
        for (int j = 0; j < image.cols(); j++) {
            int small_i = std::min(i / downsample_factor_, small_rows - 1);
            int small_j = std::min(j / downsample_factor_, small_cols - 1);
            result(i, j) = small_filtered(small_i, small_j);
        }
    }
    
    return result;
}

cv::Mat HDRDurandToneMapping::apply_tone_mapping() {
    // Convert to log domain
    const float epsilon = 1e-6f;
    cv::Mat log_luminance;
    img_.convertTo(log_luminance, CV_32F);
    log_luminance += epsilon;
    cv::log(log_luminance, log_luminance);
    log_luminance /= std::log(10.0f);  // Convert to log10

    // Apply bilateral filter to get the base layer
    cv::Mat log_base = bilateral_filter(log_luminance, sigma_color_, sigma_space_);

    // Extract the detail layer
    cv::Mat log_detail = log_luminance - log_base;

    // Compress the base layer
    cv::Mat compressed_log_base = log_base / contrast_factor_;

    // Recombine base and detail layers
    cv::Mat log_output = compressed_log_base + log_detail;

    // Convert back from log domain
    cv::Mat output_luminance;
    cv::exp(log_output * std::log(10.0f), output_luminance);

    // Normalize to [0, 1] range
    output_luminance = normalize(output_luminance);

    // Convert to desired bit depth
    cv::Mat result;
    if (output_bit_depth_ == 8) {
        output_luminance.convertTo(result, CV_8U, 255.0);
    }
    else if (output_bit_depth_ == 16) {
        output_luminance.convertTo(result, CV_16U, 65535.0);
    }
    else if (output_bit_depth_ == 32) {
        output_luminance.convertTo(result, CV_32F);
    }
    else {
        throw std::runtime_error("Unsupported output bit depth. Use 8, 16, or 32.");
    }

    return result;
}

hdr_isp::EigenImage HDRDurandToneMapping::apply_tone_mapping_eigen() {
    // Convert to Eigen
    hdr_isp::EigenImage eigen_img = hdr_isp::opencv_to_eigen(img_);
    
    // Convert to log domain
    const float epsilon = 1e-6f;
    hdr_isp::EigenImage log_luminance = eigen_img + epsilon;
    log_luminance.data() = log_luminance.data().array().log() / std::log(10.0f);

    // Apply bilateral filter to get the base layer
    hdr_isp::EigenImage log_base = bilateral_filter_eigen(log_luminance, sigma_color_, sigma_space_);

    // Extract the detail layer
    hdr_isp::EigenImage log_detail = log_luminance - log_base;

    // Compress the base layer
    hdr_isp::EigenImage compressed_log_base = log_base / contrast_factor_;

    // Recombine base and detail layers
    hdr_isp::EigenImage log_output = compressed_log_base + log_detail;

    // Convert back from log domain
    hdr_isp::EigenImage output_luminance = hdr_isp::EigenImage::Zero(log_output.rows(), log_output.cols());
    output_luminance.data() = (log_output.data().array() * std::log(10.0f)).exp();

    // Normalize to [0, 1] range
    output_luminance = normalize_eigen(output_luminance);

    // Apply bit depth scaling
    if (output_bit_depth_ == 8) {
        output_luminance = output_luminance * 255.0f;
        output_luminance = output_luminance.cwiseMax(0.0f).cwiseMin(255.0f);
    }
    else if (output_bit_depth_ == 16) {
        output_luminance = output_luminance * 65535.0f;
        output_luminance = output_luminance.cwiseMax(0.0f).cwiseMin(65535.0f);
    }
    else if (output_bit_depth_ == 32) {
        // Keep as float
    }
    else {
        throw std::runtime_error("Unsupported output bit depth. Use 8, 16, or 32.");
    }

    return output_luminance;
}

void HDRDurandToneMapping::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_hdr_durand_" + 
                                 std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + ".png";
        cv::imwrite(output_path, img_);
    }
}

cv::Mat HDRDurandToneMapping::execute() {
    if (is_enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (use_eigen_) {
            hdr_isp::EigenImage result = apply_tone_mapping_eigen();
            img_ = hdr_isp::eigen_to_opencv(result);
        } else {
            img_ = apply_tone_mapping();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    return img_;
} 
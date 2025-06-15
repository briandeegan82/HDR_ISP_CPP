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

cv::Mat HDRDurandToneMapping::bilateral_filter(const cv::Mat& image, float sigma_color, float sigma_space) {
    cv::Mat spatial_filtered;
    cv::GaussianBlur(image, spatial_filtered, cv::Size(0, 0), sigma_space);

    cv::Mat intensity_diff = image - spatial_filtered;
    cv::Mat range_kernel;
    cv::exp(-0.5 * (intensity_diff / sigma_color).mul(intensity_diff / sigma_color), range_kernel);

    return spatial_filtered + range_kernel.mul(intensity_diff);
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
        img_ = apply_tone_mapping();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    save();
    return img_;
} 
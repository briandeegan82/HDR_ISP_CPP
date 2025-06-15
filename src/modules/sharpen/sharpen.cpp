#include "sharpen.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>

Sharpen::Sharpen(const cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                 const YAML::Node& parm_shp, const std::string& conv_std)
    : img_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_shp_(parm_shp)
    , conv_std_(conv_std)
{
    get_sharpen_params();
}

void Sharpen::get_sharpen_params() {
    is_enable_ = parm_shp_["is_enable"].as<bool>();
    is_save_ = parm_shp_["is_save"].as<bool>();
    is_debug_ = parm_shp_["is_debug"].as<bool>();
    strength_ = parm_shp_["strength"].as<float>();
    kernel_size_ = parm_shp_["kernel_size"].as<int>();
    output_bit_depth_ = parm_shp_["output_bit_depth"].as<int>();
}

cv::Mat Sharpen::apply_sharpen() {
    if (is_debug_) {
        std::cout << "Applying sharpening with strength: " << strength_ 
                  << ", kernel size: " << kernel_size_ << std::endl;
    }

    // Convert to float for processing
    cv::Mat img_float;
    img_.convertTo(img_float, CV_32F);

    // Create sharpening kernel
    cv::Mat kernel = cv::Mat::zeros(kernel_size_, kernel_size_, CV_32F);
    float center = kernel_size_ / 2;
    float sigma = kernel_size_ / 6.0f;
    
    // Create Gaussian kernel
    for (int i = 0; i < kernel_size_; i++) {
        for (int j = 0; j < kernel_size_; j++) {
            float x = i - center;
            float y = j - center;
            kernel.at<float>(i, j) = std::exp(-(x*x + y*y) / (2 * sigma * sigma));
        }
    }
    
    // Normalize kernel
    kernel = kernel / cv::sum(kernel)[0];

    // Create sharpening kernel (Laplacian of Gaussian)
    cv::Mat laplacian = cv::Mat::zeros(kernel_size_, kernel_size_, CV_32F);
    laplacian.at<float>(center, center) = 1.0f;
    laplacian = laplacian - kernel;

    // Apply sharpening
    cv::Mat sharpened;
    cv::filter2D(img_float, sharpened, -1, laplacian);

    // Blend with original image
    cv::Mat result = img_float + strength_ * sharpened;

    // Convert back to original bit depth
    cv::Mat output;
    if (output_bit_depth_ == 8) {
        result.convertTo(output, CV_8U);
    } else if (output_bit_depth_ == 16) {
        result.convertTo(output, CV_16U);
    } else if (output_bit_depth_ == 32) {
        result.convertTo(output, CV_32F);
    } else {
        throw std::runtime_error("Unsupported output bit depth: " + std::to_string(output_bit_depth_));
    }

    return output;
}

void Sharpen::save(const std::string& filename) {
    if (is_save_) {
        std::filesystem::path output_path = "out_frames/intermediate";
        std::filesystem::create_directories(output_path);
        cv::imwrite((output_path / filename).string(), img_);
    }
}

cv::Mat Sharpen::execute() {
    if (!is_enable_) {
        if (is_debug_) {
            std::cout << "Sharpening is disabled" << std::endl;
        }
        return img_;
    }

    auto start = std::chrono::high_resolution_clock::now();

    img_ = apply_sharpen();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (is_debug_) {
        std::cout << "Sharpening completed in " << duration.count() << " ms" << std::endl;
    }

    save("sharpen.png");
    return img_;
} 
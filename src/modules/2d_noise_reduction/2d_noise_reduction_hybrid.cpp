#include "2d_noise_reduction_hybrid.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

NoiseReduction2DHybrid::NoiseReduction2DHybrid(const cv::Mat& img, const YAML::Node& platform,
                                               const YAML::Node& sensor_info, const YAML::Node& params)
    : NoiseReduction2D(img, platform, sensor_info, params)
{
}

NoiseReduction2DHybrid::NoiseReduction2DHybrid(const hdr_isp::EigenImage3C& img, const YAML::Node& platform,
                                               const YAML::Node& sensor_info, const YAML::Node& params)
    : NoiseReduction2D(img, platform, sensor_info, params)
{
}

Halide::Func NoiseReduction2DHybrid::create_bilateral_kernel() {
    Halide::Var x, y;
    
    // Create Gaussian spatial kernel
    Halide::Func spatial_kernel;
    
    // Calculate kernel size based on sigma_space
    int kernel_radius = static_cast<int>(std::ceil(3.0f * sigma_space_));
    int kernel_size = 2 * kernel_radius + 1;
    
    // Create spatial Gaussian kernel
    Halide::Expr x_center = kernel_radius;
    Halide::Expr y_center = kernel_radius;
    Halide::Expr x_dist = Halide::cast<float>(x - x_center);
    Halide::Expr y_dist = Halide::cast<float>(y - y_center);
    Halide::Expr spatial_weight = Halide::exp(-(x_dist * x_dist + y_dist * y_dist) / (2.0f * sigma_space_ * sigma_space_));
    
    spatial_kernel(x, y) = spatial_weight;
    
    // Schedule for optimal performance
    spatial_kernel.compute_root();
    
    return spatial_kernel;
}

Halide::Func NoiseReduction2DHybrid::apply_spatial_filter(Halide::Buffer<float> input, Halide::Func kernel) {
    Halide::Var x, y;
    
    // Calculate kernel size
    int kernel_radius = static_cast<int>(std::ceil(3.0f * sigma_space_));
    int kernel_size = 2 * kernel_radius + 1;
    
    // Create spatial filtering function
    Halide::Func spatial_filtered;
    Halide::RDom r(-kernel_radius, kernel_size, -kernel_radius, kernel_size);
    
    // Apply spatial filtering with boundary conditions
    Halide::Expr clamped_x = Halide::clamp(x + r.x, 0, input.width() - 1);
    Halide::Expr clamped_y = Halide::clamp(y + r.y, 0, input.height() - 1);
    
    Halide::Expr spatial_weight = kernel(r.x + kernel_radius, r.y + kernel_radius);
    Halide::Expr pixel_value = input(clamped_x, clamped_y);
    
    spatial_filtered(x, y) = Halide::sum(spatial_weight * pixel_value) / Halide::sum(spatial_weight);
    
    // Schedule for optimal performance
    spatial_filtered.vectorize(x, 8).parallel(y);
    
    return spatial_filtered;
}

Halide::Func NoiseReduction2DHybrid::apply_range_filter(Halide::Buffer<float> input, float sigma_color) {
    Halide::Var x, y;
    
    // Calculate kernel size
    int kernel_radius = static_cast<int>(std::ceil(3.0f * sigma_space_));
    int kernel_size = 2 * kernel_radius + 1;
    
    // Create range filtering function
    Halide::Func range_filtered;
    Halide::RDom r(-kernel_radius, kernel_size, -kernel_radius, kernel_size);
    
    // Apply range filtering with boundary conditions
    Halide::Expr clamped_x = Halide::clamp(x + r.x, 0, input.width() - 1);
    Halide::Expr clamped_y = Halide::clamp(y + r.y, 0, input.height() - 1);
    
    Halide::Expr center_value = input(x, y);
    Halide::Expr neighbor_value = input(clamped_x, clamped_y);
    Halide::Expr intensity_diff = center_value - neighbor_value;
    
    // Range weight based on intensity difference
    Halide::Expr range_weight = Halide::exp(-(intensity_diff * intensity_diff) / (2.0f * sigma_color * sigma_color));
    
    // Combine with spatial weight (simplified bilateral filter)
    Halide::Expr spatial_weight = Halide::exp(-(Halide::cast<float>(r.x * r.x + r.y * r.y)) / (2.0f * sigma_space_ * sigma_space_));
    Halide::Expr bilateral_weight = spatial_weight * range_weight;
    
    range_filtered(x, y) = Halide::sum(bilateral_weight * neighbor_value) / Halide::sum(bilateral_weight);
    
    // Schedule for optimal performance
    range_filtered.vectorize(x, 8).parallel(y);
    
    return range_filtered;
}

Halide::Buffer<float> NoiseReduction2DHybrid::apply_bilateral_filter_halide(const Halide::Buffer<float>& input) {
    int width = input.width();
    int height = input.height();
    
    // Debug output
    if (is_debug_) {
        std::cout << "2D Noise Reduction Halide - Parameters:" << std::endl;
        std::cout << "  Image size: " << width << "x" << height << std::endl;
        std::cout << "  Sigma space: " << sigma_space_ << std::endl;
        std::cout << "  Sigma color: " << sigma_color_ << std::endl;
        std::cout << "  Window size: " << window_size_ << std::endl;
    }
    
    // Create bilateral filter function
    Halide::Func bilateral_filter;
    Halide::Var x, y;
    
    // Calculate kernel size
    int kernel_radius = static_cast<int>(std::ceil(3.0f * sigma_space_));
    int kernel_size = 2 * kernel_radius + 1;
    
    // Create reduction domain for convolution
    Halide::RDom r(-kernel_radius, kernel_size, -kernel_radius, kernel_size);
    
    // Apply boundary conditions
    Halide::Expr clamped_x = Halide::clamp(x + r.x, 0, width - 1);
    Halide::Expr clamped_y = Halide::clamp(y + r.y, 0, height - 1);
    
    // Get center and neighbor pixel values
    Halide::Expr center_value = input(x, y);
    Halide::Expr neighbor_value = input(clamped_x, clamped_y);
    
    // Calculate spatial weight (Gaussian)
    Halide::Expr spatial_weight = Halide::exp(-(Halide::cast<float>(r.x * r.x + r.y * r.y)) / (2.0f * sigma_space_ * sigma_space_));
    
    // Calculate range weight (intensity difference)
    Halide::Expr intensity_diff = center_value - neighbor_value;
    Halide::Expr range_weight = Halide::exp(-(intensity_diff * intensity_diff) / (2.0f * sigma_color_ * sigma_color_));
    
    // Combine weights
    Halide::Expr bilateral_weight = spatial_weight * range_weight;
    
    // Apply bilateral filtering
    bilateral_filter(x, y) = Halide::sum(bilateral_weight * neighbor_value) / Halide::sum(bilateral_weight);
    
    // Schedule for optimal performance
    bilateral_filter.vectorize(x, 8).parallel(y);
    
    // Realize the result
    Halide::Buffer<float> output = bilateral_filter.realize({width, height});
    
    if (is_debug_) {
        // Print output statistics
        float min_val = output.min();
        float max_val = output.max();
        std::cout << "2D Noise Reduction Halide - Output - Min: " << min_val << ", Max: " << max_val << std::endl;
    }
    
    return output;
}

Halide::Buffer<float> NoiseReduction2DHybrid::opencv_to_halide(const cv::Mat& img) {
    // Convert OpenCV Mat to Halide Buffer
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F);
    
    // For single channel, create 2D buffer
    if (float_img.channels() == 1) {
        return Halide::Buffer<float>(reinterpret_cast<float*>(float_img.data), 
                                   float_img.cols, float_img.rows);
    } else {
        // For multi-channel, process each channel separately
        // For now, convert to grayscale
        cv::Mat gray_img;
        cv::cvtColor(float_img, gray_img, cv::COLOR_BGR2GRAY);
        return Halide::Buffer<float>(reinterpret_cast<float*>(gray_img.data), 
                                   gray_img.cols, gray_img.rows);
    }
}

cv::Mat NoiseReduction2DHybrid::halide_to_opencv(const Halide::Buffer<float>& buffer) {
    // Convert Halide Buffer back to OpenCV Mat
    cv::Mat result(buffer.height(), buffer.width(), CV_32F);
    
    // Copy data
    for (int y = 0; y < buffer.height(); ++y) {
        for (int x = 0; x < buffer.width(); ++x) {
            result.at<float>(y, x) = buffer(x, y);
        }
    }
    
    return result;
}

void NoiseReduction2DHybrid::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_2d_noise_reduction_hybrid_" + 
                                 std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + ".png";
        cv::imwrite(output_path, img_);
    }
}

cv::Mat NoiseReduction2DHybrid::execute() {
    if (!is_enable_) {
        return img_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Convert OpenCV image to Halide buffer
    Halide::Buffer<float> halide_input = opencv_to_halide(img_);
    
    // Apply bilateral filtering using Halide
    Halide::Buffer<float> halide_output = apply_bilateral_filter_halide(halide_input);
    
    // Convert back to OpenCV
    cv::Mat filtered_float = halide_to_opencv(halide_output);
    
    // Apply bit depth scaling
    cv::Mat result;
    if (output_bit_depth_ == 8) {
        filtered_float.convertTo(result, CV_8U, 255.0);
    }
    else if (output_bit_depth_ == 16) {
        filtered_float.convertTo(result, CV_16U, 65535.0);
    }
    else if (output_bit_depth_ == 32) {
        filtered_float.convertTo(result, CV_32F);
    }
    else {
        throw std::runtime_error("Unsupported output bit depth. Use 8, 16, or 32.");
    }
    
    img_ = result;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "  2D Noise Reduction Halide execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }

    // Save intermediate results if enabled
    if (is_save_) {
        save();
    }

    return img_;
}

hdr_isp::EigenImage3C NoiseReduction2DHybrid::execute_eigen() {
    if (!is_enable_) {
        if (has_eigen_input_) {
            return eigen_img_;
        } else {
            return hdr_isp::EigenImage3C::fromOpenCV(img_);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Get the appropriate input
    hdr_isp::EigenImage3C eigen_img;
    if (has_eigen_input_) {
        eigen_img = eigen_img_;
    } else {
        eigen_img = hdr_isp::EigenImage3C::fromOpenCV(img_);
    }
    
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    
    // Process each channel independently using Halide
    hdr_isp::EigenImage3C result(rows, cols);
    
    // Convert each channel to Halide and process
    for (int c = 0; c < 3; ++c) {
        // Create Halide buffer for this channel
        Halide::Buffer<float> channel_buffer(cols, rows);
        
        // Copy channel data
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (c == 0) channel_buffer(x, y) = eigen_img.r().data()(y, x);
                else if (c == 1) channel_buffer(x, y) = eigen_img.g().data()(y, x);
                else channel_buffer(x, y) = eigen_img.b().data()(y, x);
            }
        }
        
        // Apply bilateral filtering
        Halide::Buffer<float> filtered_channel = apply_bilateral_filter_halide(channel_buffer);
        
        // Copy back to Eigen
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (c == 0) result.r().data()(y, x) = filtered_channel(x, y);
                else if (c == 1) result.g().data()(y, x) = filtered_channel(x, y);
                else result.b().data()(y, x) = filtered_channel(x, y);
            }
        }
    }
    
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
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "  2D Noise Reduction Halide Eigen execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }
    
    return result;
} 
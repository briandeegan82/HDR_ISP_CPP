#include "scale_hybrid.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

ScaleHybrid::ScaleHybrid(cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                         const YAML::Node& parm_sca, int conv_std)
    : Scale(img, platform, sensor_info, parm_sca, conv_std)
{
    algorithm_ = get_algorithm();
    is_hardware_ = parm_sca_["is_hardware"].as<bool>();
}

ScaleHybrid::ScaleHybrid(const hdr_isp::EigenImage3C& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                         const YAML::Node& parm_sca, int conv_std)
    : Scale(img, platform, sensor_info, parm_sca, conv_std)
{
    algorithm_ = get_algorithm();
    is_hardware_ = parm_sca_["is_hardware"].as<bool>();
}

std::string ScaleHybrid::get_algorithm() {
    if (parm_sca_["algorithm"].IsDefined()) {
        return parm_sca_["algorithm"].as<std::string>();
    }
    return "Nearest_Neighbor"; // Default algorithm
}

Halide::Buffer<float> ScaleHybrid::apply_nearest_neighbor_halide(const Halide::Buffer<float>& input, int new_width, int new_height) {
    int old_width = input.width();
    int old_height = input.height();
    
    if (is_debug_) {
        std::cout << "Scale Hybrid - Nearest Neighbor Halide" << std::endl;
        std::cout << "  Input size: " << old_width << "x" << old_height << std::endl;
        std::cout << "  Output size: " << new_width << "x" << new_height << std::endl;
    }
    
    Halide::Var x, y;
    Halide::Func nearest_neighbor;
    
    // Calculate scale factors
    float scale_x = static_cast<float>(old_width) / new_width;
    float scale_y = static_cast<float>(old_height) / new_height;
    
    // Source coordinates
    Halide::Expr src_x = Halide::cast<int>(x * scale_x);
    Halide::Expr src_y = Halide::cast<int>(y * scale_y);
    
    // Clamp to valid range
    src_x = Halide::clamp(src_x, 0, old_width - 1);
    src_y = Halide::clamp(src_y, 0, old_height - 1);
    
    // Apply nearest neighbor interpolation
    nearest_neighbor(x, y) = input(src_x, src_y);
    
    // Schedule for optimal performance
    nearest_neighbor.vectorize(x, 8).parallel(y);
    
    // Realize the result
    Halide::Buffer<float> output = nearest_neighbor.realize({new_width, new_height});
    
    if (is_debug_) {
        float min_val = output.min();
        float max_val = output.max();
        std::cout << "  Output - Min: " << min_val << ", Max: " << max_val << std::endl;
    }
    
    return output;
}

Halide::Buffer<float> ScaleHybrid::apply_bilinear_halide(const Halide::Buffer<float>& input, int new_width, int new_height) {
    int old_width = input.width();
    int old_height = input.height();
    
    if (is_debug_) {
        std::cout << "Scale Hybrid - Bilinear Halide" << std::endl;
        std::cout << "  Input size: " << old_width << "x" << old_height << std::endl;
        std::cout << "  Output size: " << new_width << "x" << new_height << std::endl;
    }
    
    Halide::Var x, y;
    Halide::Func bilinear;
    
    // Calculate scale factors
    float scale_x = static_cast<float>(old_width) / new_width;
    float scale_y = static_cast<float>(old_height) / new_height;
    
    // Source coordinates (floating point)
    Halide::Expr src_x_f = x * scale_x;
    Halide::Expr src_y_f = y * scale_y;
    
    // Integer coordinates for the four neighboring pixels
    Halide::Expr x0 = Halide::cast<int>(src_x_f);
    Halide::Expr y0 = Halide::cast<int>(src_y_f);
    Halide::Expr x1 = Halide::min(x0 + 1, old_width - 1);
    Halide::Expr y1 = Halide::min(y0 + 1, old_height - 1);
    
    // Clamp coordinates
    x0 = Halide::clamp(x0, 0, old_width - 1);
    y0 = Halide::clamp(y0, 0, old_height - 1);
    
    // Interpolation weights
    Halide::Expr wx = src_x_f - Halide::cast<float>(x0);
    Halide::Expr wy = src_y_f - Halide::cast<float>(y0);
    
    // Get the four neighboring pixel values
    Halide::Expr p00 = input(x0, y0);
    Halide::Expr p01 = input(x0, y1);
    Halide::Expr p10 = input(x1, y0);
    Halide::Expr p11 = input(x1, y1);
    
    // Bilinear interpolation
    Halide::Expr result = p00 * (1.0f - wx) * (1.0f - wy) +
                         p01 * (1.0f - wx) * wy +
                         p10 * wx * (1.0f - wy) +
                         p11 * wx * wy;
    
    bilinear(x, y) = result;
    
    // Schedule for optimal performance
    bilinear.vectorize(x, 8).parallel(y);
    
    // Realize the result
    Halide::Buffer<float> output = bilinear.realize({new_width, new_height});
    
    if (is_debug_) {
        float min_val = output.min();
        float max_val = output.max();
        std::cout << "  Output - Min: " << min_val << ", Max: " << max_val << std::endl;
    }
    
    return output;
}

Halide::Buffer<float> ScaleHybrid::apply_bicubic_halide(const Halide::Buffer<float>& input, int new_width, int new_height) {
    int old_width = input.width();
    int old_height = input.height();
    
    if (is_debug_) {
        std::cout << "Scale Hybrid - Bicubic Halide" << std::endl;
        std::cout << "  Input size: " << old_width << "x" << old_height << std::endl;
        std::cout << "  Output size: " << new_width << "x" << new_height << std::endl;
    }
    
    Halide::Var x, y;
    Halide::Func bicubic;
    
    // Calculate scale factors
    float scale_x = static_cast<float>(old_width) / new_width;
    float scale_y = static_cast<float>(old_height) / new_height;
    
    // Source coordinates (floating point)
    Halide::Expr src_x_f = x * scale_x;
    Halide::Expr src_y_f = y * scale_y;
    
    // Integer coordinates for the center pixel
    Halide::Expr x0 = Halide::cast<int>(src_x_f);
    Halide::Expr y0 = Halide::cast<int>(src_y_f);
    
    // Clamp coordinates
    x0 = Halide::clamp(x0, 0, old_width - 1);
    y0 = Halide::clamp(y0, 0, old_height - 1);
    
    // For simplicity, implement a simplified bicubic using 4x4 neighborhood
    // This is a simplified version - full bicubic would require more complex kernel computation
    
    Halide::RDom r(-1, 4, -1, 4); // 4x4 neighborhood
    
    // Get neighboring pixel coordinates
    Halide::Expr nx = x0 + r.x;
    Halide::Expr ny = y0 + r.y;
    
    // Clamp to image boundaries
    nx = Halide::clamp(nx, 0, old_width - 1);
    ny = Halide::clamp(ny, 0, old_height - 1);
    
    // Simplified bicubic weight (using distance-based weighting)
    Halide::Expr dx = Halide::abs(src_x_f - Halide::cast<float>(nx));
    Halide::Expr dy = Halide::abs(src_y_f - Halide::cast<float>(ny));
    
    // Simplified bicubic kernel (using distance-based weighting)
    Halide::Expr weight = Halide::select(dx < 1.0f && dy < 1.0f, 
                                        (1.0f - dx) * (1.0f - dy), 0.0f);
    
    // Apply weighted sum
    bicubic(x, y) = Halide::sum(weight * input(nx, ny)) / Halide::sum(weight);
    
    // Schedule for optimal performance
    bicubic.vectorize(x, 8).parallel(y);
    
    // Realize the result
    Halide::Buffer<float> output = bicubic.realize({new_width, new_height});
    
    if (is_debug_) {
        float min_val = output.min();
        float max_val = output.max();
        std::cout << "  Output - Min: " << min_val << ", Max: " << max_val << std::endl;
    }
    
    return output;
}

Halide::Buffer<float> ScaleHybrid::opencv_to_halide(const cv::Mat& img) {
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

cv::Mat ScaleHybrid::halide_to_opencv(const Halide::Buffer<float>& buffer) {
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

Halide::Buffer<float> ScaleHybrid::eigen_to_halide(const hdr_isp::EigenImage& eigen_img) {
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    
    // Create Halide buffer
    Halide::Buffer<float> buffer(cols, rows);
    
    // Copy data
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            buffer(x, y) = eigen_img(y, x);
        }
    }
    
    return buffer;
}

hdr_isp::EigenImage ScaleHybrid::halide_to_eigen(const Halide::Buffer<float>& buffer, int rows, int cols) {
    // Create Eigen image
    hdr_isp::EigenImage result = hdr_isp::EigenImage::Zero(rows, cols);
    
    // Copy data
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            result(y, x) = buffer(x, y);
        }
    }
    
    return result;
}

void ScaleHybrid::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_scale_hybrid_" + 
                                 std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + ".png";
        cv::imwrite(output_path, img_);
    }
}

cv::Mat ScaleHybrid::execute() {
    if (!enable_) {
        return img_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Check if no change in size
    if (old_size_ == new_size_) {
        if (is_debug_) {
            std::cout << "Scale Hybrid - Output size is the same as input size." << std::endl;
        }
        return img_;
    }
    
    if (is_debug_) {
        std::cout << "Scale Hybrid - Using algorithm: " << algorithm_ << std::endl;
    }
    
    // Convert OpenCV image to Halide buffer
    Halide::Buffer<float> halide_input = opencv_to_halide(img_);
    
    // Apply scaling using Halide
    Halide::Buffer<float> halide_output;
    if (algorithm_ == "Nearest_Neighbor") {
        halide_output = apply_nearest_neighbor_halide(halide_input, new_size_.second, new_size_.first);
    } else if (algorithm_ == "Bilinear") {
        halide_output = apply_bilinear_halide(halide_input, new_size_.second, new_size_.first);
    } else if (algorithm_ == "Bicubic") {
        halide_output = apply_bicubic_halide(halide_input, new_size_.second, new_size_.first);
    } else {
        // Default to nearest neighbor
        halide_output = apply_nearest_neighbor_halide(halide_input, new_size_.second, new_size_.first);
    }
    
    // Convert back to OpenCV
    cv::Mat result = halide_to_opencv(halide_output);
    
    // Convert back to original type if needed
    if (img_.type() != CV_32F) {
        cv::Mat converted_result;
        result.convertTo(converted_result, img_.type());
        img_ = converted_result;
    } else {
        img_ = result;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "Scale Hybrid execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }

    // Save intermediate results if enabled
    if (is_save_) {
        save();
    }

    return img_;
}

hdr_isp::EigenImage3C ScaleHybrid::execute_eigen() {
    if (!enable_) {
        if (has_eigen_input_) {
            return eigen_img_;
        } else {
            return hdr_isp::EigenImage3C::fromOpenCV(img_);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Check if no change in size
    if (old_size_ == new_size_) {
        if (is_debug_) {
            std::cout << "Scale Hybrid - Output size is the same as input size." << std::endl;
        }
        return eigen_img_;
    }
    
    if (is_debug_) {
        std::cout << "Scale Hybrid - Using algorithm: " << algorithm_ << std::endl;
    }
    
    // Get the appropriate input
    hdr_isp::EigenImage3C eigen_img;
    if (has_eigen_input_) {
        eigen_img = eigen_img_;
    } else {
        eigen_img = hdr_isp::EigenImage3C::fromOpenCV(img_);
    }
    
    int old_rows = eigen_img.rows();
    int old_cols = eigen_img.cols();
    int new_rows = new_size_.first;
    int new_cols = new_size_.second;
    
    // Process each channel independently using Halide
    hdr_isp::EigenImage3C result(new_rows, new_cols);
    
    // Convert each channel to Halide and process
    for (int c = 0; c < 3; ++c) {
        // Create Halide buffer for this channel
        Halide::Buffer<float> channel_buffer(old_cols, old_rows);
        
        // Copy channel data
        for (int y = 0; y < old_rows; ++y) {
            for (int x = 0; x < old_cols; ++x) {
                if (c == 0) channel_buffer(x, y) = eigen_img.r().data()(y, x);
                else if (c == 1) channel_buffer(x, y) = eigen_img.g().data()(y, x);
                else channel_buffer(x, y) = eigen_img.b().data()(y, x);
            }
        }
        
        // Apply scaling
        Halide::Buffer<float> scaled_channel;
        if (algorithm_ == "Nearest_Neighbor") {
            scaled_channel = apply_nearest_neighbor_halide(channel_buffer, new_cols, new_rows);
        } else if (algorithm_ == "Bilinear") {
            scaled_channel = apply_bilinear_halide(channel_buffer, new_cols, new_rows);
        } else if (algorithm_ == "Bicubic") {
            scaled_channel = apply_bicubic_halide(channel_buffer, new_cols, new_rows);
        } else {
            // Default to nearest neighbor
            scaled_channel = apply_nearest_neighbor_halide(channel_buffer, new_cols, new_rows);
        }
        
        // Copy back to Eigen
        for (int y = 0; y < new_rows; ++y) {
            for (int x = 0; x < new_cols; ++x) {
                if (c == 0) result.r().data()(y, x) = scaled_channel(x, y);
                else if (c == 1) result.g().data()(y, x) = scaled_channel(x, y);
                else result.b().data()(y, x) = scaled_channel(x, y);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "Scale Hybrid Eigen execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }
    
    return result;
} 
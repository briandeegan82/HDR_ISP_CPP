#include "rgb_conversion.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

RGBConversion::RGBConversion(cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                           const YAML::Node& parm_rgb, const YAML::Node& parm_csc)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_rgb_(parm_rgb)
    , parm_csc_(parm_csc)
    , enable_(parm_rgb["is_enable"].as<bool>())
    , is_save_(parm_rgb["is_save"].as<bool>())
    , bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , conv_std_(parm_csc["conv_standard"].as<int>())
    , yuv_img_(img)
{
    // Pre-compute conversion matrices
    if (conv_std_ == 1) {
        // BT.709
        yuv2rgb_mat_ = (cv::Mat_<int>(3, 3) << 74, 0, 114,
                                             74, -13, -34,
                                             74, 135, 0);
    } else {
        // BT.601/407
        yuv2rgb_mat_ = (cv::Mat_<int>(3, 3) << 64, 87, 0,
                                             64, -44, -20,
                                             61, 0, 105);
    }

    // Pre-compute offset array
    offset_ = cv::Vec3i(16, 128, 128);
}

cv::Mat RGBConversion::yuv_to_rgb() {
    auto start_total = std::chrono::high_resolution_clock::now();

    // Reshape and subtract offset
    auto start_reshape = std::chrono::high_resolution_clock::now();
    cv::Mat mat_2d = yuv_img_.reshape(1, yuv_img_.total());
    cv::Mat mat_2d_float;
    mat_2d.convertTo(mat_2d_float, CV_32F);
    mat_2d_float -= cv::Scalar(offset_[0], offset_[1], offset_[2]);
    cv::Mat mat2d_t = mat_2d_float.t();
    auto end_reshape = std::chrono::high_resolution_clock::now();

    // Matrix multiplication
    auto start_mult = std::chrono::high_resolution_clock::now();
    cv::Mat yuv2rgb_float;
    yuv2rgb_mat_.convertTo(yuv2rgb_float, CV_32F);
    cv::Mat rgb_2d = yuv2rgb_float * mat2d_t;
    rgb_2d = rgb_2d / 64.0; // Equivalent to right shift by 6
    auto end_mult = std::chrono::high_resolution_clock::now();

    // Final conversion
    auto start_final = std::chrono::high_resolution_clock::now();
    cv::Mat rgb_2d_t = rgb_2d.t();
    cv::Mat rgb_reshaped = rgb_2d_t.reshape(3, yuv_img_.rows);
    cv::Mat rgb_clipped;
    cv::threshold(rgb_reshaped, rgb_clipped, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(rgb_clipped, rgb_clipped, 0, 0, cv::THRESH_TOZERO);
    rgb_clipped.convertTo(yuv_img_, CV_8UC3);
    auto end_final = std::chrono::high_resolution_clock::now();

    // Print timing information
    std::chrono::duration<double, std::milli> reshape_time = end_reshape - start_reshape;
    std::chrono::duration<double, std::milli> mult_time = end_mult - start_mult;
    std::chrono::duration<double, std::milli> final_time = end_final - start_final;
    std::chrono::duration<double, std::milli> total_time = end_final - start_total;

    std::cout << "  Matrix reshaping and offset time: " << reshape_time.count() << "ms" << std::endl;
    std::cout << "  Matrix multiplication time: " << mult_time.count() << "ms" << std::endl;
    std::cout << "  Final conversion time: " << final_time.count() << "ms" << std::endl;
    std::cout << "  Total RGB conversion time: " << total_time.count() << "ms" << std::endl;

    return yuv_img_;
}

void RGBConversion::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_rgb_conversion_" + 
            std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + "_" +
            std::to_string(bit_depth_) + "bits_" +
            sensor_info_["bayer_pattern"].as<std::string>() + ".png";
        cv::imwrite(output_path, img_);
    }
}

cv::Mat RGBConversion::execute() {
    if (enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        img_ = yuv_to_rgb();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "  Total execution time: " << duration.count() << "s" << std::endl;
    }
    save();
    return img_;
} 
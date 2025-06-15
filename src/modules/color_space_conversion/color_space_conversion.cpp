#include "color_space_conversion.hpp"
#include <chrono>
#include <iostream>
#include <cmath>  // Add this for std::round

ColorSpaceConversion::ColorSpaceConversion(const cv::Mat& img, const YAML::Node& sensor_info, 
                                         const YAML::Node& parm_csc, const YAML::Node& parm_cse)
    : raw_(img.clone())
    , sensor_info_(sensor_info)
    , parm_csc_(parm_csc)
    , parm_cse_(parm_cse)
    , bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , conv_std_(parm_csc["conv_standard"].as<int>())
    , is_save_(parm_csc["is_save"].as<bool>())
{
}

cv::Mat ColorSpaceConversion::execute() {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result = rgb_to_yuv_8bit();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Color Space Conversion execution time: " << duration.count() << " seconds" << std::endl;

    return result;
}

cv::Mat ColorSpaceConversion::rgb_to_yuv_8bit() {
    auto total_start = std::chrono::high_resolution_clock::now();

    // Set up conversion matrix based on standard
    if (conv_std_ == 1) {
        // BT.709
        float mat_data[] = {
            47, 157, 16,
            -26, -86, 112,
            112, -102, -10
        };
        rgb2yuv_mat_ = cv::Mat(3, 3, CV_32F, mat_data);
    } else {
        // BT.601/407
        float mat_data[] = {
            77, 150, 29,
            131, -110, -21,
            -44, -87, 138
        };
        rgb2yuv_mat_ = cv::Mat(3, 3, CV_32F, mat_data);
    }

    // Reshape image to Nx3 matrix
    cv::Mat mat2d = raw_.reshape(1, raw_.total());
    cv::Mat mat2d_t;
    cv::transpose(mat2d, mat2d_t);

    // Convert to YUV
    cv::Mat yuv_2d;
    cv::gemm(rgb2yuv_mat_, mat2d_t, 1.0, cv::Mat(), 0.0, yuv_2d);

    // Convert to float and apply bit depth conversion
    yuv_2d.convertTo(yuv_2d, CV_64F);
    yuv_2d /= (1 << 8);
    yuv_2d.forEach<double>([](double& pixel, const int* position) {
        pixel = std::round(pixel);
    });

    // Apply color saturation enhancement if enabled
    if (parm_cse_["is_enable"].as<bool>()) {
        double gain = parm_cse_["saturation_gain"].as<double>();
        cv::Mat uv_channels = yuv_2d.rowRange(1, 3);
        uv_channels *= gain;
    }

    // Add DC offset
    yuv_2d.row(0) += (1 << (bit_depth_ / 2));
    yuv_2d.rowRange(1, 3) += (1 << (bit_depth_ - 1));

    // Transpose back and reshape
    cv::Mat yuv2d_t;
    cv::transpose(yuv_2d, yuv2d_t);

    // Clip values
    cv::threshold(yuv2d_t, yuv2d_t, 0, (1 << bit_depth_) - 1, cv::THRESH_TRUNC);

    // Final normalization to 8-bit
    yuv2d_t /= (1 << (bit_depth_ - 8));
    yuv2d_t.forEach<double>([](double& pixel, const int* position) {
        pixel = std::round(pixel);
    });
    cv::threshold(yuv2d_t, yuv2d_t, 0, 255, cv::THRESH_TRUNC);

    // Reshape back to original dimensions
    cv::Mat result = yuv2d_t.reshape(3, raw_.rows);
    result.convertTo(result, CV_8U);

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;
    std::cout << "Total RGB to YUV conversion time: " << total_duration.count() << " seconds" << std::endl;

    return result;
} 
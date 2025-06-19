#include "color_space_conversion.hpp"
#include <chrono>
#include <iostream>
#include <cmath>  // Add this for std::round

// Helper function to print image statistics
void printImageStats(const cv::Mat& img, const std::string& stage) {
    double minVal, maxVal;
    cv::minMaxLoc(img, &minVal, &maxVal);
    cv::Scalar meanVal = cv::mean(img);
    std::cout << stage << " - Mean: " << meanVal << ", Min: " << minVal << ", Max: " << maxVal << std::endl;
}

ColorSpaceConversion::ColorSpaceConversion(const cv::Mat& img, const YAML::Node& sensor_info, 
                                         const YAML::Node& parm_csc, const YAML::Node& parm_cse)
    : raw_(img.clone())
    , sensor_info_(sensor_info)
    , parm_csc_(parm_csc)
    , parm_cse_(parm_cse)
    , bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , conv_std_(parm_csc["conv_standard"].as<int>())
    , is_save_(parm_csc["is_save"].as<bool>())
    , use_eigen_(true) // Use Eigen by default
{
    printImageStats(raw_, "Input raw image");
}

cv::Mat ColorSpaceConversion::execute() {
    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat result;
    if (use_eigen_) {
        hdr_isp::EigenImage eigen_result = rgb_to_yuv_8bit_eigen();
        result = hdr_isp::eigen_to_opencv(eigen_result);
    } else {
        result = rgb_to_yuv_8bit();
    }
    
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
        std::cout << "BT.709 Matrix:" << std::endl;
        rgb2yuv_mat_ = cv::Mat(3, 3, CV_32F, mat_data);
    } else {
        // BT.601/407
        float mat_data[] = {
            77, 150, 29,
            131, -110, -21,
            -44, -87, 138
        };
        std::cout << "BT.601/407 Matrix:" << std::endl;
        rgb2yuv_mat_ = cv::Mat(3, 3, CV_32F, mat_data);
    }

    // Reshape image to Nx3 matrix
    cv::Mat mat2d = raw_.reshape(1, raw_.total());
    cv::Mat mat2d_t;
    cv::transpose(mat2d, mat2d_t);
    printImageStats(mat2d_t, "After reshape and transpose");

    // Convert to float to match rgb2yuv_mat_ type
    cv::Mat mat2d_t_float;
    mat2d_t.convertTo(mat2d_t_float, CV_32F);
    printImageStats(mat2d_t_float, "After convert to float");

    // Convert to YUV
    cv::Mat yuv_2d;
    cv::gemm(rgb2yuv_mat_, mat2d_t_float, 1.0, cv::Mat(), 0.0, yuv_2d);
    printImageStats(yuv_2d, "After matrix multiplication (YUV)");

    // Convert to float and apply bit depth conversion
    yuv_2d.convertTo(yuv_2d, CV_64F);
    yuv_2d /= (1 << 8);
    printImageStats(yuv_2d, "After division by 256");
    
    yuv_2d.forEach<double>([](double& pixel, const int* position) {
        pixel = std::round(pixel);
    });
    printImageStats(yuv_2d, "After rounding");

    // Apply color saturation enhancement if enabled
    if (parm_cse_["is_enable"].as<bool>()) {
        double gain = parm_cse_["saturation_gain"].as<double>();
        cv::Mat uv_channels = yuv_2d.rowRange(1, 3);
        uv_channels *= gain;
        printImageStats(yuv_2d, "After saturation enhancement");
    }

    // Add DC offset
    yuv_2d.row(0) += (1 << (bit_depth_ / 2));
    yuv_2d.rowRange(1, 3) += (1 << (bit_depth_ - 1));
    printImageStats(yuv_2d, "After adding DC offset");

    // Transpose back and reshape
    cv::Mat yuv2d_t;
    cv::transpose(yuv_2d, yuv2d_t);
    printImageStats(yuv2d_t, "After transpose back");

    // Clip values
    cv::threshold(yuv2d_t, yuv2d_t, 0, (1 << bit_depth_) - 1, cv::THRESH_TRUNC);
    printImageStats(yuv2d_t, "After thresholding");

    // Final normalization to 8-bit
    yuv2d_t /= (1 << (bit_depth_ - 8));
    yuv2d_t.forEach<double>([](double& pixel, const int* position) {
        pixel = std::round(pixel);
    });
    cv::threshold(yuv2d_t, yuv2d_t, 0, 255, cv::THRESH_TRUNC);
    printImageStats(yuv2d_t, "After final normalization to 8-bit");

    // Reshape back to original dimensions
    cv::Mat result = yuv2d_t.reshape(3, raw_.rows);
    result.convertTo(result, CV_8U);
    printImageStats(result, "Final result");

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;
    std::cout << "Total RGB to YUV conversion time: " << total_duration.count() << " seconds" << std::endl;

    return result;
}

hdr_isp::EigenImage ColorSpaceConversion::rgb_to_yuv_8bit_eigen() {
    // Convert to Eigen 3-channel image
    hdr_isp::EigenImage3C eigen_raw = hdr_isp::EigenImage3C::fromOpenCV(raw_);
    
    // Set up conversion matrix based on standard
    Eigen::Matrix3f rgb2yuv_eigen;
    if (conv_std_ == 1) {
        // BT.709
        rgb2yuv_eigen << 47, 157, 16,
                        -26, -86, 112,
                        112, -102, -10;
        std::cout << "BT.709 Matrix (Eigen):" << std::endl;
    } else {
        // BT.601/407
        rgb2yuv_eigen << 77, 150, 29,
                        131, -110, -21,
                        -44, -87, 138;
        std::cout << "BT.601/407 Matrix (Eigen):" << std::endl;
    }

    // Apply color space transformation
    hdr_isp::EigenImage3C yuv_3c = eigen_raw * rgb2yuv_eigen;
    
    // Convert to float and apply bit depth conversion
    yuv_3c = yuv_3c * (1.0f / (1 << 8));
    
    // Round values
    yuv_3c.r() = hdr_isp::EigenImage(yuv_3c.r().data().array().round().matrix());
    yuv_3c.g() = hdr_isp::EigenImage(yuv_3c.g().data().array().round().matrix());
    yuv_3c.b() = hdr_isp::EigenImage(yuv_3c.b().data().array().round().matrix());
    
    // Apply color saturation enhancement if enabled
    if (parm_cse_["is_enable"].as<bool>()) {
        double gain = parm_cse_["saturation_gain"].as<double>();
        yuv_3c.g() = yuv_3c.g() * gain; // U channel
        yuv_3c.b() = yuv_3c.b() * gain; // V channel
    }

    // Add DC offset
    yuv_3c.r() = yuv_3c.r() + (1 << (bit_depth_ / 2));
    yuv_3c.g() = yuv_3c.g() + (1 << (bit_depth_ - 1));
    yuv_3c.b() = yuv_3c.b() + (1 << (bit_depth_ - 1));
    
    // Clip values
    float max_val = static_cast<float>((1 << bit_depth_) - 1);
    yuv_3c = yuv_3c.clip(0.0f, max_val);
    
    // Final normalization to 8-bit
    yuv_3c = yuv_3c * (1.0f / (1 << (bit_depth_ - 8)));
    yuv_3c.r() = hdr_isp::EigenImage(yuv_3c.r().data().array().round().matrix());
    yuv_3c.g() = hdr_isp::EigenImage(yuv_3c.g().data().array().round().matrix());
    yuv_3c.b() = hdr_isp::EigenImage(yuv_3c.b().data().array().round().matrix());
    yuv_3c = yuv_3c.clip(0.0f, 255.0f);
    
    // Convert back to OpenCV and then to single channel for return
    cv::Mat yuv_cv = yuv_3c.toOpenCV(CV_8UC3);
    cv::Mat yuv_single;
    cv::cvtColor(yuv_cv, yuv_single, cv::COLOR_BGR2GRAY);
    
    return hdr_isp::opencv_to_eigen(yuv_single);
} 
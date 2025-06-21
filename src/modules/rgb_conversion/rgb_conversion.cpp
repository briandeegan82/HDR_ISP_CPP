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
    , is_debug_(parm_rgb["is_debug"].as<bool>())
    , bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , conv_std_(parm_csc["conv_standard"].as<int>())
    , yuv_img_(img)
    , use_eigen_(true) // Use Eigen by default
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

cv::Mat RGBConversion::yuv_to_rgb_opencv() {
    auto start_total = std::chrono::high_resolution_clock::now();

    // Check if image is single-channel or multi-channel
    if (yuv_img_.channels() == 1) {
        // Original single-channel YUV processing
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
    } else if (yuv_img_.channels() == 3) {
        // Multi-channel processing - assume it's already RGB
        // For now, just pass through the image without modification
        // In a full implementation, you'd detect if it's YUV and convert accordingly
        auto end_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_time = end_total - start_total;
        
        std::cout << "  Multi-channel image detected - passing through without conversion" << std::endl;
        std::cout << "  Total RGB conversion time: " << total_time.count() << "ms" << std::endl;
        
        return yuv_img_.clone();
    } else {
        throw std::runtime_error("Unsupported number of channels. Use 1 or 3 channels.");
    }
}

hdr_isp::EigenImage RGBConversion::yuv_to_rgb_eigen() {
    auto start_total = std::chrono::high_resolution_clock::now();

    // Check if image is single-channel or multi-channel
    if (yuv_img_.channels() == 1) {
        // Single-channel processing (original YUV format)
        hdr_isp::EigenImage eigen_yuv = hdr_isp::opencv_to_eigen(yuv_img_);
        int rows = eigen_yuv.rows();
        int cols = eigen_yuv.cols();

        // Reshape and subtract offset
        auto start_reshape = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXf yuv_2d = eigen_yuv.data().reshaped(rows * cols, 1);
        
        // Create offset matrix
        Eigen::VectorXf offset_vec(rows * cols);
        for (int i = 0; i < rows * cols; ++i) {
            int channel = i % 3;
            offset_vec(i) = static_cast<float>(offset_[channel]);
        }
        
        yuv_2d = yuv_2d - offset_vec;
        auto end_reshape = std::chrono::high_resolution_clock::now();

        // Matrix multiplication
        auto start_mult = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXf yuv2rgb_eigen = Eigen::Map<Eigen::MatrixXf>(
            reinterpret_cast<float*>(yuv2rgb_mat_.data), 3, 3);
        
        // Reshape yuv_2d to 3xN for matrix multiplication
        Eigen::MatrixXf yuv_3xn = yuv_2d.reshaped(3, rows * cols / 3);
        Eigen::MatrixXf rgb_3xn = yuv2rgb_eigen * yuv_3xn;
        rgb_3xn = rgb_3xn / 64.0f; // Equivalent to right shift by 6
        auto end_mult = std::chrono::high_resolution_clock::now();

        // Final conversion
        auto start_final = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXf rgb_reshaped = rgb_3xn.reshaped(rows, cols);
        
        // Clip values
        rgb_reshaped = rgb_reshaped.cwiseMax(0.0f).cwiseMin(255.0f);
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

        return hdr_isp::EigenImage(rgb_reshaped);
    } else if (yuv_img_.channels() == 3) {
        // Multi-channel processing - assume it's already RGB
        // Convert to EigenImage3C for 3-channel processing
        hdr_isp::EigenImage3C eigen_img = hdr_isp::EigenImage3C::fromOpenCV(yuv_img_);
        
        // For now, just return the red channel as a single-channel representation
        // In a full implementation, you'd detect if it's YUV and convert accordingly
        auto end_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_time = end_total - start_total;
        
        std::cout << "  Multi-channel image detected - returning red channel" << std::endl;
        std::cout << "  Total RGB conversion time: " << total_time.count() << "ms" << std::endl;
        
        return eigen_img.r(); // Return red channel as single-channel using public method
    } else {
        throw std::runtime_error("Unsupported number of channels. Use 1 or 3 channels.");
    }
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
        
        if (use_eigen_) {
            hdr_isp::EigenImage result = yuv_to_rgb_eigen();
            img_ = hdr_isp::eigen_to_opencv(result);
        } else {
            img_ = yuv_to_rgb_opencv();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    return img_;
} 
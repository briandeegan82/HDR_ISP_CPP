#include "auto_white_balance.hpp"
#include "gray_world.hpp"
#include "norm_gray_world.hpp"
#include "pca.hpp"
#include <chrono>
#include <iostream>

AutoWhiteBalance::AutoWhiteBalance(const cv::Mat& raw, const YAML::Node& sensor_info, const YAML::Node& parm_awb)
    : raw_(raw)
    , sensor_info_(sensor_info)
    , parm_awb_(parm_awb)
    , enable_(parm_awb["is_enable"].as<bool>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , is_debug_(parm_awb["is_debug"].as<bool>())
    , underexposed_percentage_(parm_awb["underexposed_percentage"].as<float>())
    , overexposed_percentage_(parm_awb["overexposed_percentage"].as<float>())
    , bayer_(sensor_info["bayer_pattern"].as<std::string>())
    , algorithm_(parm_awb["algorithm"].as<std::string>())
    , use_eigen_(true) // Use Eigen by default
{
}

std::tuple<double, double> AutoWhiteBalance::determine_white_balance_gain()
{
    if (use_eigen_) {
        return determine_white_balance_gain_eigen();
    }
    
    // Original OpenCV implementation
    // Convert to float for calculations
    cv::Mat raw_float;
    raw_.convertTo(raw_float, CV_32F);

    // Extract Bayer pattern channels
    int height = raw_float.rows;
    int width = raw_float.cols;
    std::vector<float> r_values, g_values, b_values;

    if (bayer_ == "rggb") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                r_values.push_back(raw_float.at<float>(i, j));
                g_values.push_back(raw_float.at<float>(i, j + 1));
                g_values.push_back(raw_float.at<float>(i + 1, j));
                b_values.push_back(raw_float.at<float>(i + 1, j + 1));
            }
        }
    }
    else if (bayer_ == "bggr") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                b_values.push_back(raw_float.at<float>(i, j));
                g_values.push_back(raw_float.at<float>(i, j + 1));
                g_values.push_back(raw_float.at<float>(i + 1, j));
                r_values.push_back(raw_float.at<float>(i + 1, j + 1));
            }
        }
    }
    else if (bayer_ == "grbg") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                g_values.push_back(raw_float.at<float>(i, j));
                r_values.push_back(raw_float.at<float>(i, j + 1));
                b_values.push_back(raw_float.at<float>(i + 1, j));
                g_values.push_back(raw_float.at<float>(i + 1, j + 1));
            }
        }
    }
    else if (bayer_ == "gbrg") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                g_values.push_back(raw_float.at<float>(i, j));
                b_values.push_back(raw_float.at<float>(i, j + 1));
                r_values.push_back(raw_float.at<float>(i + 1, j));
                g_values.push_back(raw_float.at<float>(i + 1, j + 1));
            }
        }
    }

    // Create flattened image for algorithm input
    flatten_img_ = cv::Mat(3, r_values.size(), CV_32F);
    for (size_t i = 0; i < r_values.size(); i++) {
        flatten_img_.at<float>(0, i) = r_values[i];
        flatten_img_.at<float>(1, i) = g_values[i];
        flatten_img_.at<float>(2, i) = b_values[i];
    }

    std::tuple<double, double> gains;
    if (algorithm_ == "norm_2") {
        gains = apply_norm_gray_world();
    }
    else if (algorithm_ == "pca") {
        gains = apply_pca_illuminant_estimation();
    }
    else {
        gains = apply_gray_world();
    }

    // Ensure gains are at least 1.0
    double rgain = std::max(1.0, std::get<0>(gains));
    double bgain = std::max(1.0, std::get<1>(gains));

    if (is_debug_) {
        std::cout << "   - AWB Actual Gains: " << std::endl;
        std::cout << "   - AWB - RGain = " << rgain << std::endl;
        std::cout << "   - AWB - Bgain = " << bgain << std::endl;
    }

    // Apply gains to the original Bayer pattern
    cv::Mat result = raw_.clone();
    result.convertTo(result, CV_32F);

    // Create gain matrices for each color
    cv::Mat r_gain_mat = cv::Mat::ones(result.size(), CV_32F) * rgain;
    cv::Mat b_gain_mat = cv::Mat::ones(result.size(), CV_32F) * bgain;
    cv::Mat g_gain_mat = cv::Mat::ones(result.size(), CV_32F);

    // Apply gains based on Bayer pattern
    if (bayer_ == "rggb") {
        for (int i = 0; i < result.rows; i += 2) {
            for (int j = 0; j < result.cols; j += 2) {
                result.at<float>(i, j) *= rgain;      // R
                result.at<float>(i, j + 1) *= 1.0;   // G
                result.at<float>(i + 1, j) *= 1.0;   // G
                result.at<float>(i + 1, j + 1) *= bgain; // B
            }
        }
    }
    else if (bayer_ == "bggr") {
        for (int i = 0; i < result.rows; i += 2) {
            for (int j = 0; j < result.cols; j += 2) {
                result.at<float>(i, j) *= bgain;      // B
                result.at<float>(i, j + 1) *= 1.0;   // G
                result.at<float>(i + 1, j) *= 1.0;   // G
                result.at<float>(i + 1, j + 1) *= rgain; // R
            }
        }
    }
    else if (bayer_ == "grbg") {
        for (int i = 0; i < result.rows; i += 2) {
            for (int j = 0; j < result.cols; j += 2) {
                result.at<float>(i, j) *= 1.0;       // G
                result.at<float>(i, j + 1) *= rgain; // R
                result.at<float>(i + 1, j) *= bgain; // B
                result.at<float>(i + 1, j + 1) *= 1.0; // G
            }
        }
    }
    else if (bayer_ == "gbrg") {
        for (int i = 0; i < result.rows; i += 2) {
            for (int j = 0; j < result.cols; j += 2) {
                result.at<float>(i, j) *= 1.0;       // G
                result.at<float>(i, j + 1) *= bgain; // B
                result.at<float>(i + 1, j) *= rgain; // R
                result.at<float>(i + 1, j + 1) *= 1.0; // G
            }
        }
    }

    // Clip values to valid range
    double max_val = (1 << bit_depth_) - 1;
    cv::threshold(result, result, max_val, max_val, cv::THRESH_TRUNC);

    // Convert back to original type
    cv::Mat result_final;
    result.convertTo(result_final, raw_.type());
    raw_ = result_final;

    return {rgain, bgain};
}

std::tuple<double, double> AutoWhiteBalance::determine_white_balance_gain_eigen()
{
    // Convert to Eigen
    hdr_isp::EigenImage eigen_raw = hdr_isp::opencv_to_eigen(raw_);
    
    // Extract Bayer pattern channels using Eigen
    int height = eigen_raw.rows();
    int width = eigen_raw.cols();
    std::vector<float> r_values, g_values, b_values;

    if (bayer_ == "rggb") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                r_values.push_back(eigen_raw.data()(i, j));
                g_values.push_back(eigen_raw.data()(i, j + 1));
                g_values.push_back(eigen_raw.data()(i + 1, j));
                b_values.push_back(eigen_raw.data()(i + 1, j + 1));
            }
        }
    }
    else if (bayer_ == "bggr") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                b_values.push_back(eigen_raw.data()(i, j));
                g_values.push_back(eigen_raw.data()(i, j + 1));
                g_values.push_back(eigen_raw.data()(i + 1, j));
                r_values.push_back(eigen_raw.data()(i + 1, j + 1));
            }
        }
    }
    else if (bayer_ == "grbg") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                g_values.push_back(eigen_raw.data()(i, j));
                r_values.push_back(eigen_raw.data()(i, j + 1));
                b_values.push_back(eigen_raw.data()(i + 1, j));
                g_values.push_back(eigen_raw.data()(i + 1, j + 1));
            }
        }
    }
    else if (bayer_ == "gbrg") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                g_values.push_back(eigen_raw.data()(i, j));
                b_values.push_back(eigen_raw.data()(i, j + 1));
                r_values.push_back(eigen_raw.data()(i + 1, j));
                g_values.push_back(eigen_raw.data()(i + 1, j + 1));
            }
        }
    }

    // Create flattened image for algorithm input (simplified)
    // In a full implementation, you'd create a proper 3xN Eigen matrix
    flatten_img_ = cv::Mat(3, r_values.size(), CV_32F);
    for (size_t i = 0; i < r_values.size(); i++) {
        flatten_img_.at<float>(0, i) = r_values[i];
        flatten_img_.at<float>(1, i) = g_values[i];
        flatten_img_.at<float>(2, i) = b_values[i];
    }

    std::tuple<double, double> gains;
    if (algorithm_ == "norm_2") {
        gains = apply_norm_gray_world_eigen();
    }
    else if (algorithm_ == "pca") {
        gains = apply_pca_illuminant_estimation_eigen();
    }
    else {
        gains = apply_gray_world_eigen();
    }

    // Ensure gains are at least 1.0
    double rgain = std::max(1.0, std::get<0>(gains));
    double bgain = std::max(1.0, std::get<1>(gains));

    if (is_debug_) {
        std::cout << "   - AWB Actual Gains (Eigen): " << std::endl;
        std::cout << "   - AWB - RGain = " << rgain << std::endl;
        std::cout << "   - AWB - Bgain = " << bgain << std::endl;
    }

    // Apply gains to the original Bayer pattern using Eigen
    hdr_isp::EigenImage result = eigen_raw;

    // Apply gains based on Bayer pattern
    if (bayer_ == "rggb") {
        for (int i = 0; i < result.rows(); i += 2) {
            for (int j = 0; j < result.cols(); j += 2) {
                result.data()(i, j) *= rgain;      // R
                result.data()(i, j + 1) *= 1.0f;   // G
                result.data()(i + 1, j) *= 1.0f;   // G
                result.data()(i + 1, j + 1) *= bgain; // B
            }
        }
    }
    else if (bayer_ == "bggr") {
        for (int i = 0; i < result.rows(); i += 2) {
            for (int j = 0; j < result.cols(); j += 2) {
                result.data()(i, j) *= bgain;      // B
                result.data()(i, j + 1) *= 1.0f;   // G
                result.data()(i + 1, j) *= 1.0f;   // G
                result.data()(i + 1, j + 1) *= rgain; // R
            }
        }
    }
    else if (bayer_ == "grbg") {
        for (int i = 0; i < result.rows(); i += 2) {
            for (int j = 0; j < result.cols(); j += 2) {
                result.data()(i, j) *= 1.0f;       // G
                result.data()(i, j + 1) *= rgain; // R
                result.data()(i + 1, j) *= bgain; // B
                result.data()(i + 1, j + 1) *= 1.0f; // G
            }
        }
    }
    else if (bayer_ == "gbrg") {
        for (int i = 0; i < result.rows(); i += 2) {
            for (int j = 0; j < result.cols(); j += 2) {
                result.data()(i, j) *= 1.0f;       // G
                result.data()(i, j + 1) *= bgain; // B
                result.data()(i + 1, j) *= rgain; // R
                result.data()(i + 1, j + 1) *= 1.0f; // G
            }
        }
    }

    // Clip values to valid range
    float max_val = static_cast<float>((1 << bit_depth_) - 1);
    result = result.cwiseMax(0.0f).cwiseMin(max_val);

    // Convert back to OpenCV and update raw_
    raw_ = hdr_isp::eigen_to_opencv(result);

    return {rgain, bgain};
}

std::tuple<double, double> AutoWhiteBalance::apply_gray_world() {
    GrayWorld gwa(flatten_img_);
    return gwa.calculate_gains();
}

std::tuple<double, double> AutoWhiteBalance::apply_gray_world_eigen() {
    // Simplified Eigen implementation - reuse OpenCV for now
    GrayWorld gwa(flatten_img_);
    return gwa.calculate_gains();
}

std::tuple<double, double> AutoWhiteBalance::apply_norm_gray_world() {
    NormGrayWorld ngw(flatten_img_);
    return ngw.calculate_gains();
}

std::tuple<double, double> AutoWhiteBalance::apply_norm_gray_world_eigen() {
    // Simplified Eigen implementation - reuse OpenCV for now
    NormGrayWorld ngw(flatten_img_);
    return ngw.calculate_gains();
}

std::tuple<double, double> AutoWhiteBalance::apply_pca_illuminant_estimation() {
    float pixel_percentage = parm_awb_["percentage"].as<float>();
    PCAIlluminEstimation pca(flatten_img_, pixel_percentage);
    return pca.calculate_gains();
}

std::tuple<double, double> AutoWhiteBalance::apply_pca_illuminant_estimation_eigen() {
    // Simplified Eigen implementation - reuse OpenCV for now
    float pixel_percentage = parm_awb_["percentage"].as<float>();
    PCAIlluminEstimation pca(flatten_img_, pixel_percentage);
    return pca.calculate_gains();
}

std::array<double, 2> AutoWhiteBalance::execute() {
    if (!enable_) {
        return {1.0, 1.0};
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    auto [rgain, bgain] = determine_white_balance_gain();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "  Execution time: " << elapsed.count() << "s" << std::endl;
    
    return {rgain, bgain};
} 
#include "auto_white_balance.hpp"
#include "gray_world.hpp"
#include "norm_gray_world.hpp"
#include "pca.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

AutoWhiteBalance::AutoWhiteBalance(const hdr_isp::EigenImageU32& raw, const YAML::Node& sensor_info, const YAML::Node& parm_awb)
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
{
}

std::tuple<double, double> AutoWhiteBalance::determine_white_balance_gain()
{
    // Extract Bayer pattern channels using Eigen
    int height = raw_.rows();
    int width = raw_.cols();
    std::vector<float> r_values, g_values, b_values;

    if (bayer_ == "rggb") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                r_values.push_back(static_cast<float>(raw_.data()(i, j)));
                g_values.push_back(static_cast<float>(raw_.data()(i, j + 1)));
                g_values.push_back(static_cast<float>(raw_.data()(i + 1, j)));
                b_values.push_back(static_cast<float>(raw_.data()(i + 1, j + 1)));
            }
        }
    }
    else if (bayer_ == "bggr") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                b_values.push_back(static_cast<float>(raw_.data()(i, j)));
                g_values.push_back(static_cast<float>(raw_.data()(i, j + 1)));
                g_values.push_back(static_cast<float>(raw_.data()(i + 1, j)));
                r_values.push_back(static_cast<float>(raw_.data()(i + 1, j + 1)));
            }
        }
    }
    else if (bayer_ == "grbg") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                g_values.push_back(static_cast<float>(raw_.data()(i, j)));
                r_values.push_back(static_cast<float>(raw_.data()(i, j + 1)));
                b_values.push_back(static_cast<float>(raw_.data()(i + 1, j)));
                g_values.push_back(static_cast<float>(raw_.data()(i + 1, j + 1)));
            }
        }
    }
    else if (bayer_ == "gbrg") {
        for (int i = 0; i < height; i += 2) {
            for (int j = 0; j < width; j += 2) {
                g_values.push_back(static_cast<float>(raw_.data()(i, j)));
                b_values.push_back(static_cast<float>(raw_.data()(i, j + 1)));
                r_values.push_back(static_cast<float>(raw_.data()(i + 1, j)));
                g_values.push_back(static_cast<float>(raw_.data()(i + 1, j + 1)));
            }
        }
    }

    // Create flattened image for algorithm input using Eigen
    flatten_img_ = Eigen::MatrixXf(3, r_values.size());
    for (size_t i = 0; i < r_values.size(); i++) {
        flatten_img_(0, i) = r_values[i];
        flatten_img_(1, i) = g_values[i];
        flatten_img_(2, i) = b_values[i];
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

    // Apply gains to the original Bayer pattern using Eigen
    hdr_isp::EigenImageU32 result = raw_.clone();

    // Apply gains based on Bayer pattern
    if (bayer_ == "rggb") {
        for (int i = 0; i < result.rows(); i += 2) {
            for (int j = 0; j < result.cols(); j += 2) {
                result.data()(i, j) = static_cast<uint32_t>(std::round(result.data()(i, j) * rgain));      // R
                result.data()(i, j + 1) = result.data()(i, j + 1);   // G
                result.data()(i + 1, j) = result.data()(i + 1, j);   // G
                result.data()(i + 1, j + 1) = static_cast<uint32_t>(std::round(result.data()(i + 1, j + 1) * bgain)); // B
            }
        }
    }
    else if (bayer_ == "bggr") {
        for (int i = 0; i < result.rows(); i += 2) {
            for (int j = 0; j < result.cols(); j += 2) {
                result.data()(i, j) = static_cast<uint32_t>(std::round(result.data()(i, j) * bgain));      // B
                result.data()(i, j + 1) = result.data()(i, j + 1);   // G
                result.data()(i + 1, j) = result.data()(i + 1, j);   // G
                result.data()(i + 1, j + 1) = static_cast<uint32_t>(std::round(result.data()(i + 1, j + 1) * rgain)); // R
            }
        }
    }
    else if (bayer_ == "grbg") {
        for (int i = 0; i < result.rows(); i += 2) {
            for (int j = 0; j < result.cols(); j += 2) {
                result.data()(i, j) = result.data()(i, j);       // G
                result.data()(i, j + 1) = static_cast<uint32_t>(std::round(result.data()(i, j + 1) * rgain)); // R
                result.data()(i + 1, j) = static_cast<uint32_t>(std::round(result.data()(i + 1, j) * bgain)); // B
                result.data()(i + 1, j + 1) = result.data()(i + 1, j + 1); // G
            }
        }
    }
    else if (bayer_ == "gbrg") {
        for (int i = 0; i < result.rows(); i += 2) {
            for (int j = 0; j < result.cols(); j += 2) {
                result.data()(i, j) = result.data()(i, j);       // G
                result.data()(i, j + 1) = static_cast<uint32_t>(std::round(result.data()(i, j + 1) * bgain)); // B
                result.data()(i + 1, j) = static_cast<uint32_t>(std::round(result.data()(i + 1, j) * rgain)); // R
                result.data()(i + 1, j + 1) = result.data()(i + 1, j + 1); // G
            }
        }
    }

    // Clip values to valid range
    uint32_t max_val = (1U << bit_depth_) - 1;
    result = result.cwiseMin(max_val);

    // Update the raw image with the result
    raw_ = result;

    return {rgain, bgain};
}

std::tuple<double, double> AutoWhiteBalance::apply_gray_world() {
    // Convert Eigen matrix to OpenCV for the existing GrayWorld implementation
    cv::Mat flatten_cv(3, flatten_img_.cols(), CV_32F);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < flatten_img_.cols(); j++) {
            flatten_cv.at<float>(i, j) = flatten_img_(i, j);
        }
    }
    
    GrayWorld gwa(flatten_cv);
    return gwa.calculate_gains();
}

std::tuple<double, double> AutoWhiteBalance::apply_norm_gray_world() {
    // Convert Eigen matrix to OpenCV for the existing NormGrayWorld implementation
    cv::Mat flatten_cv(3, flatten_img_.cols(), CV_32F);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < flatten_img_.cols(); j++) {
            flatten_cv.at<float>(i, j) = flatten_img_(i, j);
        }
    }
    
    NormGrayWorld ngw(flatten_cv);
    return ngw.calculate_gains();
}

std::tuple<double, double> AutoWhiteBalance::apply_pca_illuminant_estimation() {
    // Convert Eigen matrix to OpenCV for the existing PCA implementation
    cv::Mat flatten_cv(3, flatten_img_.cols(), CV_32F);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < flatten_img_.cols(); j++) {
            flatten_cv.at<float>(i, j) = flatten_img_(i, j);
        }
    }
    
    float pixel_percentage = parm_awb_["percentage"].as<float>();
    PCAIlluminEstimation pca(flatten_cv, pixel_percentage);
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
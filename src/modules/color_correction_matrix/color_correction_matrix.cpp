#include "color_correction_matrix.hpp"
#include <chrono>
#include <iostream>

ColorCorrectionMatrix::ColorCorrectionMatrix(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_ccm)
    : raw_(img)
    , sensor_info_(sensor_info)
    , parm_ccm_(parm_ccm)
    , enable_(parm_ccm["is_enable"].as<bool>())
    , output_bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , is_save_(parm_ccm["is_save"].as<bool>())
{
}

cv::Mat ColorCorrectionMatrix::execute() {
    if (!enable_) {
        return raw_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result = apply_ccm();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Color Correction Matrix execution time: " << duration.count() << " seconds" << std::endl;

    return result;
}

cv::Mat ColorCorrectionMatrix::apply_ccm() {
    // Get CCM parameters
    std::vector<float> corrected_red = parm_ccm_["corrected_red"].as<std::vector<float>>();
    std::vector<float> corrected_green = parm_ccm_["corrected_green"].as<std::vector<float>>();
    std::vector<float> corrected_blue = parm_ccm_["corrected_blue"].as<std::vector<float>>();

    // Create CCM matrix
    ccm_mat_ = cv::Mat(3, 3, CV_32F);
    for (int i = 0; i < 3; i++) {
        ccm_mat_.at<float>(0, i) = corrected_red[i];
        ccm_mat_.at<float>(1, i) = corrected_green[i];
        ccm_mat_.at<float>(2, i) = corrected_blue[i];
    }

    // Normalize image to 0-1 range
    cv::Mat normalized;
    raw_.convertTo(normalized, CV_32F);
    normalized /= (1 << output_bit_depth_) - 1;

    // Reshape image to Nx3 matrix
    cv::Mat reshaped = normalized.reshape(1, normalized.total());

    // Apply CCM
    cv::Mat result;
    cv::gemm(reshaped, ccm_mat_, 1.0, cv::Mat(), 0.0, result);

    // Clip values to [0, 1]
    cv::threshold(result, result, 0, 1, cv::THRESH_TRUNC);

    // Reshape back to original dimensions
    result = result.reshape(3, normalized.rows);

    // Convert back to original bit depth
    cv::Mat output;
    result.convertTo(output, raw_.type(), (1 << output_bit_depth_) - 1);

    return output;
} 
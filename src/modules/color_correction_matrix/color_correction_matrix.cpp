#include "color_correction_matrix.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>

ColorCorrectionMatrix::ColorCorrectionMatrix(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_ccm)
    : raw_(img)
    , sensor_info_(sensor_info)
    , parm_ccm_(parm_ccm)
    , enable_(parm_ccm["is_enable"].as<bool>())
    , output_bit_depth_(sensor_info["bit_depth"].as<int>())
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

    if (is_save_) {
        try {
            std::filesystem::create_directories("out_frames/intermediate");
            std::string output_path = "out_frames/intermediate/Out_color_correction_matrix_" + 
                                    std::to_string(result.cols) + "x" + std::to_string(result.rows) + ".png";
            
            // Convert to 8-bit before saving
            cv::Mat save_img;
            result.convertTo(save_img, CV_8U, 255.0 / ((1 << output_bit_depth_) - 1));
            
            // Debug prints for image statistics
            double min_val, max_val;
            cv::minMaxLoc(save_img, &min_val, &max_val);
            cv::Scalar mean_val = cv::mean(save_img);
            std::cout << "CCM Save image statistics:" << std::endl;
            std::cout << "  Mean: " << mean_val[0] << std::endl;
            std::cout << "  Min: " << min_val << std::endl;
            std::cout << "  Max: " << max_val << std::endl;
            std::cout << "  Image size: " << save_img.size() << std::endl;
            std::cout << "  Number of channels: " << save_img.channels() << std::endl;
            
            bool write_success = cv::imwrite(output_path, save_img);
            if (!write_success) {
                std::cerr << "Error: Failed to write image to: " << output_path << std::endl;
            } else {
                std::cout << "Successfully wrote image to: " << output_path << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error saving image: " << e.what() << std::endl;
        }
    }

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

    // Print input image statistics
    std::cout << "CCM Input image statistics:" << std::endl;
    std::cout << "  Type: " << raw_.type() << " (CV_8U=" << CV_8U << ", CV_16U=" << CV_16U << ", CV_32F=" << CV_32F << ")" << std::endl;
    std::cout << "  Channels: " << raw_.channels() << std::endl;
    std::cout << "  Size: " << raw_.size() << std::endl;
    double min_val, max_val;
    cv::minMaxLoc(raw_, &min_val, &max_val);
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;

    // Convert to float and normalize to 0-1 range
    cv::Mat normalized;
    raw_.convertTo(normalized, CV_32F);
    int input_bit_depth = sensor_info_["bit_depth"].as<int>();
    float scale = 1.0f / ((1 << input_bit_depth) - 1);
    normalized *= scale;

    // Ensure values are in [0,1] range
    cv::threshold(normalized, normalized, 0, 1, cv::THRESH_TRUNC);

    // Print normalized image statistics
    std::cout << "CCM Normalized image statistics:" << std::endl;
    cv::minMaxLoc(normalized, &min_val, &max_val);
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;

    // Reshape image to Nx3 matrix
    cv::Mat reshaped = normalized.reshape(1, normalized.total());

    // Apply CCM
    cv::Mat result;
    cv::gemm(reshaped, ccm_mat_, 1.0, cv::Mat(), 0.0, result);

    // Clip values to [0, 1]
    cv::threshold(result, result, 0, 1, cv::THRESH_TRUNC);

    // Reshape back to original dimensions
    result = result.reshape(3, normalized.rows);

    // Print result statistics before conversion
    std::cout << "CCM Result before conversion statistics:" << std::endl;
    cv::minMaxLoc(result, &min_val, &max_val);
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;

    // Convert back to output bit depth
    cv::Mat output;
    int output_bit_depth = output_bit_depth_;
    result.convertTo(output, CV_16UC3, (1 << output_bit_depth) - 1);

    // Print output image statistics
    std::cout << "CCM Output image statistics:" << std::endl;
    std::cout << "  Type: " << output.type() << std::endl;
    std::cout << "  Channels: " << output.channels() << std::endl;
    std::cout << "  Size: " << output.size() << std::endl;
    cv::minMaxLoc(output, &min_val, &max_val);
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;

    return output;
} 
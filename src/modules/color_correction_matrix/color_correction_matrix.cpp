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
    , use_eigen_(true) // Use Eigen by default
{
}

cv::Mat ColorCorrectionMatrix::execute() {
    if (!enable_) {
        return raw_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat result;
    if (use_eigen_) {
        result = apply_ccm_eigen();
    } else {
        result = apply_ccm_opencv();
    }
    
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

cv::Mat ColorCorrectionMatrix::apply_ccm_opencv() {
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

    // Debug print CCM matrix
    std::cout << "CCM Matrix:" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << ccm_mat_.at<float>(i, j) << "\t";
        }
        std::cout << std::endl;
    }

    // Print input image statistics
    std::cout << "CCM Input image statistics:" << std::endl;
    std::cout << "  Type: " << raw_.type() << " (CV_8U=" << CV_8U << ", CV_16U=" << CV_16U << ", CV_32F=" << CV_32F << ")" << std::endl;
    std::cout << "  Channels: " << raw_.channels() << std::endl;
    std::cout << "  Size: " << raw_.size() << std::endl;
    double min_val, max_val;
    cv::minMaxLoc(raw_, &min_val, &max_val);
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;

    // Convert to float for CCM operation
    cv::Mat float_img;
    raw_.convertTo(float_img, CV_32F);

    // Reshape image to Nx3 matrix
    cv::Mat reshaped = float_img.reshape(1, float_img.total());

    // Debug print reshaped matrix dimensions
    std::cout << "Reshaped matrix dimensions: " << reshaped.rows << "x" << reshaped.cols << std::endl;

    // Apply CCM
    cv::Mat result;
    
    // Transpose CCM matrix for correct multiplication order
    cv::Mat ccm_transposed;
    cv::transpose(ccm_mat_, ccm_transposed);
    
    // Debug print for center pixel CCM multiplication
    int center_idx = reshaped.rows / 2;
    std::cout << "\nCCM Matrix multiplication for center pixel:" << std::endl;
    std::cout << "Input RGB values: [" << reshaped.at<float>(center_idx, 0) << ", "
              << reshaped.at<float>(center_idx, 1) << ", "
              << reshaped.at<float>(center_idx, 2) << "]" << std::endl;
    
    std::cout << "CCM Matrix (transposed):" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "[";
        for (int j = 0; j < 3; j++) {
            std::cout << ccm_transposed.at<float>(i, j);
            if (j < 2) std::cout << " ";
        }
        std::cout << "]" << std::endl;
    }
    
    cv::gemm(reshaped, ccm_transposed, 1.0, cv::Mat(), 0.0, result);
    
    // Print result for center pixel
    std::cout << "Output RGB values: [" << result.at<float>(center_idx, 0) << ", "
              << result.at<float>(center_idx, 1) << ", "
              << result.at<float>(center_idx, 2) << "]" << std::endl << std::endl;

    // Reshape back to original dimensions
    result = result.reshape(3, float_img.rows);

    // Print result statistics before conversion
    std::cout << "CCM Result before conversion statistics:" << std::endl;
    cv::minMaxLoc(result, &min_val, &max_val);
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;

    // Convert back to output bit depth
    cv::Mat output;
    result.convertTo(output, CV_16UC3);

    // Print output image statistics
    std::cout << "CCM Output image statistics:" << std::endl;
    std::cout << "  Type: " << output.type() << std::endl;
    std::cout << "  Channels: " << output.channels() << std::endl;
    std::cout << "  Size: " << output.size() << std::endl;
    cv::minMaxLoc(output, &min_val, &max_val);
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;

    return output;
}

cv::Mat ColorCorrectionMatrix::apply_ccm_eigen() {
    // Get CCM parameters
    std::vector<float> corrected_red = parm_ccm_["corrected_red"].as<std::vector<float>>();
    std::vector<float> corrected_green = parm_ccm_["corrected_green"].as<std::vector<float>>();
    std::vector<float> corrected_blue = parm_ccm_["corrected_blue"].as<std::vector<float>>();

    // Create CCM matrix using Eigen
    Eigen::Matrix3f ccm_eigen;
    ccm_eigen.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data());
    ccm_eigen.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data());
    ccm_eigen.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data());

    // Convert input to EigenImage3C for 3-channel RGB image
    hdr_isp::EigenImage3C eigen_img = hdr_isp::EigenImage3C::fromOpenCV(raw_);

    // Apply CCM using matrix multiplication
    hdr_isp::EigenImage3C result = eigen_img * ccm_eigen;

    // Apply bit depth conversion
    if (output_bit_depth_ == 8) {
        result = result.clip(0.0f, 255.0f);
    } else if (output_bit_depth_ == 16) {
        result = result.clip(0.0f, 65535.0f);
    }

    // Convert back to OpenCV Mat
    cv::Mat output = result.toOpenCV(CV_16UC3);
    return output;
} 
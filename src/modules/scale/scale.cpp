#include "scale.hpp"
#include "scale_2d.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <regex>

namespace fs = std::filesystem;

Scale::Scale(cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info,
            const YAML::Node& parm_sca, int conv_std)
    : img_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_sca_(YAML::Clone(parm_sca))
    , enable_(parm_sca_["is_enable"].as<bool>())
    , is_save_(parm_sca_["is_save"].as<bool>())
    , conv_std_(conv_std)
    , use_eigen_(true) // Use Eigen by default
{
    get_scaling_params();
}

void Scale::get_scaling_params() {
    is_debug_ = parm_sca_["is_debug"].as<bool>();
    old_size_ = std::make_pair(sensor_info_["height"].as<int>(), sensor_info_["width"].as<int>());
    new_size_ = std::make_pair(parm_sca_["new_height"].as<int>(), parm_sca_["new_width"].as<int>());
}

cv::Mat Scale::apply_scaling_opencv() {
    // Check if no change in size
    if (old_size_ == new_size_) {
        if (is_debug_) {
            std::cout << "   - Output size is the same as input size." << std::endl;
        }
        return img_;
    }

    cv::Mat scaled_img;
    if (img_.type() == CV_32F) {
        scaled_img = cv::Mat(new_size_.first, new_size_.second, CV_32FC3);
    } else {
        scaled_img = cv::Mat(new_size_.first, new_size_.second, CV_8UC3);
    }

    // Loop over each channel to resize the image
    std::vector<cv::Mat> channels;
    cv::split(img_, channels);
    std::vector<cv::Mat> scaled_channels;

    for (int i = 0; i < 3; ++i) {
        Scale2D scale_2d(channels[i], sensor_info_, parm_sca_);
        cv::Mat scaled_ch = scale_2d.execute();

        // If input size is invalid, return the original image
        if (scaled_ch.size() == cv::Size(old_size_.second, old_size_.first)) {
            return img_;
        }

        scaled_channels.push_back(scaled_ch);
        // Turn off debug flag after first channel
        parm_sca_["is_debug"] = false;
    }

    cv::merge(scaled_channels, scaled_img);
    return scaled_img;
}

hdr_isp::EigenImage Scale::apply_scaling_eigen() {
    // Check if no change in size
    if (old_size_ == new_size_) {
        if (is_debug_) {
            std::cout << "   - Output size is the same as input size." << std::endl;
        }
        return hdr_isp::opencv_to_eigen(img_);
    }

    // Convert to Eigen
    hdr_isp::EigenImage eigen_img = hdr_isp::opencv_to_eigen(img_);
    int old_rows = eigen_img.rows();
    int old_cols = eigen_img.cols();
    int new_rows = new_size_.first;
    int new_cols = new_size_.second;

    // Create scaled image
    hdr_isp::EigenImage scaled_img = hdr_isp::EigenImage::Zero(new_rows, new_cols);

    // Simple nearest neighbor scaling for Eigen implementation
    float scale_x = static_cast<float>(old_cols) / new_cols;
    float scale_y = static_cast<float>(old_rows) / new_rows;

    for (int i = 0; i < new_rows; ++i) {
        for (int j = 0; j < new_cols; ++j) {
            int src_i = static_cast<int>(i * scale_y);
            int src_j = static_cast<int>(j * scale_x);
            
            // Clamp to valid range
            src_i = std::min(src_i, old_rows - 1);
            src_j = std::min(src_j, old_cols - 1);
            
            scaled_img(i, j) = eigen_img(src_i, src_j);
        }
    }

    return scaled_img;
}

void Scale::save() {
    if (is_save_) {
        // Update size in filename
        std::string filename = platform_["in_file"].as<std::string>();
        std::regex size_pattern(R"(\d+x\d+)");
        std::string new_size_str = std::to_string(img_.cols) + "x" + std::to_string(img_.rows);
        filename = std::regex_replace(filename, size_pattern, new_size_str);

        std::string output_path = "out_frames/intermediate/Out_scale_" + filename;
        cv::imwrite(output_path, img_);
    }
}

cv::Mat Scale::execute() {
    std::cout << "Scale = " << (enable_ ? "true" : "false") << std::endl;

    if (enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (use_eigen_) {
            hdr_isp::EigenImage eigen_result = apply_scaling_eigen();
            img_ = hdr_isp::eigen_to_opencv(eigen_result);
        } else {
            img_ = apply_scaling_opencv();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "  Execution time: " << duration.count() << "s" << std::endl;
    }
    save();
    return img_;
} 
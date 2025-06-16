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
    , algorithm_(parm_awb["algorithm"].as<std::string>()) {
}

std::tuple<double, double> AutoWhiteBalance::determine_white_balance_gain() {
    auto total_start = std::chrono::high_resolution_clock::now();

    int max_pixel_value = (1 << bit_depth_) - 1;
    float approx_percentage = max_pixel_value / 100.0f;
    
    // Calculate overexposed and underexposed limits
    float overexposed_limit = max_pixel_value - (overexposed_percentage_ * approx_percentage);
    float underexposed_limit = underexposed_percentage_ * approx_percentage;

    if (is_debug_) {
        std::cout << "   - AWB - Underexposed Pixel Limit = " << underexposed_limit << std::endl;
        std::cout << "   - AWB - Overexposed Pixel Limit  = " << overexposed_limit << std::endl;
    }

    // Extract Bayer channels
    cv::Mat r_channel, gr_channel, gb_channel, b_channel;
    
    if (is_debug_) {
        std::cout << "   - AWB - Input image dimensions: " << raw_.cols << "x" << raw_.rows << std::endl;
        std::cout << "   - AWB - Bayer pattern: " << bayer_ << std::endl;
    }

    // Ensure input dimensions are even
    if (raw_.cols % 2 != 0 || raw_.rows % 2 != 0) {
        throw std::runtime_error("Input image dimensions must be even");
    }

    // Ensure input is not empty
    if (raw_.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    int half_cols = raw_.cols / 2;
    int half_rows = raw_.rows / 2;

    if (is_debug_) {
        std::cout << "   - AWB - Half dimensions: " << half_cols << "x" << half_rows << std::endl;
    }
    
    try {
        if (bayer_ == "rggb") {
            r_channel = raw_(cv::Rect(0, 0, half_cols, half_rows));
            gr_channel = raw_(cv::Rect(half_cols, 0, half_cols, half_rows));
            gb_channel = raw_(cv::Rect(0, half_rows, half_cols, half_rows));
            b_channel = raw_(cv::Rect(half_cols, half_rows, half_cols, half_rows));
        }
        else if (bayer_ == "bggr") {
            b_channel = raw_(cv::Rect(0, 0, half_cols, half_rows));
            gb_channel = raw_(cv::Rect(half_cols, 0, half_cols, half_rows));
            gr_channel = raw_(cv::Rect(0, half_rows, half_cols, half_rows));
            r_channel = raw_(cv::Rect(half_cols, half_rows, half_cols, half_rows));
        }
        else if (bayer_ == "grbg") {
            gr_channel = raw_(cv::Rect(0, 0, half_cols, half_rows));
            r_channel = raw_(cv::Rect(half_cols, 0, half_cols, half_rows));
            b_channel = raw_(cv::Rect(0, half_rows, half_cols, half_rows));
            gb_channel = raw_(cv::Rect(half_cols, half_rows, half_cols, half_rows));
        }
        else if (bayer_ == "gbrg") {
            gb_channel = raw_(cv::Rect(0, 0, half_cols, half_rows));
            b_channel = raw_(cv::Rect(half_cols, 0, half_cols, half_rows));
            r_channel = raw_(cv::Rect(0, half_rows, half_cols, half_rows));
            gr_channel = raw_(cv::Rect(half_cols, half_rows, half_cols, half_rows));
        }
        else {
            throw std::runtime_error("Unsupported Bayer pattern: " + bayer_);
        }
    }
    catch (const cv::Exception& e) {
        throw std::runtime_error("OpenCV error during channel extraction: " + std::string(e.what()));
    }

    // Verify channel dimensions
    if (r_channel.empty() || gr_channel.empty() || gb_channel.empty() || b_channel.empty()) {
        throw std::runtime_error("Failed to extract Bayer channels");
    }

    if (is_debug_) {
        std::cout << "   - AWB - Channel dimensions:" << std::endl;
        std::cout << "     R: " << r_channel.cols << "x" << r_channel.rows << std::endl;
        std::cout << "     Gr: " << gr_channel.cols << "x" << gr_channel.rows << std::endl;
        std::cout << "     Gb: " << gb_channel.cols << "x" << gb_channel.rows << std::endl;
        std::cout << "     B: " << b_channel.cols << "x" << b_channel.rows << std::endl;
    }

    // Calculate average green channel
    cv::Mat g_channel;
    cv::add(gr_channel, gb_channel, g_channel);
    g_channel.convertTo(g_channel, g_channel.type(), 0.5);

    if (is_debug_) {
        std::cout << "   - AWB - Green channel dimensions: " << g_channel.cols << "x" << g_channel.rows << std::endl;
    }

    // Ensure all channels have the same type
    cv::Mat r_channel_float, g_channel_float, b_channel_float;
    r_channel.convertTo(r_channel_float, CV_32F);
    g_channel.convertTo(g_channel_float, CV_32F);
    b_channel.convertTo(b_channel_float, CV_32F);

    // Stack channels for gain calculation
    std::vector<cv::Mat> channels = {r_channel_float, g_channel_float, b_channel_float};
    cv::Mat bayer_channels;
    cv::merge(channels, bayer_channels);

    if (is_debug_) {
        std::cout << "   - AWB - Merged channels dimensions: " << bayer_channels.cols << "x" << bayer_channels.rows << std::endl;
        std::cout << "   - AWB - Merged channels type: " << bayer_channels.type() << std::endl;
    }

    // Prepare flattened image for gain calculation
    cv::Mat flattened = bayer_channels.reshape(1, bayer_channels.total());
    flatten_img_ = flattened;

    if (is_debug_) {
        std::cout << "   - AWB - Flattened image dimensions: " << flatten_img_.cols << "x" << flatten_img_.rows << std::endl;
        std::cout << "   - AWB - Flattened image type: " << flatten_img_.type() << std::endl;
    }

    // Apply selected algorithm
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

    if (bayer_ == "rggb") {
        // Red pixels
        for (int i = 0; i < result.rows; i += 2) {
            for (int j = 0; j < result.cols; j += 2) {
                result.at<float>(i, j) *= r_gain_mat.at<float>(i, j);
            }
        }
        // Blue pixels
        for (int i = 1; i < result.rows; i += 2) {
            for (int j = 1; j < result.cols; j += 2) {
                result.at<float>(i, j) *= b_gain_mat.at<float>(i, j);
            }
        }
    }
    else if (bayer_ == "bggr") {
        // Blue pixels
        for (int i = 0; i < result.rows; i += 2) {
            for (int j = 0; j < result.cols; j += 2) {
                result.at<float>(i, j) *= b_gain_mat.at<float>(i, j);
            }
        }
        // Red pixels
        for (int i = 1; i < result.rows; i += 2) {
            for (int j = 1; j < result.cols; j += 2) {
                result.at<float>(i, j) *= r_gain_mat.at<float>(i, j);
            }
        }
    }
    else if (bayer_ == "grbg") {
        // Blue pixels
        for (int i = 1; i < result.rows; i += 2) {
            for (int j = 0; j < result.cols; j += 2) {
                result.at<float>(i, j) *= b_gain_mat.at<float>(i, j);
            }
        }
        // Red pixels
        for (int i = 0; i < result.rows; i += 2) {
            for (int j = 1; j < result.cols; j += 2) {
                result.at<float>(i, j) *= r_gain_mat.at<float>(i, j);
            }
        }
    }
    else if (bayer_ == "gbrg") {
        // Red pixels
        for (int i = 1; i < result.rows; i += 2) {
            for (int j = 0; j < result.cols; j += 2) {
                result.at<float>(i, j) *= r_gain_mat.at<float>(i, j);
            }
        }
        // Blue pixels
        for (int i = 0; i < result.rows; i += 2) {
            for (int j = 1; j < result.cols; j += 2) {
                result.at<float>(i, j) *= b_gain_mat.at<float>(i, j);
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

std::tuple<double, double> AutoWhiteBalance::apply_gray_world() {
    GrayWorld gwa(flatten_img_);
    return gwa.calculate_gains();
}

std::tuple<double, double> AutoWhiteBalance::apply_norm_gray_world() {
    NormGrayWorld ngw(flatten_img_);
    return ngw.calculate_gains();
}

std::tuple<double, double> AutoWhiteBalance::apply_pca_illuminant_estimation() {
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
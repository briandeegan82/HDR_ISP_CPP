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
    
    if (bayer_ == "rggb") {
        r_channel = raw_(cv::Rect(0, 0, raw_.cols/2, raw_.rows/2));
        gr_channel = raw_(cv::Rect(raw_.cols/2, 0, raw_.cols/2, raw_.rows/2));
        gb_channel = raw_(cv::Rect(0, raw_.rows/2, raw_.cols/2, raw_.rows/2));
        b_channel = raw_(cv::Rect(raw_.cols/2, raw_.rows/2, raw_.cols/2, raw_.rows/2));
    }
    else if (bayer_ == "bggr") {
        b_channel = raw_(cv::Rect(0, 0, raw_.cols/2, raw_.rows/2));
        gb_channel = raw_(cv::Rect(raw_.cols/2, 0, raw_.cols/2, raw_.rows/2));
        gr_channel = raw_(cv::Rect(0, raw_.rows/2, raw_.cols/2, raw_.rows/2));
        r_channel = raw_(cv::Rect(raw_.cols/2, raw_.rows/2, raw_.cols/2, raw_.rows/2));
    }
    else if (bayer_ == "grbg") {
        gr_channel = raw_(cv::Rect(0, 0, raw_.cols/2, raw_.rows/2));
        r_channel = raw_(cv::Rect(raw_.cols/2, 0, raw_.cols/2, raw_.rows/2));
        b_channel = raw_(cv::Rect(0, raw_.rows/2, raw_.cols/2, raw_.rows/2));
        gb_channel = raw_(cv::Rect(raw_.cols/2, raw_.rows/2, raw_.cols/2, raw_.rows/2));
    }
    else if (bayer_ == "gbrg") {
        gb_channel = raw_(cv::Rect(0, 0, raw_.cols/2, raw_.rows/2));
        b_channel = raw_(cv::Rect(raw_.cols/2, 0, raw_.cols/2, raw_.rows/2));
        r_channel = raw_(cv::Rect(0, raw_.rows/2, raw_.cols/2, raw_.rows/2));
        gr_channel = raw_(cv::Rect(raw_.cols/2, raw_.rows/2, raw_.cols/2, raw_.rows/2));
    }

    // Calculate average green channel
    cv::Mat g_channel = (gr_channel + gb_channel) * 0.5;

    // Stack channels
    std::vector<cv::Mat> channels = {r_channel, g_channel, b_channel};
    cv::Mat bayer_channels;
    cv::merge(channels, bayer_channels);

    // Remove bad pixels
    cv::Mat bad_pixels = (bayer_channels < underexposed_limit) | (bayer_channels > overexposed_limit);
    cv::Mat bad_pixel_sum;
    cv::reduce(bad_pixels.reshape(1, bad_pixels.total()), bad_pixel_sum, 1, cv::REDUCE_SUM);
    
    std::vector<cv::Mat> valid_pixels;
    for (int c = 0; c < 3; ++c) {
        cv::Mat channel = bayer_channels.reshape(1, bayer_channels.total()).col(c);
        cv::Mat valid = channel.clone();
        valid.setTo(0, bad_pixel_sum > 0);
        valid_pixels.push_back(valid);
    }
    
    cv::Mat valid_channels;
    cv::merge(valid_pixels, valid_channels);
    flatten_img_ = valid_channels.reshape(3, valid_channels.total());

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
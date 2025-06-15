#include "bayer_noise_reduction.hpp"
#include <chrono>
#include <iostream>

BayerNoiseReduction::BayerNoiseReduction(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_bnr)
    : raw_(img)
    , sensor_info_(sensor_info)
    , parm_bnr_(parm_bnr)
    , enable_(parm_bnr["is_enable"].as<bool>())
    , bit_depth_(sensor_info["hdr_bit_depth"].as<int>())
    , bayer_pattern_(sensor_info["bayer_pattern"].as<std::string>())
    , width_(sensor_info["width"].as<int>())
    , height_(sensor_info["height"].as<int>())
    , is_save_(parm_bnr["is_save"].as<bool>())
{
}

cv::Mat BayerNoiseReduction::execute() {
    if (!enable_) {
        return raw_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result = apply_bnr();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Bayer Noise Reduction execution time: " << duration.count() << " seconds" << std::endl;

    return result;
}

cv::Mat BayerNoiseReduction::apply_bnr() {
    // Convert to float32 and normalize
    cv::Mat in_img;
    raw_.convertTo(in_img, CV_32F);
    if (in_img.at<float>(0, 0) > 1.0f) {
        in_img /= (1 << bit_depth_) - 1;
    }

    // Extract R and B channels
    cv::Mat in_img_r, in_img_b;
    extract_channels(in_img, in_img_r, in_img_b);

    // Interpolate green channel
    cv::Mat interp_g = interpolate_green_channel(in_img);

    // Extract guide image at R and B positions
    cv::Mat guide_r, guide_b;
    if (bayer_pattern_ == "rggb") {
        guide_r = interp_g(cv::Rect(0, 0, width_/2, height_/2));
        guide_b = interp_g(cv::Rect(1, 1, width_/2, height_/2));
    } else if (bayer_pattern_ == "bggr") {
        guide_r = interp_g(cv::Rect(1, 1, width_/2, height_/2));
        guide_b = interp_g(cv::Rect(0, 0, width_/2, height_/2));
    } else if (bayer_pattern_ == "grbg") {
        guide_r = interp_g(cv::Rect(1, 0, width_/2, height_/2));
        guide_b = interp_g(cv::Rect(0, 1, width_/2, height_/2));
    } else if (bayer_pattern_ == "gbrg") {
        guide_r = interp_g(cv::Rect(0, 1, width_/2, height_/2));
        guide_b = interp_g(cv::Rect(1, 0, width_/2, height_/2));
    }

    // Get filter parameters
    int filt_size_g = parm_bnr_["filter_window"].as<int>();
    int filt_size_r = (filt_size_g + 1) / 2;
    int filt_size_b = (filt_size_g + 1) / 2;

    // Apply joint bilateral filter
    cv::Mat out_img_r = fast_joint_bilateral_filter(
        in_img_r, guide_r, filt_size_r,
        parm_bnr_["r_std_dev_r"].as<double>(),
        parm_bnr_["r_std_dev_s"].as<double>()
    );

    cv::Mat out_img_g = fast_joint_bilateral_filter(
        interp_g, interp_g, filt_size_g,
        parm_bnr_["g_std_dev_r"].as<double>(),
        parm_bnr_["g_std_dev_s"].as<double>()
    );

    cv::Mat out_img_b = fast_joint_bilateral_filter(
        in_img_b, guide_b, filt_size_b,
        parm_bnr_["b_std_dev_r"].as<double>(),
        parm_bnr_["b_std_dev_s"].as<double>()
    );

    // Combine channels
    cv::Mat bnr_out_img;
    combine_channels(out_img_r, out_img_g, out_img_b, bnr_out_img);

    // Convert back to original bit depth
    cv::Mat result;
    bnr_out_img.convertTo(result, CV_32F, (1 << bit_depth_) - 1);
    return result;
}

void BayerNoiseReduction::extract_channels(const cv::Mat& img, cv::Mat& r_channel, cv::Mat& b_channel) {
    if (bayer_pattern_ == "rggb") {
        r_channel = img(cv::Rect(0, 0, width_/2, height_/2));
        b_channel = img(cv::Rect(1, 1, width_/2, height_/2));
    } else if (bayer_pattern_ == "bggr") {
        r_channel = img(cv::Rect(1, 1, width_/2, height_/2));
        b_channel = img(cv::Rect(0, 0, width_/2, height_/2));
    } else if (bayer_pattern_ == "grbg") {
        r_channel = img(cv::Rect(1, 0, width_/2, height_/2));
        b_channel = img(cv::Rect(0, 1, width_/2, height_/2));
    } else if (bayer_pattern_ == "gbrg") {
        r_channel = img(cv::Rect(0, 1, width_/2, height_/2));
        b_channel = img(cv::Rect(1, 0, width_/2, height_/2));
    }
}

void BayerNoiseReduction::combine_channels(const cv::Mat& r_channel, const cv::Mat& g_channel, const cv::Mat& b_channel, cv::Mat& output) {
    output = g_channel.clone();
    
    if (bayer_pattern_ == "rggb") {
        r_channel.copyTo(output(cv::Rect(0, 0, width_/2, height_/2)));
        b_channel.copyTo(output(cv::Rect(1, 1, width_/2, height_/2)));
    } else if (bayer_pattern_ == "bggr") {
        r_channel.copyTo(output(cv::Rect(1, 1, width_/2, height_/2)));
        b_channel.copyTo(output(cv::Rect(0, 0, width_/2, height_/2)));
    } else if (bayer_pattern_ == "grbg") {
        r_channel.copyTo(output(cv::Rect(1, 0, width_/2, height_/2)));
        b_channel.copyTo(output(cv::Rect(0, 1, width_/2, height_/2)));
    } else if (bayer_pattern_ == "gbrg") {
        r_channel.copyTo(output(cv::Rect(0, 1, width_/2, height_/2)));
        b_channel.copyTo(output(cv::Rect(1, 0, width_/2, height_/2)));
    }
}

cv::Mat BayerNoiseReduction::interpolate_green_channel(const cv::Mat& img) {
    // Define the G interpolation kernel (5x5)
    float kernel_data[] = {
        0, 0, -1, 0, 0,
        0, 0, 2, 0, 0,
        -1, 2, 4, 2, -1,
        0, 0, 2, 0, 0,
        0, 0, -1, 0, 0
    };
    cv::Mat kernel(5, 5, CV_32F, kernel_data);
    kernel /= 8.0f;

    // Apply the kernel
    cv::Mat interp_g;
    cv::filter2D(img, interp_g, -1, kernel);
    cv::threshold(interp_g, interp_g, 0, 1, cv::THRESH_TRUNC);
    return interp_g;
}

cv::Mat BayerNoiseReduction::fast_joint_bilateral_filter(const cv::Mat& src, const cv::Mat& guide, int d, double sigmaColor, double sigmaSpace) {
    cv::Mat result;
    cv::ximgproc::jointBilateralFilter(guide, src, result, d, sigmaColor, sigmaSpace);
    return result;
} 
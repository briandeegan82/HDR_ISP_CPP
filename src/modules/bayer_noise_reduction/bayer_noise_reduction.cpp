#include "bayer_noise_reduction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

BayerNoiseReduction::BayerNoiseReduction(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_bnr)
    : raw_(img.clone())
    , sensor_info_(sensor_info)
    , parm_bnr_(parm_bnr)
    , enable_(parm_bnr["is_enable"].as<bool>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , bayer_pattern_(sensor_info["bayer_pattern"].as<std::string>())
    , width_(img.cols)
    , height_(img.rows)
    , is_save_(parm_bnr["is_save"].as<bool>())
    , use_eigen_(true) // Use Eigen by default
{
}

cv::Mat BayerNoiseReduction::execute() {
    if (!enable_) {
        return raw_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat result;
    if (use_eigen_) {
        hdr_isp::EigenImage32 eigen_result = apply_bnr_eigen();
        result = eigen_result.toOpenCV(raw_.type());
    } else {
        result = apply_bnr();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "  Execution time: " << elapsed.count() << "s" << std::endl;

    return result;
}

cv::Mat BayerNoiseReduction::apply_bnr() {
    cv::Mat r_channel, b_channel;
    extract_channels(raw_, r_channel, b_channel);
    cv::Mat g_channel = interpolate_green_channel(raw_);

    // Apply bilateral filter to each channel
    int d = parm_bnr_["filter_window"].as<int>();
    double sigmaColor_r = parm_bnr_["r_std_dev_s"].as<double>();
    double sigmaColor_g = parm_bnr_["g_std_dev_s"].as<double>();
    double sigmaColor_b = parm_bnr_["b_std_dev_s"].as<double>();
    double sigmaSpace_r = parm_bnr_["r_std_dev_r"].as<double>();
    double sigmaSpace_g = parm_bnr_["g_std_dev_r"].as<double>();
    double sigmaSpace_b = parm_bnr_["b_std_dev_r"].as<double>();
    cv::Mat filtered_r = bilateral_filter(r_channel, d, sigmaColor_r, sigmaSpace_r);
    cv::Mat filtered_g = bilateral_filter(g_channel, d, sigmaColor_g, sigmaSpace_g);
    cv::Mat filtered_b = bilateral_filter(b_channel, d, sigmaColor_b, sigmaSpace_b);

    cv::Mat output;
    combine_channels(filtered_r, filtered_g, filtered_b, output);

    return output;
}

hdr_isp::EigenImage32 BayerNoiseReduction::apply_bnr_eigen() {
    hdr_isp::EigenImage32 eigen_raw = hdr_isp::EigenImage32::fromOpenCV(raw_);
    hdr_isp::EigenImage32 r_channel, b_channel;
    extract_channels_eigen(eigen_raw, r_channel, b_channel);
    hdr_isp::EigenImage32 g_channel = interpolate_green_channel_eigen(eigen_raw);

    // Apply bilateral filter to each channel
    int d = parm_bnr_["filter_window"].as<int>();
    double sigmaColor_r = parm_bnr_["r_std_dev_s"].as<double>();
    double sigmaColor_g = parm_bnr_["g_std_dev_s"].as<double>();
    double sigmaColor_b = parm_bnr_["b_std_dev_s"].as<double>();
    double sigmaSpace_r = parm_bnr_["r_std_dev_r"].as<double>();
    double sigmaSpace_g = parm_bnr_["g_std_dev_r"].as<double>();
    double sigmaSpace_b = parm_bnr_["b_std_dev_r"].as<double>();
    hdr_isp::EigenImage32 filtered_r = bilateral_filter_eigen(r_channel, d, sigmaColor_r, sigmaSpace_r);
    hdr_isp::EigenImage32 filtered_g = bilateral_filter_eigen(g_channel, d, sigmaColor_g, sigmaSpace_g);
    hdr_isp::EigenImage32 filtered_b = bilateral_filter_eigen(b_channel, d, sigmaColor_b, sigmaSpace_b);

    hdr_isp::EigenImage32 output;
    combine_channels_eigen(filtered_r, filtered_g, filtered_b, output);

    return output;
}

void BayerNoiseReduction::extract_channels(const cv::Mat& img, cv::Mat& r_channel, cv::Mat& b_channel) {
    r_channel = cv::Mat::zeros(height_, width_, img.type());
    b_channel = cv::Mat::zeros(height_, width_, img.type());

    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (bayer_pattern_ == "rggb") {
                if (y % 2 == 0 && x % 2 == 0) {
                    r_channel.at<uint16_t>(y, x) = img.at<uint16_t>(y, x);
                } else if (y % 2 == 1 && x % 2 == 1) {
                    b_channel.at<uint16_t>(y, x) = img.at<uint16_t>(y, x);
                }
            }
        }
    }
}

void BayerNoiseReduction::extract_channels_eigen(const hdr_isp::EigenImage32& img, hdr_isp::EigenImage32& r_channel, hdr_isp::EigenImage32& b_channel) {
    r_channel = hdr_isp::EigenImage32::Zero(height_, width_);
    b_channel = hdr_isp::EigenImage32::Zero(height_, width_);

    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (bayer_pattern_ == "rggb") {
                if (y % 2 == 0 && x % 2 == 0) {
                    r_channel.data()(y, x) = img.data()(y, x);
                } else if (y % 2 == 1 && x % 2 == 1) {
                    b_channel.data()(y, x) = img.data()(y, x);
                }
            }
        }
    }
}

void BayerNoiseReduction::combine_channels(const cv::Mat& r_channel, const cv::Mat& g_channel, const cv::Mat& b_channel, cv::Mat& output) {
    std::vector<cv::Mat> channels = {r_channel, g_channel, b_channel};
    cv::merge(channels, output);
}

void BayerNoiseReduction::combine_channels_eigen(const hdr_isp::EigenImage32& r_channel, const hdr_isp::EigenImage32& g_channel, const hdr_isp::EigenImage32& b_channel, hdr_isp::EigenImage32& output) {
    // For simplicity, return the green channel as output
    // In a full implementation, you'd create a proper 3-channel EigenImage
    output = g_channel;
}

cv::Mat BayerNoiseReduction::interpolate_green_channel(const cv::Mat& img) {
    cv::Mat g_channel = cv::Mat::zeros(height_, width_, img.type());
    
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (bayer_pattern_ == "rggb") {
                if (y % 2 == 0 && x % 2 == 1) {
                    // Green pixel at (y, x)
                    g_channel.at<uint16_t>(y, x) = img.at<uint16_t>(y, x);
                } else if (y % 2 == 1 && x % 2 == 0) {
                    // Green pixel at (y, x)
                    g_channel.at<uint16_t>(y, x) = img.at<uint16_t>(y, x);
                } else {
                    // Interpolate green value
                    int sum = 0;
                    int count = 0;
                    
                    // Check neighbors
                    if (y > 0) { sum += img.at<uint16_t>(y-1, x); count++; }
                    if (y < height_-1) { sum += img.at<uint16_t>(y+1, x); count++; }
                    if (x > 0) { sum += img.at<uint16_t>(y, x-1); count++; }
                    if (x < width_-1) { sum += img.at<uint16_t>(y, x+1); count++; }
                    
                    if (count > 0) {
                        g_channel.at<uint16_t>(y, x) = sum / count;
                    }
                }
            }
        }
    }
    
    return g_channel;
}

hdr_isp::EigenImage32 BayerNoiseReduction::interpolate_green_channel_eigen(const hdr_isp::EigenImage32& img) {
    hdr_isp::EigenImage32 g_channel = hdr_isp::EigenImage32::Zero(height_, width_);
    
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (bayer_pattern_ == "rggb") {
                if (y % 2 == 0 && x % 2 == 1) {
                    // Green pixel at (y, x)
                    g_channel.data()(y, x) = img.data()(y, x);
                } else if (y % 2 == 1 && x % 2 == 0) {
                    // Green pixel at (y, x)
                    g_channel.data()(y, x) = img.data()(y, x);
                } else {
                    // Interpolate green value
                    int sum = 0;
                    int count = 0;
                    
                    // Check neighbors
                    if (y > 0) { sum += img.data()(y-1, x); count++; }
                    if (y < height_-1) { sum += img.data()(y+1, x); count++; }
                    if (x > 0) { sum += img.data()(y, x-1); count++; }
                    if (x < width_-1) { sum += img.data()(y, x+1); count++; }
                    
                    if (count > 0) {
                        g_channel.data()(y, x) = sum / count;
                    }
                }
            }
        }
    }
    
    return g_channel;
}

cv::Mat BayerNoiseReduction::bilateral_filter(const cv::Mat& src, int d, double sigmaColor, double sigmaSpace) {
    cv::Mat filtered;
    cv::bilateralFilter(src, filtered, d, sigmaColor, sigmaSpace);
    return filtered;
}

hdr_isp::EigenImage32 BayerNoiseReduction::bilateral_filter_eigen(const hdr_isp::EigenImage32& src, int d, double sigmaColor, double sigmaSpace) {
    // Simplified bilateral filter using Eigen
    // In a full implementation, you'd implement proper bilateral filtering
    int rows = src.rows();
    int cols = src.cols();
    
    // Simple Gaussian blur as approximation
    hdr_isp::EigenImage32 filtered = hdr_isp::EigenImage32::Zero(rows, cols);
    
    // Simple 3x3 Gaussian kernel (scaled by 16 for integer arithmetic)
    Eigen::Matrix3i kernel;
    kernel << 1, 2, 1,
              2, 4, 2,
              1, 2, 1;
    
    // Apply convolution
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int sum = 0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += src.data()(i + ki, j + kj) * kernel(ki + 1, kj + 1);
                }
            }
            filtered.data()(i, j) = sum / 16; // Scale back down
        }
    }
    
    return filtered;
} 
#include "bayer_noise_reduction.hpp"
#include <chrono>
#include <iostream>

BayerNoiseReduction::BayerNoiseReduction(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_bnr)
    : raw_(img)
    , sensor_info_(sensor_info)
    , parm_bnr_(parm_bnr)
    , enable_(parm_bnr["is_enable"].as<bool>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
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
    std::cout << "Bayer Noise Reduction" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result = apply_bnr();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Bayer Noise Reduction execution time: " << duration.count() << " seconds" << std::endl;

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

    if (is_save_) {
        std::string filename = "out_frames/intermediate/Out_bnr_" + std::to_string(width_) + "x" + std::to_string(height_) + ".png";
        cv::imwrite(filename, output);
    }

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

void BayerNoiseReduction::combine_channels(const cv::Mat& r_channel, const cv::Mat& g_channel, const cv::Mat& b_channel, cv::Mat& output) {
    // Create a single channel output with the same type as input
    output = cv::Mat::zeros(height_, width_, raw_.type());
    
    // Combine channels back into Bayer pattern
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (bayer_pattern_ == "rggb") {
                if (y % 2 == 0 && x % 2 == 0) {
                    output.at<uint16_t>(y, x) = r_channel.at<uint16_t>(y, x);
                } else if ((y % 2 == 0 && x % 2 == 1) || (y % 2 == 1 && x % 2 == 0)) {
                    output.at<uint16_t>(y, x) = g_channel.at<uint16_t>(y, x);
                } else if (y % 2 == 1 && x % 2 == 1) {
                    output.at<uint16_t>(y, x) = b_channel.at<uint16_t>(y, x);
                }
            }
        }
    }
}

cv::Mat BayerNoiseReduction::interpolate_green_channel(const cv::Mat& img) {
    cv::Mat g_channel = cv::Mat::zeros(height_, width_, img.type());

    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (bayer_pattern_ == "rggb") {
                if ((y % 2 == 0 && x % 2 == 1) || (y % 2 == 1 && x % 2 == 0)) {
                    g_channel.at<uint16_t>(y, x) = img.at<uint16_t>(y, x);
                } else {
                    // Interpolate green value
                    int sum = 0;
                    int count = 0;
                    for (int dy = -1; dy <= 1; dy += 2) {
                        for (int dx = -1; dx <= 1; dx += 2) {
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < height_ && nx >= 0 && nx < width_) {
                                if ((ny % 2 == 0 && nx % 2 == 1) || (ny % 2 == 1 && nx % 2 == 0)) {
                                    sum += img.at<uint16_t>(ny, nx);
                                    count++;
                                }
                            }
                        }
                    }
                    g_channel.at<uint16_t>(y, x) = count > 0 ? sum / count : 0;
                }
            }
        }
    }

    return g_channel;
}

cv::Mat BayerNoiseReduction::bilateral_filter(const cv::Mat& src, int d, double sigmaColor, double sigmaSpace) {
    // Convert to float32 for bilateral filtering
    cv::Mat float_src;
    src.convertTo(float_src, CV_32F);
    
    // Apply bilateral filter
    cv::Mat float_dst;
    cv::bilateralFilter(float_src, float_dst, d, sigmaColor, sigmaSpace);
    
    // Convert back to original type
    cv::Mat dst;
    float_dst.convertTo(dst, src.type());
    
    return dst;
} 
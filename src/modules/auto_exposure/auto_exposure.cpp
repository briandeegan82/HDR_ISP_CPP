#include "auto_exposure.hpp"
#include <chrono>
#include <iostream>
#include <cmath>

AutoExposure::AutoExposure(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_ae)
    : img_(img)
    , sensor_info_(sensor_info)
    , param_ae_(parm_ae)
    , enable_(parm_ae["is_enable"].as<bool>())
    , is_debug_(parm_ae["is_debug"].as<bool>())
    , center_illuminance_(parm_ae["center_illuminance"].as<float>())
    , histogram_skewness_range_(parm_ae["histogram_skewness_range"].as<float>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
{
}

int AutoExposure::execute() {
    if (!enable_) {
        return 0;
    }

    return get_exposure_feedback();
}

int AutoExposure::get_exposure_feedback() {
    // Convert image to grayscale
    cv::Mat gray;
    cv::cvtColor(img_, gray, cv::COLOR_BGR2GRAY);

    // Calculate histogram
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // Calculate histogram skewness
    double skewness = get_luminance_histogram_skewness(hist);

    // Determine exposure based on skewness
    return determine_exposure(skewness);
}

int AutoExposure::determine_exposure(double skewness) {
    // Simple exposure adjustment based on skewness
    if (skewness > histogram_skewness_range_) {
        return -1;  // Decrease exposure
    } else if (skewness < -histogram_skewness_range_) {
        return 1;   // Increase exposure
    }
    return 0;       // Maintain current exposure
}

double AutoExposure::get_luminance_histogram_skewness(const cv::Mat& hist) {
    double mean = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    int total = 0;

    // Calculate mean
    for (int i = 0; i < hist.rows; i++) {
        mean += i * hist.at<float>(i);
        total += hist.at<float>(i);
    }
    mean /= total;

    // Calculate variance and skewness
    for (int i = 0; i < hist.rows; i++) {
        double diff = i - mean;
        variance += diff * diff * hist.at<float>(i);
        skewness += diff * diff * diff * hist.at<float>(i);
    }
    variance /= total;
    skewness /= total;

    // Normalize skewness
    if (variance > 0) {
        skewness /= std::pow(variance, 1.5);
    }

    return skewness;
}

std::tuple<cv::Mat, double> AutoExposure::get_greyscale_image(const cv::Mat& img) {
    // Convert to grayscale using luminance weights [0.299, 0.587, 0.144]
    cv::Mat grey_img;
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    
    // Apply luminance weights
    cv::Mat weighted_sum = 0.299 * channels[0] + 0.587 * channels[1] + 0.144 * channels[2];
    
    // Clip values to bit depth range
    cv::threshold(weighted_sum, grey_img, (1 << bit_depth_) - 1, (1 << bit_depth_) - 1, cv::THRESH_TRUNC);
    grey_img.convertTo(grey_img, CV_16UC1);
    
    // Calculate average luminance
    double avg_lum = cv::mean(grey_img)[0];
    
    return {grey_img, avg_lum};
} 
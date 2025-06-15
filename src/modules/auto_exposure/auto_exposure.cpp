#include "auto_exposure.hpp"
#include <chrono>
#include <iostream>
#include <cmath>

AutoExposure::AutoExposure(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_ae)
    : img_(img)
    , enable_(parm_ae["is_enable"].as<bool>())
    , is_debug_(parm_ae["is_debug"].as<bool>())
    , center_illuminance_(parm_ae["center_illuminance"].as<float>())
    , histogram_skewness_range_(parm_ae["histogram_skewness"].as<float>())
    , sensor_info_(sensor_info)
    , param_ae_(parm_ae)
    , bit_depth_(sensor_info["bit_depth"].as<int>()) {
}

int AutoExposure::get_exposure_feedback() {
    // Convert Image into 8-bit for AE Calculation
    img_ = img_ >> (bit_depth_ - 8);
    bit_depth_ = 8;

    // Calculate the exposure metric
    return determine_exposure();
}

int AutoExposure::determine_exposure() {
    // For Luminance Histograms, Image is first converted into greyscale image
    auto [grey_img, avg_lum] = get_greyscale_image(img_);
    if (is_debug_) {
        std::cout << "Average luminance is = " << avg_lum << std::endl;
    }

    // Histogram skewness Calculation for AE Stats
    double skewness = get_luminance_histogram_skewness(grey_img);

    // Get the ranges
    float upper_limit = histogram_skewness_range_;
    float lower_limit = -1.0f * upper_limit;

    if (is_debug_) {
        std::cout << "   - AE - Histogram Skewness Range = " << upper_limit << std::endl;
    }

    // See if skewness is within range
    if (skewness < lower_limit) {
        return -1;
    }
    else if (skewness > upper_limit) {
        return 1;
    }
    else {
        return 0;
    }
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

double AutoExposure::get_luminance_histogram_skewness(const cv::Mat& img) {
    // Convert to double and subtract central luminance
    cv::Mat img_float;
    img.convertTo(img_float, CV_64F);
    img_float -= center_illuminance_;

    // Calculate moments
    double m_2 = 0.0;  // variance
    double m_3 = 0.0;  // third moment
    
    int img_size = img_float.total();
    
    // Calculate m_2 and m_3
    for (int i = 0; i < img_float.rows; ++i) {
        for (int j = 0; j < img_float.cols; ++j) {
            double val = img_float.at<double>(i, j);
            m_2 += val * val;
            m_3 += val * val * val;
        }
    }
    
    m_2 /= img_size;
    m_3 /= img_size;

    // Calculate Fisher-Pearson coefficient of skewness
    double g_1 = std::sqrt(img_size * (img_size - 1.0)) / (img_size - 2.0);
    double skewness = (m_3 / std::pow(std::abs(m_2), 1.5)) * g_1;

    // Handle NaN values
    if (std::isnan(skewness)) {
        skewness = 0.0;
    }

    if (is_debug_) {
        std::cout << "   - AE - Histogram Skewness = " << skewness << std::endl;
    }

    return skewness;
}

int AutoExposure::execute() {
    if (!enable_) {
        return 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    int ae_feedback = get_exposure_feedback();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "  Execution time: " << elapsed.count() << "s" << std::endl;
    
    return ae_feedback;
} 
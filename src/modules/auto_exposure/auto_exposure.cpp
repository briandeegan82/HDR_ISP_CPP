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
    , histogram_skewness_range_(parm_ae["histogram_skewness"].as<float>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , use_eigen_(true) // Use Eigen by default
{
}

int AutoExposure::execute() {
    if (!enable_) {
        return 0;
    }

    return get_exposure_feedback();
}

int AutoExposure::get_exposure_feedback() {
    if (is_debug_) {
        std::cout << "AutoExposure::get_exposure_feedback - Input image info:" << std::endl;
        std::cout << "  Channels: " << img_.channels() << std::endl;
        std::cout << "  Size: " << img_.cols << "x" << img_.rows << std::endl;
        std::cout << "  Type: " << img_.type() << std::endl;
        
        // Print image statistics
        double min_val, max_val;
        cv::minMaxLoc(img_, &min_val, &max_val);
        cv::Scalar mean_val = cv::mean(img_);
        std::cout << "  Min: " << min_val << ", Mean: " << mean_val << ", Max: " << max_val << std::endl;
    }
    
    if (use_eigen_) {
        // Convert image to grayscale using Eigen
        auto [gray_eigen, avg_lum] = get_greyscale_image_eigen(img_);
        
        if (is_debug_) {
            std::cout << "AutoExposure::get_exposure_feedback - Grayscale conversion:" << std::endl;
            std::cout << "  Average luminance: " << avg_lum << std::endl;
            std::cout << "  Grayscale image size: " << gray_eigen.cols() << "x" << gray_eigen.rows() << std::endl;
        }
        
        // Calculate histogram using Eigen
        hdr_isp::EigenImage hist = hdr_isp::EigenImage::Zero(256, 1);
        for (int i = 0; i < gray_eigen.rows(); i++) {
            for (int j = 0; j < gray_eigen.cols(); j++) {
                int bin = static_cast<int>(gray_eigen.data()(i, j));
                if (bin >= 0 && bin < 256) {
                    hist.data()(bin, 0) += 1.0f;
                }
            }
        }
        
        // Calculate histogram skewness
        double skewness = get_luminance_histogram_skewness_eigen(hist);
        
        if (is_debug_) {
            std::cout << "AutoExposure::get_exposure_feedback - Histogram analysis:" << std::endl;
            std::cout << "  Skewness: " << skewness << std::endl;
            std::cout << "  Skewness range: " << histogram_skewness_range_ << std::endl;
        }
        
        // Determine exposure based on skewness
        int feedback = determine_exposure(skewness);
        
        if (is_debug_) {
            std::cout << "AutoExposure::get_exposure_feedback - Exposure feedback: " << feedback << std::endl;
        }
        
        return feedback;
    } else {
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

double AutoExposure::get_luminance_histogram_skewness_eigen(const hdr_isp::EigenImage& hist) {
    double mean = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double total = 0.0;

    // Calculate mean
    for (int i = 0; i < hist.rows(); i++) {
        mean += i * hist.data()(i, 0);
        total += hist.data()(i, 0);
    }
    mean /= total;

    // Calculate variance and skewness
    for (int i = 0; i < hist.rows(); i++) {
        double diff = i - mean;
        variance += diff * diff * hist.data()(i, 0);
        skewness += diff * diff * diff * hist.data()(i, 0);
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

std::tuple<hdr_isp::EigenImage, double> AutoExposure::get_greyscale_image_eigen(const cv::Mat& img) {
    // Check if input is 3-channel RGB image
    if (img.channels() == 3) {
        // Convert 3-channel RGB to grayscale using proper luminance weights
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        
        // Convert to Eigen
        hdr_isp::EigenImage eigen_img = hdr_isp::opencv_to_eigen(gray);
        
        // Clip values to bit depth range
        float max_val = static_cast<float>((1 << bit_depth_) - 1);
        eigen_img = eigen_img.cwiseMax(0.0f).cwiseMin(max_val);
        
        // Calculate average luminance
        double avg_lum = eigen_img.data().mean();
        
        return {eigen_img, avg_lum};
    } else if (img.channels() == 1) {
        // Single channel input - convert directly to Eigen
        hdr_isp::EigenImage eigen_img = hdr_isp::opencv_to_eigen(img);
        
        // Clip values to bit depth range
        float max_val = static_cast<float>((1 << bit_depth_) - 1);
        eigen_img = eigen_img.cwiseMax(0.0f).cwiseMin(max_val);
        
        // Calculate average luminance
        double avg_lum = eigen_img.data().mean();
        
        return {eigen_img, avg_lum};
    } else {
        throw std::runtime_error("AutoExposure::get_greyscale_image_eigen: Unsupported number of channels: " + std::to_string(img.channels()));
    }
} 
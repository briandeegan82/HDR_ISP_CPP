#include "lens_shading_correction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

LensShadingCorrection::LensShadingCorrection(const cv::Mat& img, const YAML::Node& platform,
                                            const YAML::Node& sensor_info, const YAML::Node& parm_lsc)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_lsc_(parm_lsc)
    , enable_(parm_lsc["is_enable"].as<bool>())
    , use_eigen_(true) // Use Eigen by default
{
}

cv::Mat LensShadingCorrection::apply_lsc_opencv() {
    // Simple lens shading correction using OpenCV
    // In a full implementation, you'd load actual shading correction data
    
    // Convert to float for processing
    cv::Mat float_img;
    img_.convertTo(float_img, CV_32F);
    
    // Create a simple radial shading correction
    cv::Mat shading_correction = cv::Mat::ones(img_.size(), CV_32F);
    cv::Point2f center(img_.cols / 2.0f, img_.rows / 2.0f);
    float max_radius = std::sqrt(center.x * center.x + center.y * center.y);
    
    for (int i = 0; i < img_.rows; i++) {
        for (int j = 0; j < img_.cols; j++) {
            float dx = j - center.x;
            float dy = i - center.y;
            float radius = std::sqrt(dx * dx + dy * dy);
            float correction_factor = 1.0f + 0.3f * (radius / max_radius) * (radius / max_radius);
            shading_correction.at<float>(i, j) = correction_factor;
        }
    }
    
    // Apply correction
    cv::Mat corrected = float_img.mul(shading_correction);
    
    // Convert back to original type
    cv::Mat result;
    corrected.convertTo(result, img_.type());
    
    return result;
}

hdr_isp::EigenImage32 LensShadingCorrection::apply_lsc_eigen() {
    // Convert to EigenImage32 for integer processing
    hdr_isp::EigenImage32 eigen_img = hdr_isp::EigenImage32::fromOpenCV(img_);
    
    // Create a simple radial shading correction using Eigen
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    hdr_isp::EigenImage32 shading_correction = hdr_isp::EigenImage32::Ones(rows, cols);
    
    // Use integer arithmetic with scaling for precision
    int center_x = cols / 2;
    int center_y = rows / 2;
    int max_radius_sq = center_x * center_x + center_y * center_y;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int dx = j - center_x;
            int dy = i - center_y;
            int radius_sq = dx * dx + dy * dy;
            // Scale by 1000 for precision, then scale back
            int correction_factor = 1000 + (300 * radius_sq) / max_radius_sq;
            shading_correction(i, j) = correction_factor;
        }
    }
    
    // Apply correction using integer arithmetic with scaling
    hdr_isp::EigenImage32 corrected = eigen_img.cwiseProduct(shading_correction);
    corrected = corrected / 1000; // Scale back down
    
    return corrected;
}

cv::Mat LensShadingCorrection::execute() {
    if (enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (use_eigen_) {
            hdr_isp::EigenImage32 eigen_result = apply_lsc_eigen();
            img_ = eigen_result.toOpenCV(img_.type());
        } else {
            img_ = apply_lsc_opencv();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  Lens Shading Correction execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }
    
    return img_;
} 
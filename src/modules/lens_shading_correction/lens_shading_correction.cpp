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

hdr_isp::EigenImageU32 LensShadingCorrection::apply_lsc_eigen() {
    // Convert to EigenImageU32 for integer processing
    hdr_isp::EigenImageU32 eigen_img = hdr_isp::EigenImageU32::fromOpenCV(img_);
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();

    // Create shading correction matrix
    hdr_isp::EigenImageU32 shading_correction = hdr_isp::EigenImageU32::Ones(rows, cols);
    
    // Apply shading correction (simplified - in real implementation, you'd load actual shading data)
    // For now, just apply a simple radial correction
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Calculate distance from center
            float center_x = cols / 2.0f;
            float center_y = rows / 2.0f;
            float distance = std::sqrt((i - center_y) * (i - center_y) + (j - center_x) * (j - center_x));
            float max_distance = std::sqrt(center_x * center_x + center_y * center_y);
            
            // Apply correction factor (simplified)
            float correction_factor = 1.0f + 0.3f * (distance / max_distance);
            shading_correction.data()(i, j) = static_cast<uint32_t>(correction_factor * 1000); // Scale for integer arithmetic
        }
    }
    
    // Apply correction
    hdr_isp::EigenImageU32 corrected = eigen_img.cwiseProduct(shading_correction);
    
    // Scale back down
    corrected = corrected / 1000;
    
    return corrected;
}

cv::Mat LensShadingCorrection::execute() {
    if (enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (use_eigen_) {
            hdr_isp::EigenImageU32 eigen_result = apply_lsc_eigen();
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
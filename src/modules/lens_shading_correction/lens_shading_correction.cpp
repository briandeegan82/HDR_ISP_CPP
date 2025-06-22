#include "lens_shading_correction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

LensShadingCorrection::LensShadingCorrection(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
                                            const YAML::Node& sensor_info, const YAML::Node& parm_lsc)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_lsc_(parm_lsc)
    , enable_(parm_lsc["is_enable"].as<bool>())
    , is_save_(parm_lsc["is_save"].as<bool>())
    , is_debug_(parm_lsc["is_debug"].as<bool>())
{
}

hdr_isp::EigenImageU32 LensShadingCorrection::apply_lsc_eigen(const hdr_isp::EigenImageU32& img) {
    int rows = img.rows();
    int cols = img.cols();
    
    // Debug output
    if (is_debug_) {
        std::cout << "LSC Eigen - Parameters:" << std::endl;
        std::cout << "  Image size: " << cols << "x" << rows << std::endl;
        std::cout << "  Center: (" << cols / 2.0f << ", " << rows / 2.0f << ")" << std::endl;
        
        // Print input image statistics
        uint32_t min_val = img.min();
        uint32_t max_val_input = img.max();
        float mean_val = img.mean();
        std::cout << "LSC Eigen - Input image - Mean: " << mean_val << ", Min: " << min_val << ", Max: " << max_val_input << std::endl;
    }

    // Create shading correction matrix
    hdr_isp::EigenImageU32 shading_correction = hdr_isp::EigenImageU32::Ones(rows, cols);
    
    // Calculate center point
    float center_x = cols / 2.0f;
    float center_y = rows / 2.0f;
    float max_distance = std::sqrt(center_x * center_x + center_y * center_y);
    
    // Apply shading correction (simplified - in real implementation, you'd load actual shading data)
    // For now, just apply a simple radial correction
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Calculate distance from center
            float dx = j - center_x;
            float dy = i - center_y;
            float distance = std::sqrt(dx * dx + dy * dy);
            
            // Apply correction factor (simplified radial shading model)
            // In a real implementation, this would be based on actual lens calibration data
            float correction_factor = 1.0f + 0.3f * (distance / max_distance) * (distance / max_distance);
            
            // Scale for integer arithmetic (multiply by 1000 to preserve precision)
            shading_correction.data()(i, j) = static_cast<uint32_t>(correction_factor * 1000);
        }
    }
    
    if (is_debug_) {
        // Print shading correction statistics
        uint32_t min_corr = shading_correction.min();
        uint32_t max_corr = shading_correction.max();
        float mean_corr = shading_correction.mean();
        std::cout << "LSC Eigen - Shading correction - Mean: " << mean_corr << ", Min: " << min_corr << ", Max: " << max_corr << std::endl;
    }
    
    // Apply correction using element-wise multiplication
    hdr_isp::EigenImageU32 corrected = img.cwiseProduct(shading_correction);
    
    // Scale back down (divide by 1000 to restore proper scaling)
    corrected = corrected / 1000;
    
    // For HDR ISP pipeline, clamp to prevent uint32 overflow
    uint32_t max_val = 4294967295U; // 2^32 - 1
    corrected = corrected.clip(0, max_val);
    
    if (is_debug_) {
        // Print output image statistics
        uint32_t min_val_out = corrected.min();
        uint32_t max_val_out = corrected.max();
        float mean_val_out = corrected.mean();
        std::cout << "LSC Eigen - Output image - Mean: " << mean_val_out << ", Min: " << min_val_out << ", Max: " << max_val_out << std::endl;
    }
    
    return corrected;
}

void LensShadingCorrection::save(const std::string& filename_tag) {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/" + filename_tag + 
                                 std::to_string(img_.cols()) + "x" + std::to_string(img_.rows()) + ".png";
        // Convert to OpenCV for saving
        cv::Mat save_img = img_.toOpenCV(CV_32S);
        cv::imwrite(output_path, save_img);
    }
}

hdr_isp::EigenImageU32 LensShadingCorrection::execute() {
    if (!enable_) {
        return img_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Apply lens shading correction using Eigen
    img_ = apply_lsc_eigen(img_);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "  Lens Shading Correction execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }

    // Save intermediate results if enabled
    if (is_save_) {
        save("lens_shading_correction_");
    }

    return img_;
} 
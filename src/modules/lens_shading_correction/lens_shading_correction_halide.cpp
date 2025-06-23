#include "lens_shading_correction_halide.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

LensShadingCorrectionHalide::LensShadingCorrectionHalide(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
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

Halide::Func LensShadingCorrectionHalide::create_shading_correction(Halide::Buffer<uint32_t> input) {
    Halide::Var x, y;
    
    // Get image dimensions
    int width = input.width();
    int height = input.height();
    
    // Calculate center point
    float center_x = width / 2.0f;
    float center_y = height / 2.0f;
    float max_distance = std::sqrt(center_x * center_x + center_y * center_y);
    
    // Create shading correction using Halide's vectorized operations
    Halide::Func shading_correction;
    
    // Convert coordinates to float for precise calculations
    Halide::Expr x_float = Halide::cast<float>(x);
    Halide::Expr y_float = Halide::cast<float>(y);
    
    // Calculate distance from center
    Halide::Expr dx = x_float - center_x;
    Halide::Expr dy = y_float - center_y;
    Halide::Expr distance = Halide::sqrt(dx * dx + dy * dy);
    
    // Apply radial shading correction model
    // Using the same model as the original: 1.0 + 0.3 * (distance/max_distance)^2
    Halide::Expr normalized_distance = distance / max_distance;
    Halide::Expr correction_factor = 1.0f + 0.3f * normalized_distance * normalized_distance;
    
    // Scale for integer arithmetic (multiply by 1000 to preserve precision)
    shading_correction(x, y) = Halide::cast<uint32_t>(correction_factor * 1000.0f);
    
    // Schedule for optimal performance
    shading_correction.vectorize(x, 8).parallel(y);
    
    return shading_correction;
}

Halide::Func LensShadingCorrectionHalide::apply_radial_correction(Halide::Buffer<uint32_t> input, Halide::Func shading_correction) {
    Halide::Var x, y;
    
    // Apply correction using element-wise multiplication
    Halide::Func corrected;
    corrected(x, y) = input(x, y) * shading_correction(x, y);
    
    // Scale back down (divide by 1000 to restore proper scaling)
    corrected(x, y) = corrected(x, y) / 1000;
    
    // Clamp to prevent uint32 overflow
    uint32_t max_val = 4294967295U; // 2^32 - 1
    corrected(x, y) = Halide::clamp(corrected(x, y), 0, max_val);
    
    // Schedule for optimal performance
    corrected.vectorize(x, 8).parallel(y);
    
    return corrected;
}

Halide::Buffer<uint32_t> LensShadingCorrectionHalide::apply_lsc_halide(const Halide::Buffer<uint32_t>& input) {
    int width = input.width();
    int height = input.height();
    
    // Debug output
    if (is_debug_) {
        std::cout << "LSC Halide - Parameters:" << std::endl;
        std::cout << "  Image size: " << width << "x" << height << std::endl;
        std::cout << "  Center: (" << width / 2.0f << ", " << height / 2.0f << ")" << std::endl;
        
        // Print input image statistics
        uint32_t min_val = input.min();
        uint32_t max_val_input = input.max();
        std::cout << "LSC Halide - Input image - Min: " << min_val << ", Max: " << max_val_input << std::endl;
    }
    
    // Create shading correction function
    Halide::Func shading_correction = create_shading_correction(input);
    
    // Apply radial correction
    Halide::Func corrected = apply_radial_correction(input, shading_correction);
    
    // Realize the result
    Halide::Buffer<uint32_t> output = corrected.realize({width, height});
    
    if (is_debug_) {
        // Print output image statistics
        uint32_t min_val_out = output.min();
        uint32_t max_val_out = output.max();
        std::cout << "LSC Halide - Output image - Min: " << min_val_out << ", Max: " << max_val_out << std::endl;
    }
    
    return output;
}

void LensShadingCorrectionHalide::save(const std::string& filename_tag) {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/" + filename_tag + 
                                 std::to_string(img_.cols()) + "x" + std::to_string(img_.rows()) + ".png";
        // Convert to OpenCV for saving
        cv::Mat save_img = img_.toOpenCV(CV_32S);
        cv::imwrite(output_path, save_img);
    }
}

hdr_isp::EigenImageU32 LensShadingCorrectionHalide::execute() {
    if (!enable_) {
        return img_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Convert Eigen image to Halide buffer
    Halide::Buffer<uint32_t> halide_input = Halide::Buffer<uint32_t>(
        reinterpret_cast<uint32_t*>(img_.data()), 
        img_.cols(), img_.rows()
    );
    
    // Apply lens shading correction using Halide
    Halide::Buffer<uint32_t> halide_output = apply_lsc_halide(halide_input);
    
    // Convert back to Eigen image
    hdr_isp::EigenImageU32 corrected_img(
        reinterpret_cast<uint32_t*>(halide_output.data()),
        img_.rows(), img_.cols()
    );
    
    img_ = corrected_img;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "  Lens Shading Correction Halide execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }

    // Save intermediate results if enabled
    if (is_save_) {
        save("lens_shading_correction_halide_");
    }

    return img_;
} 
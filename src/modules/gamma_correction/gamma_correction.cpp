#include "gamma_correction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

GammaCorrection::GammaCorrection(const hdr_isp::EigenImage3C& img, const YAML::Node& platform,
                               const YAML::Node& sensor_info, const YAML::Node& parm_gmm)
    : img_(img)
    , enable_(parm_gmm["is_enable"] ? parm_gmm["is_enable"].as<bool>() : false)
    , sensor_info_(sensor_info)
    , output_bit_depth_(sensor_info["output_bit_depth"] ? sensor_info["output_bit_depth"].as<int>() : 8)
    , parm_gmm_(parm_gmm)
    , is_save_(parm_gmm["is_save"] ? parm_gmm["is_save"].as<bool>() : false)
    , is_debug_(parm_gmm["is_debug"] ? parm_gmm["is_debug"].as<bool>() : false)
    , platform_(platform)
{
    std::cout << "Gamma - Constructor started" << std::endl;
    std::cout << "Gamma - Input image size: " << img.rows() << "x" << img.cols() << std::endl;
    std::cout << "Gamma - Enable: " << (enable_ ? "true" : "false") << std::endl;
    std::cout << "Gamma - Output bit depth: " << output_bit_depth_ << std::endl;
    std::cout << "Gamma - Is save: " << (is_save_ ? "true" : "false") << std::endl;
    std::cout << "Gamma - Is debug: " << (is_debug_ ? "true" : "false") << std::endl;
    
    // Validate output_bit_depth
    if (output_bit_depth_ <= 0 || output_bit_depth_ > 16) {
        std::cout << "Gamma - Constructor - WARNING: Invalid output_bit_depth: " << output_bit_depth_ << ", using default 8" << std::endl;
        output_bit_depth_ = 8;
    }
    
    // Print input image statistics (calculate from individual channels)
    float min_val = std::min({img_.r().min(), img_.g().min(), img_.b().min()});
    float max_val = std::max({img_.r().max(), img_.g().max(), img_.b().max()});
    float mean_val = (img_.r().mean() + img_.g().mean() + img_.b().mean()) / 3.0f;
    std::cout << "Gamma - Input image - Mean: " << mean_val << ", Min: " << min_val << ", Max: " << max_val << std::endl;
    
    std::cout << "Gamma - Constructor completed" << std::endl;
}

std::vector<uint32_t> GammaCorrection::generate_gamma_lut(int bit_depth) {
    std::cout << "Gamma - generate_gamma_lut() started with bit_depth: " << bit_depth << std::endl;
    
    // Safety check for reasonable bit depth
    if (bit_depth <= 0 || bit_depth > 16) {
        std::cout << "Gamma - generate_gamma_lut() - WARNING: Unreasonable bit_depth: " << bit_depth << ", limiting to 16" << std::endl;
        bit_depth = std::min(16, std::max(1, bit_depth));
    }
    
    int max_val = (1 << bit_depth) - 1;
    std::vector<uint32_t> lut(max_val + 1);
    
    std::cout << "Gamma - generate_gamma_lut() - max_val: " << max_val << ", lut size: " << lut.size() << std::endl;
    
    // Safety check for reasonable LUT size
    if (lut.size() > 65536) { // 2^16
        std::cout << "Gamma - generate_gamma_lut() - ERROR: LUT size too large: " << lut.size() << ", aborting" << std::endl;
        throw std::runtime_error("Gamma LUT size too large: " + std::to_string(lut.size()));
    }
    
    std::cout << "Gamma - generate_gamma_lut() - Starting LUT generation..." << std::endl;
    
    for (int i = 0; i <= max_val; ++i) {
        lut[i] = static_cast<uint32_t>(std::round(max_val * std::pow(static_cast<double>(i) / max_val, 1.0 / 2.2)));
        
        // Progress indicator for large LUTs
        if (lut.size() > 1000 && i % 1000 == 0) {
            std::cout << "Gamma - generate_gamma_lut() - Progress: " << i << "/" << max_val << std::endl;
        }
    }
    
    std::cout << "Gamma - generate_gamma_lut() - LUT generated successfully" << std::endl;
    std::cout << "Gamma - generate_gamma_lut() - Sample LUT values: [0]=" << lut[0] << ", [max/2]=" << lut[max_val/2] << ", [max]=" << lut[max_val] << std::endl;
    
    return lut;
}

hdr_isp::EigenImage3C GammaCorrection::apply_gamma() {
    std::cout << "Gamma - apply_gamma() started" << std::endl;
    
    auto lut = generate_gamma_lut(output_bit_depth_);
    
    // Create a copy of the input image
    hdr_isp::EigenImage3C result = img_.clone();
    int rows = result.rows();
    int cols = result.cols();
    
    std::cout << "Gamma - apply_gamma() - Processing image: " << rows << "x" << cols << std::endl;
    
    // Print sample input values
    std::cout << "Gamma - apply_gamma() - Sample input values (top-left 3x3):" << std::endl;
    for (int i = 0; i < std::min(3, rows); ++i) {
        for (int j = 0; j < std::min(3, cols); ++j) {
            std::cout << "  [" << i << "," << j << "] R=" << result.r().data()(i, j) 
                      << " G=" << result.g().data()(i, j) 
                      << " B=" << result.b().data()(i, j) << std::endl;
        }
    }
    
    // Apply LUT to each channel
    std::cout << "Gamma - apply_gamma() - Starting pixel processing..." << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Progress indicator for large images
            if (rows * cols > 1000000 && (i * cols + j) % 100000 == 0) {
                std::cout << "Gamma - apply_gamma() - Progress: " << (i * cols + j) << "/" << (rows * cols) << std::endl;
            }
            
            // Apply to R channel
            int r_val = static_cast<int>(result.r().data()(i, j));
            if (r_val >= 0 && r_val < static_cast<int>(lut.size()))
                result.r().data()(i, j) = static_cast<float>(lut[r_val]);
            
            // Apply to G channel
            int g_val = static_cast<int>(result.g().data()(i, j));
            if (g_val >= 0 && g_val < static_cast<int>(lut.size()))
                result.g().data()(i, j) = static_cast<float>(lut[g_val]);
            
            // Apply to B channel
            int b_val = static_cast<int>(result.b().data()(i, j));
            if (b_val >= 0 && b_val < static_cast<int>(lut.size()))
                result.b().data()(i, j) = static_cast<float>(lut[b_val]);
        }
    }
    std::cout << "Gamma - apply_gamma() - Pixel processing completed" << std::endl;
    
    // Print sample output values
    std::cout << "Gamma - apply_gamma() - Sample output values (top-left 3x3):" << std::endl;
    for (int i = 0; i < std::min(3, rows); ++i) {
        for (int j = 0; j < std::min(3, cols); ++j) {
            std::cout << "  [" << i << "," << j << "] R=" << result.r().data()(i, j) 
                      << " G=" << result.g().data()(i, j) 
                      << " B=" << result.b().data()(i, j) << std::endl;
        }
    }
    
    // Print output image statistics (calculate from individual channels)
    float min_val_out = std::min({result.r().min(), result.g().min(), result.b().min()});
    float max_val_out = std::max({result.r().max(), result.g().max(), result.b().max()});
    float mean_val_out = (result.r().mean() + result.g().mean() + result.b().mean()) / 3.0f;
    std::cout << "Gamma - apply_gamma() - Output image - Mean: " << mean_val_out << ", Min: " << min_val_out << ", Max: " << max_val_out << std::endl;
    
    std::cout << "Gamma - apply_gamma() completed" << std::endl;
    return result;
}

void GammaCorrection::save() {
    std::cout << "Gamma - save() started" << std::endl;
    
    if (is_save_) {
        fs::path output_dir = "out_frames";
        fs::create_directories(output_dir);
        
        // Try to get filename from platform config - handle both "in_file" and "filename" keys
        std::string in_file;
        if (platform_["in_file"]) {
            in_file = platform_["in_file"].as<std::string>();
        } else if (platform_["filename"]) {
            in_file = platform_["filename"].as<std::string>();
        } else {
            in_file = "unknown";
        }
        
        std::string bayer_pattern = sensor_info_["bayer_pattern"].as<std::string>();
        fs::path output_path = output_dir / ("Out_gamma_correction_" + in_file + "_" + bayer_pattern + ".png");
        
        std::cout << "Gamma - save() - Saving to: " << output_path.string() << std::endl;
        
        // Convert to OpenCV for saving
        cv::Mat opencv_img = img_.toOpenCV(CV_32FC3);
        cv::imwrite(output_path.string(), opencv_img);
        
        std::cout << "Gamma - save() - File saved successfully" << std::endl;
    } else {
        std::cout << "Gamma - save() - Save disabled, skipping" << std::endl;
    }
}

hdr_isp::EigenImage3C GammaCorrection::execute() {
    std::cout << "Gamma - execute() started" << std::endl;
    std::cout << "Gamma - execute() - enable_: " << (enable_ ? "true" : "false") << std::endl;
    
    if (enable_) {
        std::cout << "Gamma - execute() - Applying gamma correction..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        img_ = apply_gamma();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Gamma - execute() - Gamma correction applied successfully" << std::endl;
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    } else {
        std::cout << "Gamma - execute() - Gamma correction disabled, returning input" << std::endl;
    }

    std::cout << "Gamma - execute() completed" << std::endl;
    return img_;
} 
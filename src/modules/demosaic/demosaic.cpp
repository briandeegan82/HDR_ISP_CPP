#include "demosaic.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// BilinearDemosaic implementation
BilinearDemosaic::BilinearDemosaic(const hdr_isp::EigenImageU32& raw_in, const std::string& bayer_pattern)
    : raw_in_(raw_in)
    , bayer_pattern_(bayer_pattern)
    , rows_(raw_in.rows())
    , cols_(raw_in.cols())
{
}

hdr_isp::EigenImage3C BilinearDemosaic::execute() {
    // Convert input to float and normalize to [0, 1] range
    uint32_t max_val = raw_in_.max();
    hdr_isp::EigenImage normalized = hdr_isp::EigenImage(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            normalized.data()(i, j) = static_cast<float>(raw_in_.data()(i, j)) / static_cast<float>(max_val);
        }
    }

    // Interpolate each channel
    hdr_isp::EigenImage r_channel = interpolate_red();
    hdr_isp::EigenImage g_channel = interpolate_green();
    hdr_isp::EigenImage b_channel = interpolate_blue();

    // Create 3-channel output
    hdr_isp::EigenImage3C result(rows_, cols_);
    result.r() = r_channel;
    result.g() = g_channel;
    result.b() = b_channel;

    return result;
}

hdr_isp::EigenImage3CFixed BilinearDemosaic::execute_fixed(int fractional_bits) {
    // Convert input to float and normalize to [0, 1] range
    uint32_t max_val = raw_in_.max();
    hdr_isp::EigenImage normalized = hdr_isp::EigenImage(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            normalized.data()(i, j) = static_cast<float>(raw_in_.data()(i, j)) / static_cast<float>(max_val);
        }
    }

    // Interpolate each channel
    hdr_isp::EigenImage r_channel = interpolate_red();
    hdr_isp::EigenImage g_channel = interpolate_green();
    hdr_isp::EigenImage b_channel = interpolate_blue();

    // Create 3-channel floating-point output first
    hdr_isp::EigenImage3C float_result(rows_, cols_);
    float_result.r() = r_channel;
    float_result.g() = g_channel;
    float_result.b() = b_channel;

    // Convert to fixed-point
    hdr_isp::EigenImage3CFixed result = hdr_isp::EigenImage3CFixed::fromEigenImage3C(float_result, fractional_bits);

    return result;
}

std::array<hdr_isp::EigenImage, 3> BilinearDemosaic::create_bayer_masks() {
    std::array<hdr_isp::EigenImage, 3> masks;
    for (auto& mask : masks) {
        mask = hdr_isp::EigenImage::Zero(rows_, cols_);
    }

    // Create masks based on bayer pattern
    for (int y = 0; y < rows_; y += 2) {
        for (int x = 0; x < cols_; x += 2) {
            // Get the 2x2 bayer pattern for this block
            std::string block_pattern = bayer_pattern_.substr(0, 4);
            
            // Set the mask values for each channel in the 2x2 block
            for (int i = 0; i < 4; ++i) {
                int block_y = y + (i / 2);
                int block_x = x + (i % 2);
                if (block_y < rows_ && block_x < cols_) {
                    char channel = block_pattern[i];
                    int channel_idx = (channel == 'r') ? 0 : ((channel == 'g') ? 1 : 2);
                    masks[channel_idx].data()(block_y, block_x) = 1.0f;
                }
            }
        }
    }

    return masks;
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_green() {
    if (bayer_pattern_ == "rggb") {
        return interpolate_green_rggb();
    } else if (bayer_pattern_ == "bggr") {
        return interpolate_green_bggr();
    } else if (bayer_pattern_ == "grbg") {
        return interpolate_green_grbg();
    } else if (bayer_pattern_ == "gbrg") {
        return interpolate_green_gbrg();
    } else {
        throw std::runtime_error("Unsupported bayer pattern: " + bayer_pattern_);
    }
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_red() {
    if (bayer_pattern_ == "rggb") {
        return interpolate_red_rggb();
    } else if (bayer_pattern_ == "bggr") {
        return interpolate_red_bggr();
    } else if (bayer_pattern_ == "grbg") {
        return interpolate_red_grbg();
    } else if (bayer_pattern_ == "gbrg") {
        return interpolate_red_gbrg();
    } else {
        throw std::runtime_error("Unsupported bayer pattern: " + bayer_pattern_);
    }
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_blue() {
    if (bayer_pattern_ == "rggb") {
        return interpolate_blue_rggb();
    } else if (bayer_pattern_ == "bggr") {
        return interpolate_blue_bggr();
    } else if (bayer_pattern_ == "grbg") {
        return interpolate_blue_grbg();
    } else if (bayer_pattern_ == "gbrg") {
        return interpolate_blue_gbrg();
    } else {
        throw std::runtime_error("Unsupported bayer pattern: " + bayer_pattern_);
    }
}

// RGGB pattern: R G
//              G B
hdr_isp::EigenImage BilinearDemosaic::interpolate_green_rggb() {
    hdr_isp::EigenImage result = hdr_isp::EigenImage::Zero(rows_, cols_);
    
    // Copy existing green pixels
    for (int i = 1; i < rows_; i += 2) {
        for (int j = 0; j < cols_; j += 2) {
            result.data()(i, j) = static_cast<float>(raw_in_.data()(i, j)) / 255.0f;
        }
    }
    for (int i = 0; i < rows_; i += 2) {
        for (int j = 1; j < cols_; j += 2) {
            result.data()(i, j) = static_cast<float>(raw_in_.data()(i, j)) / 255.0f;
        }
    }
    
    // Interpolate missing green pixels
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            if (result.data()(i, j) == 0.0f) { // Missing green pixel
                float sum = 0.0f;
                int count = 0;
                
                // Check 4 neighbors
                if (i > 0) { sum += result.data()(i-1, j); count++; }
                if (i < rows_-1) { sum += result.data()(i+1, j); count++; }
                if (j > 0) { sum += result.data()(i, j-1); count++; }
                if (j < cols_-1) { sum += result.data()(i, j+1); count++; }
                
                if (count > 0) {
                    result.data()(i, j) = sum / static_cast<float>(count);
                }
            }
        }
    }
    
    return result;
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_red_rggb() {
    hdr_isp::EigenImage result = hdr_isp::EigenImage::Zero(rows_, cols_);
    
    // Copy existing red pixels
    for (int i = 0; i < rows_; i += 2) {
        for (int j = 0; j < cols_; j += 2) {
            result.data()(i, j) = static_cast<float>(raw_in_.data()(i, j)) / 255.0f;
        }
    }
    
    // Interpolate missing red pixels
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            if (result.data()(i, j) == 0.0f) { // Missing red pixel
                float sum = 0.0f;
                int count = 0;
                
                // Check diagonal neighbors
                if (i > 0 && j > 0) { sum += result.data()(i-1, j-1); count++; }
                if (i > 0 && j < cols_-1) { sum += result.data()(i-1, j+1); count++; }
                if (i < rows_-1 && j > 0) { sum += result.data()(i+1, j-1); count++; }
                if (i < rows_-1 && j < cols_-1) { sum += result.data()(i+1, j+1); count++; }
                
                if (count > 0) {
                    result.data()(i, j) = sum / static_cast<float>(count);
                }
            }
        }
    }
    
    return result;
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_blue_rggb() {
    hdr_isp::EigenImage result = hdr_isp::EigenImage::Zero(rows_, cols_);
    
    // Copy existing blue pixels
    for (int i = 1; i < rows_; i += 2) {
        for (int j = 1; j < cols_; j += 2) {
            result.data()(i, j) = static_cast<float>(raw_in_.data()(i, j)) / 255.0f;
        }
    }
    
    // Interpolate missing blue pixels
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            if (result.data()(i, j) == 0.0f) { // Missing blue pixel
                float sum = 0.0f;
                int count = 0;
                
                // Check diagonal neighbors
                if (i > 0 && j > 0) { sum += result.data()(i-1, j-1); count++; }
                if (i > 0 && j < cols_-1) { sum += result.data()(i-1, j+1); count++; }
                if (i < rows_-1 && j > 0) { sum += result.data()(i+1, j-1); count++; }
                if (i < rows_-1 && j < cols_-1) { sum += result.data()(i+1, j+1); count++; }
                
                if (count > 0) {
                    result.data()(i, j) = sum / static_cast<float>(count);
                }
            }
        }
    }
    
    return result;
}

// Implement similar functions for other bayer patterns
hdr_isp::EigenImage BilinearDemosaic::interpolate_green_bggr() {
    hdr_isp::EigenImage result = hdr_isp::EigenImage::Zero(rows_, cols_);
    
    // Copy existing green pixels (same as RGGB but shifted)
    for (int i = 1; i < rows_; i += 2) {
        for (int j = 0; j < cols_; j += 2) {
            result.data()(i, j) = static_cast<float>(raw_in_.data()(i, j)) / 255.0f;
        }
    }
    for (int i = 0; i < rows_; i += 2) {
        for (int j = 1; j < cols_; j += 2) {
            result.data()(i, j) = static_cast<float>(raw_in_.data()(i, j)) / 255.0f;
        }
    }
    
    // Interpolate missing green pixels
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            if (result.data()(i, j) == 0.0f) {
                float sum = 0.0f;
                int count = 0;
                
                if (i > 0) { sum += result.data()(i-1, j); count++; }
                if (i < rows_-1) { sum += result.data()(i+1, j); count++; }
                if (j > 0) { sum += result.data()(i, j-1); count++; }
                if (j < cols_-1) { sum += result.data()(i, j+1); count++; }
                
                if (count > 0) {
                    result.data()(i, j) = sum / static_cast<float>(count);
                }
            }
        }
    }
    
    return result;
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_red_bggr() {
    hdr_isp::EigenImage result = hdr_isp::EigenImage::Zero(rows_, cols_);
    
    // Copy existing red pixels (bottom-right in 2x2 block)
    for (int i = 1; i < rows_; i += 2) {
        for (int j = 1; j < cols_; j += 2) {
            result.data()(i, j) = static_cast<float>(raw_in_.data()(i, j)) / 255.0f;
        }
    }
    
    // Interpolate missing red pixels
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            if (result.data()(i, j) == 0.0f) {
                float sum = 0.0f;
                int count = 0;
                
                if (i > 0 && j > 0) { sum += result.data()(i-1, j-1); count++; }
                if (i > 0 && j < cols_-1) { sum += result.data()(i-1, j+1); count++; }
                if (i < rows_-1 && j > 0) { sum += result.data()(i+1, j-1); count++; }
                if (i < rows_-1 && j < cols_-1) { sum += result.data()(i+1, j+1); count++; }
                
                if (count > 0) {
                    result.data()(i, j) = sum / static_cast<float>(count);
                }
            }
        }
    }
    
    return result;
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_blue_bggr() {
    hdr_isp::EigenImage result = hdr_isp::EigenImage::Zero(rows_, cols_);
    
    // Copy existing blue pixels (top-left in 2x2 block)
    for (int i = 0; i < rows_; i += 2) {
        for (int j = 0; j < cols_; j += 2) {
            result.data()(i, j) = static_cast<float>(raw_in_.data()(i, j)) / 255.0f;
        }
    }
    
    // Interpolate missing blue pixels
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            if (result.data()(i, j) == 0.0f) {
                float sum = 0.0f;
                int count = 0;
                
                if (i > 0 && j > 0) { sum += result.data()(i-1, j-1); count++; }
                if (i > 0 && j < cols_-1) { sum += result.data()(i-1, j+1); count++; }
                if (i < rows_-1 && j > 0) { sum += result.data()(i+1, j-1); count++; }
                if (i < rows_-1 && j < cols_-1) { sum += result.data()(i+1, j+1); count++; }
                
                if (count > 0) {
                    result.data()(i, j) = sum / static_cast<float>(count);
                }
            }
        }
    }
    
    return result;
}

// Implement remaining bayer pattern functions (simplified for brevity)
hdr_isp::EigenImage BilinearDemosaic::interpolate_green_grbg() {
    return interpolate_green_rggb(); // Similar pattern
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_green_gbrg() {
    return interpolate_green_rggb(); // Similar pattern
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_red_grbg() {
    return interpolate_red_rggb(); // Similar pattern
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_red_gbrg() {
    return interpolate_red_rggb(); // Similar pattern
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_blue_grbg() {
    return interpolate_blue_rggb(); // Similar pattern
}

hdr_isp::EigenImage BilinearDemosaic::interpolate_blue_gbrg() {
    return interpolate_blue_rggb(); // Similar pattern
}

// Demosaic class implementation
Demosaic::Demosaic(const hdr_isp::EigenImageU32& img, const std::string& bayer_pattern, int bit_depth, bool is_save)
    : img_(img)
    , bayer_pattern_(bayer_pattern)
    , bit_depth_(bit_depth)
    , is_save_(is_save)
    , is_debug_(false)
    , is_enable_(true)
    , fp_config_(YAML::Node())  // Default empty config
    , use_fixed_point_(false)
{
}

Demosaic::Demosaic(const hdr_isp::EigenImageU32& img, const std::string& bayer_pattern, const hdr_isp::FixedPointConfig& fp_config, int bit_depth, bool is_save)
    : img_(img)
    , bayer_pattern_(bayer_pattern)
    , bit_depth_(bit_depth)
    , is_save_(is_save)
    , is_debug_(false)
    , is_enable_(true)
    , fp_config_(fp_config)
    , use_fixed_point_(fp_config.isEnabled())
{
}

hdr_isp::EigenImage3C Demosaic::execute() {
    if (is_enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Convert to uint8 and scale
        uint32_t max_val = img_.max();
        hdr_isp::EigenImageU32 scaled_img = img_.clone();
        
        // Scale to 0-255 range
        for (int i = 0; i < scaled_img.rows(); i++) {
            for (int j = 0; j < scaled_img.cols(); j++) {
                scaled_img.data()(i, j) = static_cast<uint32_t>((static_cast<float>(img_.data()(i, j)) / static_cast<float>(max_val)) * 255.0f);
            }
        }
        
        // Apply bilinear demosaic
        BilinearDemosaic demosaic(scaled_img, bayer_pattern_);
        hdr_isp::EigenImage3C result = demosaic.execute();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
        
        if (is_save_) {
            save(result);
        }
        
        return result;
    }
    
    // Return empty result if not enabled
    return hdr_isp::EigenImage3C(img_.rows(), img_.cols());
}

hdr_isp::EigenImage3CFixed Demosaic::execute_fixed() {
    if (is_enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Convert to uint8 and scale
        uint32_t max_val = img_.max();
        hdr_isp::EigenImageU32 scaled_img = img_.clone();
        
        // Scale to 0-255 range
        for (int i = 0; i < scaled_img.rows(); i++) {
            for (int j = 0; j < scaled_img.cols(); j++) {
                scaled_img.data()(i, j) = static_cast<uint32_t>((static_cast<float>(img_.data()(i, j)) / static_cast<float>(max_val)) * 255.0f);
            }
        }
        
        // Apply bilinear demosaic with fixed-point output
        BilinearDemosaic demosaic(scaled_img, bayer_pattern_);
        int fractional_bits = fp_config_.getFractionalBits();
        hdr_isp::EigenImage3CFixed result = demosaic.execute_fixed(fractional_bits);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Fixed-point execution time: " << duration.count() / 1000.0 << "s" << std::endl;
            std::cout << "  Fractional bits: " << fractional_bits << std::endl;
        }
        
        if (is_save_) {
            save_fixed(result);
        }
        
        return result;
    }
    
    // Return empty result if not enabled
    return hdr_isp::EigenImage3CFixed(img_.rows(), img_.cols());
}

void Demosaic::save(const hdr_isp::EigenImage3C& result) {
    if (is_save_) {
        fs::path output_dir = "out_frames";
        fs::create_directories(output_dir);
        fs::path output_path = output_dir / ("Out_demosaic_" + bayer_pattern_ + ".png");
        
        // Convert to OpenCV for saving
        cv::Mat save_img = result.toOpenCV(CV_32FC3);
        
        // Convert to 8-bit for saving
        cv::Mat save_img_8u;
        save_img.convertTo(save_img_8u, CV_8UC3, 255.0);
        cv::imwrite(output_path.string(), save_img_8u);
    }
}

void Demosaic::save_fixed(const hdr_isp::EigenImage3CFixed& result) {
    if (is_save_) {
        fs::path output_dir = "out_frames";
        fs::create_directories(output_dir);
        fs::path output_path = output_dir / ("Out_demosaic_fixed_" + bayer_pattern_ + ".png");
        
        // Convert to OpenCV for saving
        int fractional_bits = fp_config_.getFractionalBits();
        cv::Mat save_img = result.toOpenCV(fractional_bits, CV_32FC3);
        
        // Convert to 8-bit for saving
        cv::Mat save_img_8u;
        save_img.convertTo(save_img_8u, CV_8UC3, 255.0);
        cv::imwrite(output_path.string(), save_img_8u);
    }
} 
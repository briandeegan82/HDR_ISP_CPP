#include "bayer_noise_reduction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

BayerNoiseReduction::BayerNoiseReduction(const hdr_isp::EigenImageU32& img, const YAML::Node& sensor_info, const YAML::Node& parm_bnr)
    : img_(img)
    , sensor_info_(sensor_info)
    , parm_bnr_(parm_bnr)
    , width_(img.cols())
    , height_(img.rows())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , bayer_pattern_(sensor_info["bayer_pattern"].as<std::string>())
    , is_debug_(false)  // Default to false
    , is_save_(false) { // Default to false
    
    // Try to get is_debug from parameters, but don't fail if it's not there
    try {
        if (parm_bnr["is_debug"].IsDefined()) {
            is_debug_ = parm_bnr["is_debug"].as<bool>();
        }
    } catch (const std::exception& e) {
        std::cout << "BNR - Warning: is_debug parameter not found, using default (false)" << std::endl;
        is_debug_ = false;
    }
    
    // Try to get is_save from parameters, but don't fail if it's not there
    try {
        if (parm_bnr["is_save"].IsDefined()) {
            is_save_ = parm_bnr["is_save"].as<bool>();
        }
    } catch (const std::exception& e) {
        std::cout << "BNR - Warning: is_save parameter not found, using default (false)" << std::endl;
        is_save_ = false;
    }
    
    std::cout << "BNR - Constructor completed. Debug mode: " << (is_debug_ ? "enabled" : "disabled") << ", Save mode: " << (is_save_ ? "enabled" : "disabled") << std::endl;
}

hdr_isp::EigenImageU32 BayerNoiseReduction::execute() {
    std::cout << "BNR - execute() started" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        hdr_isp::EigenImageU32 result = apply_bnr_eigen(img_);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "BNR - Execution time: " << elapsed.count() << "s" << std::endl;
        
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "BNR - Exception in execute(): " << e.what() << std::endl;
        return img_; // Return original image if processing fails
    }
    catch (...) {
        std::cerr << "BNR - Unknown exception in execute()" << std::endl;
        return img_; // Return original image if processing fails
    }
}

hdr_isp::EigenImageU32 BayerNoiseReduction::apply_bnr_eigen(const hdr_isp::EigenImageU32& img) {
    std::cout << "BNR Eigen - Starting apply_bnr_eigen..." << std::endl;
    
    // Debug output
    if (is_debug_) {
        std::cout << "BNR Eigen - Parameters:" << std::endl;
        std::cout << "  Image size: " << width_ << "x" << height_ << std::endl;
        std::cout << "  Bayer pattern: " << bayer_pattern_ << std::endl;
        std::cout << "  Bit depth: " << bit_depth_ << std::endl;
        
        // Print input image statistics
        uint32_t min_val = img.min();
        uint32_t max_val_input = img.max();
        float mean_val = img.mean();
        std::cout << "BNR Eigen - Input image - Mean: " << mean_val << ", Min: " << min_val << ", Max: " << max_val_input << std::endl;
    }

    try {
        std::cout << "BNR Eigen - Extracting channels..." << std::endl;
        hdr_isp::EigenImageU32 r_channel, b_channel;
        extract_channels_eigen(img, r_channel, b_channel);
        std::cout << "BNR Eigen - Channels extracted successfully" << std::endl;
        
        std::cout << "BNR Eigen - Interpolating green channel..." << std::endl;
        hdr_isp::EigenImageU32 g_channel = interpolate_green_channel_eigen(img);
        std::cout << "BNR Eigen - Green channel interpolated successfully" << std::endl;

        // For now, skip the bilateral filtering to isolate the issue
        // Just return the green channel as output
        std::cout << "BNR Eigen - Skipping bilateral filtering for now..." << std::endl;
        
        hdr_isp::EigenImageU32 output = g_channel;
        
        // For HDR ISP pipeline, clamp to prevent uint32 overflow
        uint32_t max_val = 4294967295U; // 2^32 - 1
        output = output.clip(0, max_val);
        
        if (is_debug_) {
            // Print output image statistics
            uint32_t min_val_out = output.min();
            uint32_t max_val_out = output.max();
            float mean_val_out = output.mean();
            std::cout << "BNR Eigen - Output image - Mean: " << mean_val_out << ", Min: " << min_val_out << ", Max: " << max_val_out << std::endl;
        }

        std::cout << "BNR Eigen - apply_bnr_eigen completed successfully" << std::endl;
        return output;
    }
    catch (const std::exception& e) {
        std::cerr << "BNR Eigen - Exception during processing: " << e.what() << std::endl;
        // Return original image if processing fails
        return img;
    }
    catch (...) {
        std::cerr << "BNR Eigen - Unknown exception during processing" << std::endl;
        // Return original image if processing fails
        return img;
    }
}

void BayerNoiseReduction::extract_channels_eigen(const hdr_isp::EigenImageU32& img, hdr_isp::EigenImageU32& r_channel, hdr_isp::EigenImageU32& b_channel) {
    std::cout << "BNR Eigen - extract_channels_eigen started" << std::endl;
    
    r_channel = hdr_isp::EigenImageU32::Zero(height_, width_);
    b_channel = hdr_isp::EigenImageU32::Zero(height_, width_);

    std::cout << "BNR Eigen - Channels initialized, processing pixels..." << std::endl;
    
    int r_count = 0, b_count = 0;
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (bayer_pattern_ == "rggb") {
                if (y % 2 == 0 && x % 2 == 0) {
                    r_channel.data()(y, x) = img.data()(y, x);
                    r_count++;
                } else if (y % 2 == 1 && x % 2 == 1) {
                    b_channel.data()(y, x) = img.data()(y, x);
                    b_count++;
                }
            }
        }
    }
    
    std::cout << "BNR Eigen - extract_channels_eigen completed. R pixels: " << r_count << ", B pixels: " << b_count << std::endl;
}

void BayerNoiseReduction::combine_channels_eigen(const hdr_isp::EigenImageU32& r_channel, const hdr_isp::EigenImageU32& g_channel, const hdr_isp::EigenImageU32& b_channel, hdr_isp::EigenImageU32& output) {
    // For simplicity, return the green channel as output
    // In a full implementation, you'd create a proper 3-channel EigenImage
    output = g_channel;
}

hdr_isp::EigenImageU32 BayerNoiseReduction::interpolate_green_channel_eigen(const hdr_isp::EigenImageU32& img) {
    std::cout << "BNR Eigen - interpolate_green_channel_eigen started" << std::endl;
    
    hdr_isp::EigenImageU32 g_channel = hdr_isp::EigenImageU32::Zero(height_, width_);
    
    int green_pixels = 0, interpolated_pixels = 0;
    
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (bayer_pattern_ == "rggb") {
                if (y % 2 == 0 && x % 2 == 1) {
                    // Green pixel at (y, x)
                    g_channel.data()(y, x) = img.data()(y, x);
                    green_pixels++;
                } else if (y % 2 == 1 && x % 2 == 0) {
                    // Green pixel at (y, x)
                    g_channel.data()(y, x) = img.data()(y, x);
                    green_pixels++;
                } else {
                    // Interpolate green value
                    uint64_t sum = 0; // Use uint64_t to prevent overflow
                    int count = 0;
                    
                    // Check neighbors with bounds checking
                    if (y > 0) { sum += static_cast<uint64_t>(img.data()(y-1, x)); count++; }
                    if (y < height_-1) { sum += static_cast<uint64_t>(img.data()(y+1, x)); count++; }
                    if (x > 0) { sum += static_cast<uint64_t>(img.data()(y, x-1)); count++; }
                    if (x < width_-1) { sum += static_cast<uint64_t>(img.data()(y, x+1)); count++; }
                    
                    if (count > 0) {
                        g_channel.data()(y, x) = static_cast<uint32_t>(sum / count);
                        interpolated_pixels++;
                    }
                }
            }
        }
    }
    
    std::cout << "BNR Eigen - interpolate_green_channel_eigen completed. Green pixels: " << green_pixels << ", Interpolated pixels: " << interpolated_pixels << std::endl;
    
    return g_channel;
}

hdr_isp::EigenImageU32 BayerNoiseReduction::bilateral_filter_eigen(const hdr_isp::EigenImageU32& src, int d, double sigmaColor, double sigmaSpace) {
    // For now, just return the input image to avoid potential crashes
    // This will be implemented later once we identify the crash source
    std::cout << "BNR Eigen - bilateral_filter_eigen: returning input image (filtering disabled)" << std::endl;
    return src;
}

void BayerNoiseReduction::save(const std::string& filename_tag) {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/" + filename_tag + 
                                 std::to_string(width_) + "x" + std::to_string(height_) + ".png";
        // Convert to OpenCV for saving
        cv::Mat save_img = img_.toOpenCV(CV_32S);
        cv::imwrite(output_path, save_img);
    }
} 
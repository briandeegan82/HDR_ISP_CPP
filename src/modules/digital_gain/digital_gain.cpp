#include "digital_gain.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

DigitalGain::DigitalGain(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
                        const YAML::Node& sensor_info, const YAML::Node& parm_dga)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_dga_(parm_dga)
    , is_save_(false)
    , is_debug_(false)
    , is_auto_(false)
    , current_gain_(0)
    , ae_feedback_(0.0f)
{
    try {
        is_save_ = parm_dga["is_save"].as<bool>();
        is_debug_ = parm_dga["is_debug"].as<bool>();
        is_auto_ = parm_dga["is_auto"].as<bool>();
        current_gain_ = parm_dga["current_gain"].as<int>();
        
        // Try to convert ae_feedback safely
        try {
            ae_feedback_ = parm_dga["ae_feedback"].as<float>();
        } catch (...) {
            // If float conversion fails, try int conversion
            ae_feedback_ = static_cast<float>(parm_dga["ae_feedback"].as<int>());
        }
        
        // Convert YAML sequence to vector of floats
        const YAML::Node& gains_node = parm_dga["gain_array"];
        for (const auto& gain : gains_node) {
            gains_array_.push_back(gain.as<float>());
        }
    }
    catch (const YAML::Exception& e) {
        std::cerr << "YAML Exception in DigitalGain constructor: " << e.what() << std::endl;
        throw;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in DigitalGain constructor: " << e.what() << std::endl;
        throw;
    }
}

hdr_isp::EigenImageU32 DigitalGain::apply_digital_gain_eigen(const hdr_isp::EigenImageU32& img) {
    // Get output bit depth from sensor info
    int output_bit_depth = sensor_info_["output_bit_depth"].as<int>();
    
    // Debug output
    if (is_debug_) {
        std::cout << "DigitalGain Eigen - Parameters:" << std::endl;
        std::cout << "  Output bit depth: " << output_bit_depth << std::endl;
        std::cout << "  Is auto: " << (is_auto_ ? "true" : "false") << std::endl;
        std::cout << "  Current gain index: " << current_gain_ << std::endl;
        std::cout << "  AE feedback: " << ae_feedback_ << std::endl;
        std::cout << "  Gains array size: " << gains_array_.size() << std::endl;
        std::cout << "  Image size: " << img.cols() << "x" << img.rows() << std::endl;
        
        // Print input image statistics
        uint32_t min_val = img.min();
        uint32_t max_val_input = img.max();
        float mean_val = img.mean();
        std::cout << "DigitalGain Eigen - Input image - Mean: " << mean_val << ", Min: " << min_val << ", Max: " << max_val_input << std::endl;
    }

    // Apply gains based on AE feedback
    if (is_auto_) {
        if (ae_feedback_ < 0) {
            // Increase gain for underexposed image
            current_gain_ = std::min(static_cast<int>(gains_array_.size() - 1), current_gain_ + 1);
        }
        else if (ae_feedback_ > 0) {
            // Decrease gain for overexposed image
            current_gain_ = std::max(0, current_gain_ - 1);
        }
    }

    // Apply the selected gain with proper precision
    float gain = gains_array_[current_gain_];
    
    if (is_debug_) {
        std::cout << "DigitalGain Eigen - Applied Gain = " << gain << std::endl;
    }
    
    // Convert to float for precise multiplication, then back to uint32
    Eigen::MatrixXf float_img = img.data().cast<float>();
    float_img = float_img * gain;
    
    // Convert back to uint32 - preserve full dynamic range for HDR processing
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> uint32_img = 
        float_img.cast<uint32_t>();

    // Create EigenImageU32 from the processed matrix
    hdr_isp::EigenImageU32 result(uint32_img);
    
    // For HDR ISP pipeline, only clamp to prevent overflow, not to output bit depth
    // This preserves the full dynamic range for subsequent HDR processing stages
    
    // Only clamp to prevent uint32 overflow (2^32 - 1)
    uint32_t max_val = 4294967295U; // 2^32 - 1
    result = result.clip(0, max_val);
    
    if (is_debug_) {
        // Print output image statistics
        uint32_t min_val_out = result.min();
        uint32_t max_val_out = result.max();
        float mean_val_out = result.mean();
        std::cout << "DigitalGain Eigen - Output image - Mean: " << mean_val_out << ", Min: " << min_val_out << ", Max: " << max_val_out << std::endl;
    }

    return result;
}

void DigitalGain::save(const std::string& filename_tag) {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/" + filename_tag + 
                                 std::to_string(img_.cols()) + "x" + std::to_string(img_.rows()) + ".png";
        // Convert to OpenCV for saving
        cv::Mat save_img = img_.toOpenCV(CV_32S);
        cv::imwrite(output_path, save_img);
    }
}

std::pair<hdr_isp::EigenImageU32, float> DigitalGain::execute() {
    float applied_gain = gains_array_[current_gain_];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Apply digital gain using Eigen
    img_ = apply_digital_gain_eigen(img_);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }

    // Save intermediate results if enabled
    if (is_save_) {
        save("digital_gain_");
    }

    return {img_, applied_gain};
} 
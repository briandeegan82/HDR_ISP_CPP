#include "digital_gain.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

DigitalGain::DigitalGain(const cv::Mat& img, const YAML::Node& platform,
                        const YAML::Node& sensor_info, const YAML::Node& parm_dga)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_dga_(parm_dga)
    , is_save_(parm_dga["is_save"].as<bool>())
    , is_debug_(parm_dga["is_debug"].as<bool>())
    , is_auto_(parm_dga["is_auto"].as<bool>())
    , current_gain_(parm_dga["current_gain"].as<int>())
    , ae_feedback_(parm_dga["ae_feedback"].as<float>())
{
    // Convert YAML sequence to vector of floats
    const YAML::Node& gains_node = parm_dga["gain_array"];
    for (const auto& gain : gains_node) {
        gains_array_.push_back(gain.as<float>());
    }
}

cv::Mat DigitalGain::apply_digital_gain() {
    // Get desired parameters from config
    int bpp = sensor_info_["output_bit_depth"].as<int>();

    // Convert to float image
    cv::Mat float_img;
    img_.convertTo(float_img, CV_32F);

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

    // Apply the selected gain
    float_img *= gains_array_[current_gain_];

    if (is_debug_) {
        std::cout << "   - DG  - Applied Gain = " << gains_array_[current_gain_] << std::endl;
    }

    // Convert back to original bit depth with clipping
    cv::Mat result;
    float_img.convertTo(result, CV_32S);
    cv::threshold(result, result, 0, (1 << bpp) - 1, cv::THRESH_TRUNC);
    result.convertTo(result, img_.type());

    return result;
}

void DigitalGain::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_digital_gain_" + 
                                 std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + ".png";
        cv::imwrite(output_path, img_);
    }
}

std::pair<cv::Mat, int> DigitalGain::execute() {
    auto start = std::chrono::high_resolution_clock::now();
    img_ = apply_digital_gain();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }

    save();
    return std::make_pair(img_, current_gain_);
} 
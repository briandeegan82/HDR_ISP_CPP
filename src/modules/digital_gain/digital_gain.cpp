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
    , is_save_(false)
    , is_debug_(false)
    , is_auto_(false)
    , current_gain_(0)
    , ae_feedback_(0.0f)
    , use_eigen_(true) // Use Eigen by default
{
    std::cout << "DigitalGain constructor - starting parameter extraction..." << std::endl;
    
    try {
        std::cout << "  Extracting is_save..." << std::endl;
        is_save_ = parm_dga["is_save"].as<bool>();
        std::cout << "  is_save = " << is_save_ << std::endl;
        
        std::cout << "  Extracting is_debug..." << std::endl;
        is_debug_ = parm_dga["is_debug"].as<bool>();
        std::cout << "  is_debug = " << is_debug_ << std::endl;
        
        std::cout << "  Extracting is_auto..." << std::endl;
        is_auto_ = parm_dga["is_auto"].as<bool>();
        std::cout << "  is_auto = " << is_auto_ << std::endl;
        
        std::cout << "  Extracting current_gain..." << std::endl;
        current_gain_ = parm_dga["current_gain"].as<int>();
        std::cout << "  current_gain = " << current_gain_ << std::endl;
        
        std::cout << "  Extracting ae_feedback..." << std::endl;
        // Try to convert ae_feedback safely
        try {
            ae_feedback_ = parm_dga["ae_feedback"].as<float>();
        } catch (...) {
            // If float conversion fails, try int conversion
            ae_feedback_ = static_cast<float>(parm_dga["ae_feedback"].as<int>());
        }
        std::cout << "  ae_feedback = " << ae_feedback_ << std::endl;
        
        std::cout << "  Extracting gain_array..." << std::endl;
        // Convert YAML sequence to vector of floats
        const YAML::Node& gains_node = parm_dga["gain_array"];
        for (const auto& gain : gains_node) {
            gains_array_.push_back(gain.as<float>());
        }
        std::cout << "  gain_array size = " << gains_array_.size() << std::endl;
        
        std::cout << "DigitalGain constructor - parameter extraction completed successfully" << std::endl;
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

cv::Mat DigitalGain::apply_digital_gain_opencv() {
    std::cout << "DigitalGain::apply_digital_gain_opencv() - starting..." << std::endl;
    
    try {
        // Get desired parameters from config
        std::cout << "  Getting output_bit_depth from sensor_info..." << std::endl;
        int bpp = sensor_info_["output_bit_depth"].as<int>();
        std::cout << "  output_bit_depth = " << bpp << std::endl;

        // Convert to float image
        std::cout << "  Converting image to float..." << std::endl;
        cv::Mat float_img;
        img_.convertTo(float_img, CV_32F);
        std::cout << "  Image converted to float successfully" << std::endl;

        // Apply gains based on AE feedback
        if (is_auto_) {
            std::cout << "  Auto mode enabled, checking AE feedback..." << std::endl;
            if (ae_feedback_ < 0) {
                // Increase gain for underexposed image
                current_gain_ = std::min(static_cast<int>(gains_array_.size() - 1), current_gain_ + 1);
                std::cout << "  AE feedback < 0, increasing gain to " << current_gain_ << std::endl;
            }
            else if (ae_feedback_ > 0) {
                // Decrease gain for overexposed image
                current_gain_ = std::max(0, current_gain_ - 1);
                std::cout << "  AE feedback > 0, decreasing gain to " << current_gain_ << std::endl;
            }
        }

        // Apply the selected gain
        std::cout << "  Applying gain " << gains_array_[current_gain_] << std::endl;
        float_img *= gains_array_[current_gain_];

        if (is_debug_) {
            std::cout << "   - DG  - Applied Gain = " << gains_array_[current_gain_] << std::endl;
        }

        // Convert back to original bit depth with clipping
        std::cout << "  Converting back to original bit depth..." << std::endl;
        cv::Mat result;
        float_img.convertTo(result, CV_32S);
        std::cout << "  Converted to CV_32S successfully" << std::endl;
        
        // Use cv::min and cv::max instead of cv::threshold for signed integers
        int max_val = (1 << bpp) - 1;
        cv::Mat clipped_result;
        cv::min(result, max_val, clipped_result);
        cv::max(clipped_result, 0, result);
        std::cout << "  Applied clipping successfully" << std::endl;
        
        result.convertTo(result, img_.type());
        std::cout << "  Converted to original type successfully" << std::endl;

        std::cout << "DigitalGain::apply_digital_gain_opencv() - completed successfully" << std::endl;
        return result;
    }
    catch (const YAML::Exception& e) {
        std::cerr << "YAML Exception in apply_digital_gain_opencv: " << e.what() << std::endl;
        throw;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception in apply_digital_gain_opencv: " << e.what() << std::endl;
        throw;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in apply_digital_gain_opencv: " << e.what() << std::endl;
        throw;
    }
}

hdr_isp::EigenImage DigitalGain::apply_digital_gain_eigen() {
    std::cout << "DigitalGain::apply_digital_gain_eigen() - starting..." << std::endl;
    
    try {
        // Get desired parameters from config
        std::cout << "  Getting output_bit_depth from sensor_info..." << std::endl;
        int bpp = sensor_info_["output_bit_depth"].as<int>();
        std::cout << "  output_bit_depth = " << bpp << std::endl;

        // Convert to Eigen
        std::cout << "  Converting image to Eigen..." << std::endl;
        hdr_isp::EigenImage eigen_img = hdr_isp::opencv_to_eigen(img_);
        std::cout << "  Image converted to Eigen successfully" << std::endl;

        // Apply gains based on AE feedback
        if (is_auto_) {
            std::cout << "  Auto mode enabled, checking AE feedback..." << std::endl;
            if (ae_feedback_ < 0) {
                // Increase gain for underexposed image
                current_gain_ = std::min(static_cast<int>(gains_array_.size() - 1), current_gain_ + 1);
                std::cout << "  AE feedback < 0, increasing gain to " << current_gain_ << std::endl;
            }
            else if (ae_feedback_ > 0) {
                // Decrease gain for overexposed image
                current_gain_ = std::max(0, current_gain_ - 1);
                std::cout << "  AE feedback > 0, decreasing gain to " << current_gain_ << std::endl;
            }
        }

        // Apply the selected gain
        std::cout << "  Applying gain " << gains_array_[current_gain_] << std::endl;
        eigen_img = eigen_img * gains_array_[current_gain_];

        if (is_debug_) {
            std::cout << "   - DG  - Applied Gain = " << gains_array_[current_gain_] << std::endl;
        }

        // Apply clipping
        std::cout << "  Applying clipping..." << std::endl;
        int max_val = (1 << bpp) - 1;
        eigen_img = eigen_img.cwiseMax(0.0f).cwiseMin(static_cast<float>(max_val));
        std::cout << "  Applied clipping successfully" << std::endl;

        std::cout << "DigitalGain::apply_digital_gain_eigen() - completed successfully" << std::endl;
        return eigen_img;
    }
    catch (const YAML::Exception& e) {
        std::cerr << "YAML Exception in apply_digital_gain_eigen: " << e.what() << std::endl;
        throw;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in apply_digital_gain_eigen: " << e.what() << std::endl;
        throw;
    }
}

void DigitalGain::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_digital_gain_" + 
                                 std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + ".png";
        cv::imwrite(output_path, img_);
    }
}

std::pair<cv::Mat, int> DigitalGain::execute() {
    std::cout << "Digital Gain Execute" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    if (use_eigen_) {
        hdr_isp::EigenImage eigen_result = apply_digital_gain_eigen();
        img_ = hdr_isp::eigen_to_opencv(eigen_result);
    } else {
        img_ = apply_digital_gain_opencv();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }

    save();
    return std::make_pair(img_, current_gain_);
} 
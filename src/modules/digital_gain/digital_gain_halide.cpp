#include "digital_gain_halide.hpp"
#include "../../common/halide_utils.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

DigitalGainHalide::DigitalGainHalide(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
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
        std::cerr << "YAML Exception in DigitalGainHalide constructor: " << e.what() << std::endl;
        throw;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in DigitalGainHalide constructor: " << e.what() << std::endl;
        throw;
    }
}

Halide::Func DigitalGainHalide::vectorized_multiply(const Halide::Buffer<uint32_t>& input, float gain) {
    // Define Halide variables for coordinates
    Halide::Var x, y;
    
    // Create input function from buffer
    Halide::Func input_func("input_func");
    input_func(x, y) = input(x, y);
    
    // Create the gain multiplication function
    Halide::Func gain_func("gain_func");
    
    // Convert to float for precise multiplication, then back to uint32
    gain_func(x, y) = Halide::cast<uint32_t>(
        Halide::cast<float>(input_func(x, y)) * gain
    );
    
    // Schedule for optimal performance
    gain_func.vectorize(x, 8);  // Vectorize by 8 pixels
    gain_func.parallel(y);      // Parallelize across rows
    
    return gain_func;
}

Halide::Buffer<uint32_t> DigitalGainHalide::apply_digital_gain_halide(const Halide::Buffer<uint32_t>& input) {
    // Get output bit depth from sensor info
    int output_bit_depth = sensor_info_["output_bit_depth"].as<int>();
    
    // Debug output
    if (is_debug_) {
        std::cout << "DigitalGain Halide - Parameters:" << std::endl;
        std::cout << "  Output bit depth: " << output_bit_depth << std::endl;
        std::cout << "  Is auto: " << (is_auto_ ? "true" : "false") << std::endl;
        std::cout << "  Current gain index: " << current_gain_ << std::endl;
        std::cout << "  AE feedback: " << ae_feedback_ << std::endl;
        std::cout << "  Gains array size: " << gains_array_.size() << std::endl;
        std::cout << "  Image size: " << input.width() << "x" << input.height() << std::endl;
        
        // Print input image statistics
        uint32_t min_val = input.min();
        uint32_t max_val_input = input.max();
        float mean_val = 0;
        for (int y = 0; y < input.height(); y++) {
            for (int x = 0; x < input.width(); x++) {
                mean_val += input(x, y);
            }
        }
        mean_val /= (input.width() * input.height());
        std::cout << "DigitalGain Halide - Input image - Mean: " << mean_val << ", Min: " << min_val << ", Max: " << max_val_input << std::endl;
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

    // Apply the selected gain
    float gain = gains_array_[current_gain_];
    
    if (is_debug_) {
        std::cout << "DigitalGain Halide - Applied Gain = " << gain << std::endl;
    }
    
    // Create Halide function for gain multiplication
    Halide::Func gain_func = vectorized_multiply(input, gain);
    
    // Create output buffer
    Halide::Buffer<uint32_t> output(input.width(), input.height());
    
    // Realize the function
    gain_func.realize(output);
    
    // For HDR ISP pipeline, only clamp to prevent overflow, not to output bit depth
    // This preserves the full dynamic range for subsequent HDR processing stages
    
    // Only clamp to prevent uint32 overflow (2^32 - 1)
    uint32_t max_val = 4294967295U; // 2^32 - 1
    
    // Apply clamping using Halide
    Halide::Func clamp_func("clamp_func");
    Halide::Var x, y;
    clamp_func(x, y) = Halide::min(output(x, y), max_val);
    
    Halide::Buffer<uint32_t> clamped_output(input.width(), input.height());
    clamp_func.realize(clamped_output);
    
    if (is_debug_) {
        // Print output image statistics
        uint32_t min_val_out = clamped_output.min();
        uint32_t max_val_out = clamped_output.max();
        float mean_val_out = 0;
        for (int y = 0; y < clamped_output.height(); y++) {
            for (int x = 0; x < clamped_output.width(); x++) {
                mean_val_out += clamped_output(x, y);
            }
        }
        mean_val_out /= (clamped_output.width() * clamped_output.height());
        std::cout << "DigitalGain Halide - Output image - Mean: " << mean_val_out << ", Min: " << min_val_out << ", Max: " << max_val_out << std::endl;
    }

    return clamped_output;
}

void DigitalGainHalide::save(const std::string& filename_tag) {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/" + filename_tag + 
                                 std::to_string(img_.cols()) + "x" + std::to_string(img_.rows()) + ".png";
        // Convert to OpenCV for saving
        cv::Mat save_img = img_.toOpenCV(CV_32S);
        cv::imwrite(output_path, save_img);
    }
}

std::pair<hdr_isp::EigenImageU32, float> DigitalGainHalide::execute() {
    float applied_gain = gains_array_[current_gain_];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Convert Eigen image to Halide buffer
    Halide::Buffer<uint32_t> halide_input = hdr_isp::eigenToHalide(img_);
    
    // Apply digital gain using Halide
    Halide::Buffer<uint32_t> halide_output = apply_digital_gain_halide(halide_input);
    
    // Convert back to Eigen image
    img_ = hdr_isp::halideToEigen(halide_output);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }

    // Save intermediate results if enabled
    if (is_save_) {
        save("digital_gain_halide_");
    }

    return {img_, applied_gain};
} 
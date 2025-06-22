#include "pwc_generation.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <stdexcept>

namespace fs = std::filesystem;

PiecewiseCurve::PiecewiseCurve(const hdr_isp::EigenImageU32& img, const YAML::Node& platform, const YAML::Node& sensor_info, const YAML::Node& parm_cmpd)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_cmpd_(parm_cmpd)
    , enable_(parm_cmpd["is_enable"].as<bool>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , companded_pin_(parm_cmpd["companded_pin"].as<std::vector<int>>())
    , companded_pout_(parm_cmpd["companded_pout"].as<std::vector<int>>())
    , is_save_(parm_cmpd["is_save"].as<bool>())
    , is_debug_(parm_cmpd["is_debug"].as<bool>())
{
}

std::vector<double> PiecewiseCurve::generate_decompanding_lut(
    const std::vector<int>& companded_pin,
    const std::vector<int>& companded_pout,
    int max_input_value
) {
    // Ensure the input and output lists are of the same length
    if (companded_pin.size() != companded_pout.size()) {
        throw std::runtime_error("companded_pin and companded_pout must have the same length");
    }

    // Initialize the LUT with zeros
    std::vector<double> lut(max_input_value + 1, 0.0);

    // Generate the LUT by interpolating between the knee points
    for (size_t i = 0; i < companded_pin.size() - 1; ++i) {
        int start_in = companded_pin[i];
        int end_in = companded_pin[i + 1];
        int start_out = companded_pout[i];
        int end_out = companded_pout[i + 1];

        // Linear interpolation between the knee points
        for (int x = start_in; x <= end_in; ++x) {
            double t = static_cast<double>(x - start_in) / (end_in - start_in);
            lut[x] = start_out + t * (end_out - start_out);
        }
    }

    // Handle values beyond the last knee point (extend the last segment)
    int last_in = companded_pin.back();
    int last_out = companded_pout.back();
    std::fill(lut.begin() + last_in, lut.end(), last_out);

    return lut;
}

void PiecewiseCurve::save(const std::string& filename_tag) {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/" + filename_tag + 
            std::to_string(img_.cols()) + "x" + std::to_string(img_.rows()) + "_" +
            std::to_string(bit_depth_) + "bits_" +
            sensor_info_["bayer_pattern"].as<std::string>() + ".png";
        // Convert to OpenCV for saving
        cv::Mat save_img = img_.toOpenCV(CV_32S);
        cv::imwrite(output_path, save_img);
    }
}

hdr_isp::EigenImageU32 PiecewiseCurve::execute() {
    if (!enable_) {
        return img_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Apply decompanding using Eigen
    img_ = apply_decompanding_eigen(img_);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }

    // Save intermediate results if enabled
    if (is_save_) {
        save("decompanding_");
    }

    return img_;
}

hdr_isp::EigenImageU32 PiecewiseCurve::apply_decompanding_eigen(const hdr_isp::EigenImageU32& img) {
    // Generate decompanding LUT
    std::vector<double> lut = generate_decompanding_lut(
        companded_pin_,
        companded_pout_,
        companded_pin_.back()
    );
    
    int rows = img.rows();
    int cols = img.cols();
    
    // Debug output
    if (is_debug_) {
        std::cout << "PWC Eigen - Parameters:" << std::endl;
        std::cout << "  Bit depth: " << bit_depth_ << std::endl;
        std::cout << "  LUT size: " << lut.size() << std::endl;
        std::cout << "  Image size: " << cols << "x" << rows << std::endl;
        
        // Print input image statistics
        uint32_t min_val = img.min();
        uint32_t max_val_input = img.max();
        float mean_val = img.mean();
        std::cout << "PWC Eigen - Input image - Mean: " << mean_val << ", Min: " << min_val << ", Max: " << max_val_input << std::endl;
    }
    
    hdr_isp::EigenImageU32 result = img.clone();
    
    // Apply LUT to each pixel using Eigen
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uint32_t pixel_value = result.data()(i, j);
            if (pixel_value < lut.size()) {
                result.data()(i, j) = static_cast<uint32_t>(lut[pixel_value]);
            } else {
                result.data()(i, j) = 0;
            }
        }
    }
    
    // Subtract pedestal and clip negative values
    uint32_t pedestal = static_cast<uint32_t>(parm_cmpd_["pedestal"].as<double>());
    result = result - pedestal;
    
    // For early modules, clamp to 2^32 when bit depth is 32, otherwise use the configured bit depth
    uint32_t max_val;
    if (bit_depth_ == 32) {
        max_val = 4294967295U; // 2^32 - 1
    } else {
        max_val = (1U << bit_depth_) - 1;
    }
    
    result = result.clip(0, max_val);
    
    if (is_debug_) {
        // Print output image statistics
        uint32_t min_val_out = result.min();
        uint32_t max_val_out = result.max();
        float mean_val_out = result.mean();
        std::cout << "PWC Eigen - Output image - Mean: " << mean_val_out << ", Min: " << min_val_out << ", Max: " << max_val_out << std::endl;
    }
    
    return result;
} 
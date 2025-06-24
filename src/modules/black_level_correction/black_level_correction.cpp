#include "black_level_correction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>

BlackLevelCorrection::BlackLevelCorrection(const hdr_isp::EigenImageU32& img, const YAML::Node& sensor_info, const YAML::Node& parm_blc)
    : raw_(img.clone())
    , sensor_info_(sensor_info)
    , parm_blc_(parm_blc)
    , enable_(parm_blc["is_enable"].as<bool>())
    , is_linearize_(parm_blc["is_linear"].as<bool>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , bayer_pattern_(sensor_info["bayer_pattern"].as<std::string>())
    , is_save_(parm_blc["is_save"].as<bool>())
{
}

hdr_isp::EigenImageU32 BlackLevelCorrection::execute() {
    if (!enable_) {
        return raw_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Use EigenImageU32 for integer processing
    hdr_isp::EigenImageU32 result = apply_blc_parameters_eigen(raw_);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Black Level Correction execution time: " << duration.count() << " seconds" << std::endl;

    // Save intermediate results if enabled
    if (is_save_) {
        save("blc_");
    }

    return result;
}

hdr_isp::EigenImageU32 BlackLevelCorrection::apply_blc_parameters_eigen(const hdr_isp::EigenImageU32& img) {
    int r_offset = static_cast<int>(parm_blc_["r_offset"].as<double>());
    int gb_offset = static_cast<int>(parm_blc_["gb_offset"].as<double>());
    int gr_offset = static_cast<int>(parm_blc_["gr_offset"].as<double>());
    int b_offset = static_cast<int>(parm_blc_["b_offset"].as<double>());
    int r_sat = static_cast<int>(parm_blc_["r_sat"].as<double>());
    int gr_sat = static_cast<int>(parm_blc_["gr_sat"].as<double>());
    int gb_sat = static_cast<int>(parm_blc_["gb_sat"].as<double>());
    int b_sat = static_cast<int>(parm_blc_["b_sat"].as<double>());
    
    hdr_isp::EigenImageU32 result = img.clone();
    apply_blc_bayer_eigen(result, r_offset, gr_offset, gb_offset, b_offset, r_sat, gr_sat, gb_sat, b_sat);
    
    return result;
}

void BlackLevelCorrection::apply_blc_bayer_eigen(hdr_isp::EigenImageU32& img, int r_offset, int gr_offset, int gb_offset, int b_offset, int r_sat, int gr_sat, int gb_sat, int b_sat) {
    int rows = img.rows();
    int cols = img.cols();
    
    // For early modules, clamp to 2^32 when bit depth is 32, otherwise use the configured bit depth
    uint32_t max_val;
    if (bit_depth_ == 32) {
        max_val = 4294967295U; // 2^32 - 1
    } else {
        max_val = (1U << bit_depth_) - 1;
    }
    
    // Convert offsets to uint32_t to avoid type mismatches
    uint32_t r_offset_u = static_cast<uint32_t>(r_offset);
    uint32_t gr_offset_u = static_cast<uint32_t>(gr_offset);
    uint32_t gb_offset_u = static_cast<uint32_t>(gb_offset);
    uint32_t b_offset_u = static_cast<uint32_t>(b_offset);
    
    // Black level correction should ONLY subtract the offset
    // Linearization should be a separate step in the pipeline
    
    if (bayer_pattern_ == "rggb") {
        // R channel
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > r_offset_u) ? (pixel_val - r_offset_u) : 0;
            }
        }
        // GR channel
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > gr_offset_u) ? (pixel_val - gr_offset_u) : 0;
            }
        }
        // GB channel
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > gb_offset_u) ? (pixel_val - gb_offset_u) : 0;
            }
        }
        // B channel
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > b_offset_u) ? (pixel_val - b_offset_u) : 0;
            }
        }
    }
    else if (bayer_pattern_ == "bggr") {
        // B channel
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > b_offset_u) ? (pixel_val - b_offset_u) : 0;
            }
        }
        // GB channel
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > gb_offset_u) ? (pixel_val - gb_offset_u) : 0;
            }
        }
        // GR channel
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > gr_offset_u) ? (pixel_val - gr_offset_u) : 0;
            }
        }
        // R channel
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > r_offset_u) ? (pixel_val - r_offset_u) : 0;
            }
        }
    }
    else if (bayer_pattern_ == "grbg") {
        // GR channel
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > gr_offset_u) ? (pixel_val - gr_offset_u) : 0;
            }
        }
        // R channel
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > r_offset_u) ? (pixel_val - r_offset_u) : 0;
            }
        }
        // B channel
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > b_offset_u) ? (pixel_val - b_offset_u) : 0;
            }
        }
        // GB channel
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > gb_offset_u) ? (pixel_val - gb_offset_u) : 0;
            }
        }
    }
    else if (bayer_pattern_ == "gbrg") {
        // GB channel
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > gb_offset_u) ? (pixel_val - gb_offset_u) : 0;
            }
        }
        // B channel
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > b_offset_u) ? (pixel_val - b_offset_u) : 0;
            }
        }
        // R channel
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > r_offset_u) ? (pixel_val - r_offset_u) : 0;
            }
        }
        // GR channel
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                uint32_t pixel_val = img.data()(i, j);
                img.data()(i, j) = (pixel_val > gr_offset_u) ? (pixel_val - gr_offset_u) : 0;
            }
        }
    }
    else {
        std::cerr << "Warning: Unsupported Bayer pattern '" << bayer_pattern_ << "' in Eigen BLC implementation" << std::endl;
    }
    
    // Clip values to valid range (0 to 2^32 for 32-bit, otherwise use configured bit depth)
    img = img.clip(0, max_val);
}

void BlackLevelCorrection::save(const std::string& filename_tag) {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/" + filename_tag + 
                                 std::to_string(raw_.cols()) + "x" + std::to_string(raw_.rows()) + ".png";
        // Convert to OpenCV for saving
        cv::Mat save_img = raw_.toOpenCV(CV_32S);
        cv::imwrite(output_path, save_img);
    }
} 
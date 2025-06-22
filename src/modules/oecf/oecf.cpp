#include "oecf.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

OECF::OECF(const hdr_isp::EigenImageU32& img, const YAML::Node& platform, const YAML::Node& sensor_info, const YAML::Node& parm_oecf)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_oecf_(parm_oecf)
    , enable_(parm_oecf["is_enable"].as<bool>())
    , is_save_(parm_oecf["is_save"].as<bool>())
    , is_debug_(parm_oecf["is_debug"].as<bool>())
{
}

hdr_isp::EigenImageU32 OECF::apply_oecf_eigen(const hdr_isp::EigenImageU32& img) {
    std::string bayer = sensor_info_["bayer_pattern"].as<std::string>();
    int bpp = sensor_info_["bit_depth"].as<int>();
    
    // For early modules, clamp to 2^32 when bit depth is 32, otherwise use the configured bit depth
    uint32_t max_value;
    if (bpp == 32) {
        max_value = 4294967295U; // 2^32 - 1
    } else {
        max_value = (1U << bpp) - 1;
    }

    // Get LUTs from parameters
    std::vector<uint16_t> r_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();
    std::vector<uint16_t> gr_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();
    std::vector<uint16_t> gb_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();
    std::vector<uint16_t> bl_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();

    int rows = img.rows();
    int cols = img.cols();
    
    // Debug output
    if (is_debug_) {
        std::cout << "OECF Eigen - Parameters:" << std::endl;
        std::cout << "  Bit depth: " << bpp << std::endl;
        std::cout << "  Bayer pattern: " << bayer << std::endl;
        std::cout << "  Max value: " << max_value << std::endl;
        std::cout << "  Image size: " << cols << "x" << rows << std::endl;
        std::cout << "  LUT sizes - R: " << r_lut.size() << ", GR: " << gr_lut.size() 
                  << ", GB: " << gb_lut.size() << ", B: " << bl_lut.size() << std::endl;
        
        // Print input image statistics
        uint32_t min_val = img.min();
        uint32_t max_val_input = img.max();
        float mean_val = img.mean();
        std::cout << "OECF Eigen - Input image - Mean: " << mean_val << ", Min: " << min_val << ", Max: " << max_val_input << std::endl;
    }
    
    hdr_isp::EigenImageU32 result = img.clone();

    if (bayer == "rggb") {
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_value = result.data()(i, j);
                if (pixel_value < r_lut.size())
                    result.data()(i, j) = static_cast<uint32_t>(r_lut[pixel_value]);
                pixel_value = result.data()(i, j + 1);
                if (pixel_value < gr_lut.size())
                    result.data()(i, j + 1) = static_cast<uint32_t>(gr_lut[pixel_value]);
                pixel_value = result.data()(i + 1, j);
                if (pixel_value < gb_lut.size())
                    result.data()(i + 1, j) = static_cast<uint32_t>(gb_lut[pixel_value]);
                pixel_value = result.data()(i + 1, j + 1);
                if (pixel_value < bl_lut.size())
                    result.data()(i + 1, j + 1) = static_cast<uint32_t>(bl_lut[pixel_value]);
            }
        }
    }
    else if (bayer == "bggr") {
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_value = result.data()(i, j);
                if (pixel_value < bl_lut.size())
                    result.data()(i, j) = static_cast<uint32_t>(bl_lut[pixel_value]);
                pixel_value = result.data()(i, j + 1);
                if (pixel_value < gb_lut.size())
                    result.data()(i, j + 1) = static_cast<uint32_t>(gb_lut[pixel_value]);
                pixel_value = result.data()(i + 1, j);
                if (pixel_value < gr_lut.size())
                    result.data()(i + 1, j) = static_cast<uint32_t>(gr_lut[pixel_value]);
                pixel_value = result.data()(i + 1, j + 1);
                if (pixel_value < r_lut.size())
                    result.data()(i + 1, j + 1) = static_cast<uint32_t>(r_lut[pixel_value]);
            }
        }
    }
    else if (bayer == "grbg") {
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_value = result.data()(i, j);
                if (pixel_value < gr_lut.size())
                    result.data()(i, j) = static_cast<uint32_t>(gr_lut[pixel_value]);
                pixel_value = result.data()(i, j + 1);
                if (pixel_value < r_lut.size())
                    result.data()(i, j + 1) = static_cast<uint32_t>(r_lut[pixel_value]);
                pixel_value = result.data()(i + 1, j);
                if (pixel_value < bl_lut.size())
                    result.data()(i + 1, j) = static_cast<uint32_t>(bl_lut[pixel_value]);
                pixel_value = result.data()(i + 1, j + 1);
                if (pixel_value < gb_lut.size())
                    result.data()(i + 1, j + 1) = static_cast<uint32_t>(gb_lut[pixel_value]);
            }
        }
    }
    else if (bayer == "gbrg") {
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                uint32_t pixel_value = result.data()(i, j);
                if (pixel_value < gb_lut.size())
                    result.data()(i, j) = static_cast<uint32_t>(gb_lut[pixel_value]);
                pixel_value = result.data()(i, j + 1);
                if (pixel_value < bl_lut.size())
                    result.data()(i, j + 1) = static_cast<uint32_t>(bl_lut[pixel_value]);
                pixel_value = result.data()(i + 1, j);
                if (pixel_value < r_lut.size())
                    result.data()(i + 1, j) = static_cast<uint32_t>(r_lut[pixel_value]);
                pixel_value = result.data()(i + 1, j + 1);
                if (pixel_value < gr_lut.size())
                    result.data()(i + 1, j + 1) = static_cast<uint32_t>(gr_lut[pixel_value]);
            }
        }
    }

    // Clip values to valid range using Eigen (0 to 2^32 for 32-bit, otherwise use configured bit depth)
    result = result.clip(0, max_value);
    
    if (is_debug_) {
        // Print output image statistics
        uint32_t min_val_out = result.min();
        uint32_t max_val_out = result.max();
        float mean_val_out = result.mean();
        std::cout << "OECF Eigen - Output image - Mean: " << mean_val_out << ", Min: " << min_val_out << ", Max: " << max_val_out << std::endl;
    }
    
    return result;
}

void OECF::save(const std::string& filename_tag) {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/" + filename_tag + 
            std::to_string(img_.cols()) + "x" + std::to_string(img_.rows()) + "_" +
            std::to_string(sensor_info_["bit_depth"].as<int>()) + "bits_" +
            sensor_info_["bayer_pattern"].as<std::string>() + ".png";
        // Convert to OpenCV for saving
        cv::Mat save_img = img_.toOpenCV(CV_32S);
        cv::imwrite(output_path, save_img);
    }
}

hdr_isp::EigenImageU32 OECF::execute() {
    if (!enable_) {
        return img_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Apply OECF using Eigen
    img_ = apply_oecf_eigen(img_);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (is_debug_) {
        std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
    }

    // Save intermediate results if enabled
    if (is_save_) {
        save("oecf_");
    }

    return img_;
} 
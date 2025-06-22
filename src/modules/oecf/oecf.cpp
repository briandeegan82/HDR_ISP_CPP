#include "oecf.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

OECF::OECF(cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info, const YAML::Node& parm_oecf)
    : img_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_oecf_(parm_oecf)
    , enable_(parm_oecf["is_enable"].as<bool>())
    , is_save_(parm_oecf["is_save"].as<bool>())
    , use_eigen_(true) // Use Eigen by default
    , is_debug_(parm_oecf["is_debug"].as<bool>())
{
}

cv::Mat OECF::apply_oecf_opencv() {
    cv::Mat raw = img_.clone();
    std::string bayer = sensor_info_["bayer_pattern"].as<std::string>();
    int bpp = sensor_info_["bit_depth"].as<int>();
    
    // For early modules, clamp to 2^32 when bit depth is 32, otherwise use the configured bit depth
    int max_value;
    if (bpp == 32) {
        max_value = 4294967295; // 2^32 - 1
    } else {
        max_value = (1 << bpp) - 1;
    }

    // Get LUTs from parameters
    std::vector<uint16_t> r_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();
    std::vector<uint16_t> gr_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();
    std::vector<uint16_t> gb_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();
    std::vector<uint16_t> bl_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();

    cv::Mat raw_oecf = cv::Mat::zeros(raw.size(), raw.type());

    if (bayer == "rggb") {
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw_oecf.at<uint16_t>(i, j) = r_lut[raw.at<uint16_t>(i, j)];
                raw_oecf.at<uint16_t>(i, j + 1) = gr_lut[raw.at<uint16_t>(i, j + 1)];
                raw_oecf.at<uint16_t>(i + 1, j) = gb_lut[raw.at<uint16_t>(i + 1, j)];
                raw_oecf.at<uint16_t>(i + 1, j + 1) = bl_lut[raw.at<uint16_t>(i + 1, j + 1)];
            }
        }
    }
    else if (bayer == "bggr") {
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw_oecf.at<uint16_t>(i, j) = bl_lut[raw.at<uint16_t>(i, j)];
                raw_oecf.at<uint16_t>(i, j + 1) = gb_lut[raw.at<uint16_t>(i, j + 1)];
                raw_oecf.at<uint16_t>(i + 1, j) = gr_lut[raw.at<uint16_t>(i + 1, j)];
                raw_oecf.at<uint16_t>(i + 1, j + 1) = r_lut[raw.at<uint16_t>(i + 1, j + 1)];
            }
        }
    }
    else if (bayer == "grbg") {
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw_oecf.at<uint16_t>(i, j) = gr_lut[raw.at<uint16_t>(i, j)];
                raw_oecf.at<uint16_t>(i, j + 1) = r_lut[raw.at<uint16_t>(i, j + 1)];
                raw_oecf.at<uint16_t>(i + 1, j) = bl_lut[raw.at<uint16_t>(i + 1, j)];
                raw_oecf.at<uint16_t>(i + 1, j + 1) = gb_lut[raw.at<uint16_t>(i + 1, j + 1)];
            }
        }
    }
    else if (bayer == "gbrg") {
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw_oecf.at<uint16_t>(i, j) = gb_lut[raw.at<uint16_t>(i, j)];
                raw_oecf.at<uint16_t>(i, j + 1) = bl_lut[raw.at<uint16_t>(i, j + 1)];
                raw_oecf.at<uint16_t>(i + 1, j) = r_lut[raw.at<uint16_t>(i + 1, j)];
                raw_oecf.at<uint16_t>(i + 1, j + 1) = gr_lut[raw.at<uint16_t>(i + 1, j + 1)];
            }
        }
    }

    // Clip values to valid range (0 to 2^32 for 32-bit, otherwise use configured bit depth)
    cv::threshold(raw_oecf, raw_oecf, max_value, max_value, cv::THRESH_TRUNC);
    return raw_oecf;
}

hdr_isp::EigenImageU32 OECF::apply_oecf_eigen() {
    std::string bayer = sensor_info_["bayer_pattern"].as<std::string>();
    int bpp = sensor_info_["bit_depth"].as<int>();
    
    // For early modules, clamp to 2^32 when bit depth is 32, otherwise use the configured bit depth
    int max_value;
    if (bpp == 32) {
        max_value = 4294967295; // 2^32 - 1
    } else {
        max_value = (1 << bpp) - 1;
    }

    // Get LUTs from parameters
    std::vector<uint16_t> r_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();
    std::vector<uint16_t> gr_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();
    std::vector<uint16_t> gb_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();
    std::vector<uint16_t> bl_lut = parm_oecf_["r_lut"].as<std::vector<uint16_t>>();

    hdr_isp::EigenImageU32 eigen_img = hdr_isp::EigenImageU32::fromOpenCV(img_);
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();

    if (bayer == "rggb") {
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                int pixel_value = eigen_img.data()(i, j);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(r_lut.size()))
                    eigen_img.data()(i, j) = static_cast<int>(r_lut[pixel_value]);
                pixel_value = eigen_img.data()(i, j + 1);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(gr_lut.size()))
                    eigen_img.data()(i, j + 1) = static_cast<int>(gr_lut[pixel_value]);
                pixel_value = eigen_img.data()(i + 1, j);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(gb_lut.size()))
                    eigen_img.data()(i + 1, j) = static_cast<int>(gb_lut[pixel_value]);
                pixel_value = eigen_img.data()(i + 1, j + 1);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(bl_lut.size()))
                    eigen_img.data()(i + 1, j + 1) = static_cast<int>(bl_lut[pixel_value]);
            }
        }
    }
    else if (bayer == "bggr") {
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                int pixel_value = eigen_img.data()(i, j);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(bl_lut.size()))
                    eigen_img.data()(i, j) = static_cast<int>(bl_lut[pixel_value]);
                pixel_value = eigen_img.data()(i, j + 1);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(gb_lut.size()))
                    eigen_img.data()(i, j + 1) = static_cast<int>(gb_lut[pixel_value]);
                pixel_value = eigen_img.data()(i + 1, j);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(gr_lut.size()))
                    eigen_img.data()(i + 1, j) = static_cast<int>(gr_lut[pixel_value]);
                pixel_value = eigen_img.data()(i + 1, j + 1);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(r_lut.size()))
                    eigen_img.data()(i + 1, j + 1) = static_cast<int>(r_lut[pixel_value]);
            }
        }
    }
    else if (bayer == "grbg") {
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                int pixel_value = eigen_img.data()(i, j);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(gr_lut.size()))
                    eigen_img.data()(i, j) = static_cast<int>(gr_lut[pixel_value]);
                pixel_value = eigen_img.data()(i, j + 1);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(r_lut.size()))
                    eigen_img.data()(i, j + 1) = static_cast<int>(r_lut[pixel_value]);
                pixel_value = eigen_img.data()(i + 1, j);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(bl_lut.size()))
                    eigen_img.data()(i + 1, j) = static_cast<int>(bl_lut[pixel_value]);
                pixel_value = eigen_img.data()(i + 1, j + 1);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(gb_lut.size()))
                    eigen_img.data()(i + 1, j + 1) = static_cast<int>(gb_lut[pixel_value]);
            }
        }
    }
    else if (bayer == "gbrg") {
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                int pixel_value = eigen_img.data()(i, j);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(gb_lut.size()))
                    eigen_img.data()(i, j) = static_cast<int>(gb_lut[pixel_value]);
                pixel_value = eigen_img.data()(i, j + 1);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(bl_lut.size()))
                    eigen_img.data()(i, j + 1) = static_cast<int>(bl_lut[pixel_value]);
                pixel_value = eigen_img.data()(i + 1, j);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(r_lut.size()))
                    eigen_img.data()(i + 1, j) = static_cast<int>(r_lut[pixel_value]);
                pixel_value = eigen_img.data()(i + 1, j + 1);
                if (pixel_value >= 0 && pixel_value < static_cast<int>(gr_lut.size()))
                    eigen_img.data()(i + 1, j + 1) = static_cast<int>(gr_lut[pixel_value]);
            }
        }
    }

    // Clip values to valid range using Eigen (0 to 2^32 for 32-bit, otherwise use configured bit depth)
    eigen_img = eigen_img.clip(0, max_value);
    return eigen_img;
}

void OECF::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_oecf_" + 
            std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + "_" +
            std::to_string(sensor_info_["bit_depth"].as<int>()) + "bits_" +
            sensor_info_["bayer_pattern"].as<std::string>() + ".png";
        cv::imwrite(output_path, img_);
    }
}

cv::Mat OECF::execute() {
    if (enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (use_eigen_) {
            hdr_isp::EigenImageU32 result = apply_oecf_eigen();
            img_ = result.toOpenCV(img_.type());
        } else {
            img_ = apply_oecf_opencv();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    return img_;
} 
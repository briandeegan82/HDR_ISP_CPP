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
{
}

cv::Mat OECF::apply_oecf() {
    cv::Mat raw = img_.clone();
    std::string bayer = sensor_info_["bayer_pattern"].as<std::string>();
    int bpp = sensor_info_["bit_depth"].as<int>();
    int max_value = (1 << bpp) - 1;

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

    // Clip values to valid range
    cv::threshold(raw_oecf, raw_oecf, max_value, max_value, cv::THRESH_TRUNC);
    return raw_oecf;
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
        img_ = apply_oecf();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "OECF execution time: " << duration.count() << "s" << std::endl;
    }
    save();
    return img_;
} 
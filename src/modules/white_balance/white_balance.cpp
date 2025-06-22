#include "white_balance.hpp"
#include "../../common/common.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

WhiteBalance::WhiteBalance(const hdr_isp::EigenImageU32& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                         const YAML::Node& parm_wbc)
    : img_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_wbc_(parm_wbc)
{
    is_enable_ = parm_wbc_["is_enable"].as<bool>();
    is_save_ = parm_wbc_["is_save"].as<bool>();
    is_auto_ = parm_wbc_["is_auto"].as<bool>();
    is_debug_ = parm_wbc_["is_debug"].as<bool>();
    bayer_ = sensor_info_["bayer_pattern"].as<std::string>();
    bpp_ = sensor_info_["bit_depth"].as<int>();
    raw_ = img.clone();
}

hdr_isp::EigenImageU32 WhiteBalance::execute()
{
    if (is_enable_) {
        std::cout << "White balancing = True" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        img_ = apply_wb_parameters();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "  Execution time: " << duration.count() << "s" << std::endl;
    }

    return img_;
}

hdr_isp::EigenImageU32 WhiteBalance::apply_wb_parameters()
{
    // Get config parameters
    float red_gain = parm_wbc_["r_gain"].as<float>();
    float blue_gain = parm_wbc_["b_gain"].as<float>();
    
    if (is_debug_) {
        std::cout << "   - WB  - red gain : " << red_gain << std::endl;
        std::cout << "   - WB  - blue gain: " << blue_gain << std::endl;
    }

    hdr_isp::EigenImageU32 result = img_.clone();
    int rows = result.rows();
    int cols = result.cols();

    // Apply gains based on Bayer pattern
    if (bayer_ == "rggb") {
        // Red pixels
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                result.data()(i, j) = static_cast<uint32_t>(std::round(result.data()(i, j) * red_gain));
            }
        }
        // Blue pixels
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                result.data()(i, j) = static_cast<uint32_t>(std::round(result.data()(i, j) * blue_gain));
            }
        }
    }
    else if (bayer_ == "bggr") {
        // Blue pixels
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                result.data()(i, j) = static_cast<uint32_t>(std::round(result.data()(i, j) * blue_gain));
            }
        }
        // Red pixels
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                result.data()(i, j) = static_cast<uint32_t>(std::round(result.data()(i, j) * red_gain));
            }
        }
    }
    else if (bayer_ == "grbg") {
        // Blue pixels
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                result.data()(i, j) = static_cast<uint32_t>(std::round(result.data()(i, j) * blue_gain));
            }
        }
        // Red pixels
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                result.data()(i, j) = static_cast<uint32_t>(std::round(result.data()(i, j) * red_gain));
            }
        }
    }
    else if (bayer_ == "gbrg") {
        // Red pixels
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                result.data()(i, j) = static_cast<uint32_t>(std::round(result.data()(i, j) * red_gain));
            }
        }
        // Blue pixels
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                result.data()(i, j) = static_cast<uint32_t>(std::round(result.data()(i, j) * blue_gain));
            }
        }
    }

    // Clip values to valid range
    uint32_t max_val = (1U << bpp_) - 1;
    result = result.cwiseMin(max_val);

    return result;
}

void WhiteBalance::save()
{
    if (is_save_) {
        std::string filename = common::get_output_filename(platform_["in_file"].as<std::string>(), "Out_white_balance_");
        // Convert to OpenCV for saving
        cv::Mat img_cv = img_.toOpenCV(CV_32S);
        common::save_image(img_cv, filename);
    }
} 
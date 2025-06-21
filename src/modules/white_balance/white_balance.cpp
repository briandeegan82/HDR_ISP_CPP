#include "white_balance.hpp"
#include "../../common/common.hpp"
#include <chrono>
#include <iostream>

WhiteBalance::WhiteBalance(const cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info,
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
    use_eigen_ = true; // Use Eigen by default
}

cv::Mat WhiteBalance::execute()
{
    if (is_enable_) {
        std::cout << "White balancing = True" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        if (use_eigen_) {
            hdr_isp::EigenImage result = apply_wb_parameters_eigen();
            img_ = result.toOpenCV(img_.type());
        } else {
            img_ = apply_wb_parameters_opencv();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "  Execution time: " << duration.count() << "s" << std::endl;
    }

    return img_;
}

cv::Mat WhiteBalance::apply_wb_parameters_opencv()
{
    // Get config parameters
    float red_gain = parm_wbc_["r_gain"].as<float>();
    float blue_gain = parm_wbc_["b_gain"].as<float>();
    
    if (is_debug_) {
        std::cout << "   - WB  - red gain : " << red_gain << std::endl;
        std::cout << "   - WB  - blue gain: " << blue_gain << std::endl;
    }

    // Convert to float32 for calculations
    cv::Mat raw_float;
    img_.convertTo(raw_float, CV_32F);

    // Apply gains based on Bayer pattern
    if (bayer_ == "rggb") {
        // Red pixels
        for (int i = 0; i < raw_float.rows; i += 2) {
            for (int j = 0; j < raw_float.cols; j += 2) {
                raw_float.at<float>(i, j) *= red_gain;
            }
        }
        // Blue pixels
        for (int i = 1; i < raw_float.rows; i += 2) {
            for (int j = 1; j < raw_float.cols; j += 2) {
                raw_float.at<float>(i, j) *= blue_gain;
            }
        }
    }
    else if (bayer_ == "bggr") {
        // Blue pixels
        for (int i = 0; i < raw_float.rows; i += 2) {
            for (int j = 0; j < raw_float.cols; j += 2) {
                raw_float.at<float>(i, j) *= blue_gain;
            }
        }
        // Red pixels
        for (int i = 1; i < raw_float.rows; i += 2) {
            for (int j = 1; j < raw_float.cols; j += 2) {
                raw_float.at<float>(i, j) *= red_gain;
            }
        }
    }
    else if (bayer_ == "grbg") {
        // Blue pixels
        for (int i = 1; i < raw_float.rows; i += 2) {
            for (int j = 0; j < raw_float.cols; j += 2) {
                raw_float.at<float>(i, j) *= blue_gain;
            }
        }
        // Red pixels
        for (int i = 0; i < raw_float.rows; i += 2) {
            for (int j = 1; j < raw_float.cols; j += 2) {
                raw_float.at<float>(i, j) *= red_gain;
            }
        }
    }
    else if (bayer_ == "gbrg") {
        // Red pixels
        for (int i = 1; i < raw_float.rows; i += 2) {
            for (int j = 0; j < raw_float.cols; j += 2) {
                raw_float.at<float>(i, j) *= red_gain;
            }
        }
        // Blue pixels
        for (int i = 0; i < raw_float.rows; i += 2) {
            for (int j = 1; j < raw_float.cols; j += 2) {
                raw_float.at<float>(i, j) *= blue_gain;
            }
        }
    }

    // Clip values to valid range and convert back to original type
    cv::Mat raw_whitebal;
    double max_val = (1 << bpp_) - 1;
    cv::threshold(raw_float, raw_float, max_val, max_val, cv::THRESH_TRUNC);
    raw_float.convertTo(raw_whitebal, img_.type());

    return raw_whitebal;
}

hdr_isp::EigenImage WhiteBalance::apply_wb_parameters_eigen()
{
    float red_gain = parm_wbc_["r_gain"].as<float>();
    float blue_gain = parm_wbc_["b_gain"].as<float>();
    if (is_debug_) {
        std::cout << "   - WB  - red gain : " << red_gain << std::endl;
        std::cout << "   - WB  - blue gain: " << blue_gain << std::endl;
    }
    hdr_isp::EigenImage eigen_img = hdr_isp::EigenImage::fromOpenCV(img_);
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    if (bayer_ == "rggb") {
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                eigen_img.data()(i, j) *= red_gain;
            }
        }
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                eigen_img.data()(i, j) *= blue_gain;
            }
        }
    }
    else if (bayer_ == "bggr") {
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                eigen_img.data()(i, j) *= blue_gain;
            }
        }
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                eigen_img.data()(i, j) *= red_gain;
            }
        }
    }
    else if (bayer_ == "grbg") {
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                eigen_img.data()(i, j) *= blue_gain;
            }
        }
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                eigen_img.data()(i, j) *= red_gain;
            }
        }
    }
    else if (bayer_ == "gbrg") {
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                eigen_img.data()(i, j) *= red_gain;
            }
        }
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                eigen_img.data()(i, j) *= blue_gain;
            }
        }
    }
    float max_val = static_cast<float>((1 << bpp_) - 1);
    eigen_img.data() = eigen_img.data().cwiseMin(max_val);
    return eigen_img;
}

void WhiteBalance::save()
{
    if (is_save_) {
        std::string filename = common::get_output_filename(platform_["in_file"].as<std::string>(), "Out_white_balance_");
        common::save_image(img_, filename);
    }
} 
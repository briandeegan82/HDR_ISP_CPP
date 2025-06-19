#include "pwc_generation.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <stdexcept>

namespace fs = std::filesystem;

PiecewiseCurve::PiecewiseCurve(cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info, const YAML::Node& parm_cmpd)
    : img_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_cmpd_(parm_cmpd)
    , enable_(parm_cmpd["is_enable"].as<bool>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , companded_pin_(parm_cmpd["companded_pin"].as<std::vector<int>>())
    , companded_pout_(parm_cmpd["companded_pout"].as<std::vector<int>>())
    , is_save_(parm_cmpd["is_save"].as<bool>())
    , use_eigen_(true) // Use Eigen by default
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

void PiecewiseCurve::save() {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/Out_decompanding_" + 
            std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + "_" +
            std::to_string(bit_depth_) + "bits_" +
            sensor_info_["bayer_pattern"].as<std::string>() + ".png";
        cv::imwrite(output_path, img_);
    }
}

cv::Mat PiecewiseCurve::execute() {
    if (!enable_) {
        save();
        return img_;
    }
    if (use_eigen_) {
        auto start = std::chrono::high_resolution_clock::now();
        hdr_isp::EigenImage eigen_img = hdr_isp::EigenImage::fromOpenCV(img_);
        hdr_isp::EigenImage result = execute_eigen();
        result.data() = result.data().cwiseMax(0.0f); // Ensure no negative values
        result.data() = result.data().unaryExpr([](float v) { return std::max(0.0f, v); });
        result.toOpenCV(img_.type()).copyTo(img_);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "PWC generation (Eigen) execution time: " << duration.count() << "s" << std::endl;
    } else {
        img_ = execute_opencv();
    }
    save();
    return img_;
}

cv::Mat PiecewiseCurve::execute_opencv() {
    // Generate decompanding LUT
    std::vector<double> lut = generate_decompanding_lut(
        companded_pin_,
        companded_pout_,
        companded_pin_.back()
    );

    // Convert image to float for processing
    cv::Mat img_float;
    img_.convertTo(img_float, CV_32F);

    // Apply LUT to each pixel
    for (int i = 0; i < img_float.rows; ++i) {
        for (int j = 0; j < img_float.cols; ++j) {
            int pixel_value = static_cast<int>(img_float.at<float>(i, j));
            img_float.at<float>(i, j) = lut[pixel_value];
        }
    }

    // Subtract pedestal and clip negative values
    double pedestal = parm_cmpd_["pedestal"].as<double>();
    cv::subtract(img_float, cv::Scalar(pedestal), img_float);
    cv::max(img_float, 0.0, img_float);

    // Convert back to original type
    img_float.convertTo(img_, CV_32S);

    return img_;
}

hdr_isp::EigenImage PiecewiseCurve::execute_eigen() {
    std::vector<double> lut = generate_decompanding_lut(
        companded_pin_,
        companded_pout_,
        companded_pin_.back()
    );
    hdr_isp::EigenImage eigen_img = hdr_isp::EigenImage::fromOpenCV(img_);
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    // Apply LUT to each pixel using Eigen
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int pixel_value = static_cast<int>(eigen_img.data()(i, j));
            if (pixel_value >= 0 && pixel_value < static_cast<int>(lut.size()))
                eigen_img.data()(i, j) = static_cast<float>(lut[pixel_value]);
            else
                eigen_img.data()(i, j) = 0.0f;
        }
    }
    double pedestal = parm_cmpd_["pedestal"].as<double>();
    eigen_img.data().array() -= static_cast<float>(pedestal);
    eigen_img.data() = eigen_img.data().cwiseMax(0.0f);
    eigen_img.data() = eigen_img.data().unaryExpr([](float v) { return std::max(0.0f, v); });
    eigen_img.data() = eigen_img.data().array().round();
    return eigen_img;
} 
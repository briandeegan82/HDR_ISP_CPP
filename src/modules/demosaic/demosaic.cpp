#include "demosaic.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// Initialize static filter coefficients
const cv::Mat Malvar::g_at_r_and_b = (cv::Mat_<float>(5, 5) <<
    0, 0, -1, 0, 0,
    0, 0, 2, 0, 0,
    -1, 2, 4, 2, -1,
    0, 0, 2, 0, 0,
    0, 0, -1, 0, 0) * 0.125f;

const cv::Mat Malvar::r_at_gr_and_b_at_gb = (cv::Mat_<float>(5, 5) <<
    0, 0, 0.5, 0, 0,
    0, -1, 0, -1, 0,
    -1, 4, 5, 4, -1,
    0, -1, 0, -1, 0,
    0, 0, 0.5, 0, 0) * 0.125f;

const cv::Mat Malvar::r_at_gb_and_b_at_gr = r_at_gr_and_b_at_gb.t();

const cv::Mat Malvar::r_at_b_and_b_at_r = (cv::Mat_<float>(5, 5) <<
    0, 0, -1.5, 0, 0,
    0, 2, 0, 2, 0,
    -1.5, 0, 6, 0, -1.5,
    0, 2, 0, 2, 0,
    0, 0, -1.5, 0, 0) * 0.125f;

// Demosaic class implementation
Demosaic::Demosaic(const cv::Mat& img, const std::string& bayer_pattern, int bit_depth, bool is_save)
    : img_(img)
    , bayer_pattern_(bayer_pattern)
    , bit_depth_(bit_depth)
    , is_save_(is_save) {
}

std::array<cv::Mat, 3> Demosaic::masks_cfa_bayer() {
    std::array<cv::Mat, 3> channels;
    for (auto& channel : channels) {
        channel = cv::Mat::zeros(img_.size(), CV_8UC1);
    }

    // Create masks based on bayer pattern
    for (size_t i = 0; i < bayer_pattern_.length(); ++i) {
        char channel = bayer_pattern_[i];
        int y = (i / 2) % 2;
        int x = i % 2;
        channels[channel == 'r' ? 0 : (channel == 'g' ? 1 : 2)].at<uchar>(y::2, x::2) = 1;
    }

    return channels;
}

cv::Mat Demosaic::apply_cfa() {
    auto masks = masks_cfa_bayer();
    Malvar mal(img_, masks);
    cv::Mat demos_out = mal.apply_malvar();

    // Clip values to bit depth range
    cv::threshold(demos_out, demos_out, (1 << bit_depth_) - 1, (1 << bit_depth_) - 1, cv::THRESH_TRUNC);
    demos_out.convertTo(demos_out, CV_16UC3);

    return demos_out;
}

void Demosaic::save() {
    if (is_save_) {
        fs::path output_dir = "out_frames";
        fs::create_directories(output_dir);
        fs::path output_path = output_dir / ("Out_demosaic_" + bayer_pattern_ + ".png");
        cv::imwrite(output_path.string(), img_);
    }
}

cv::Mat Demosaic::execute() {
    std::cout << "CFA interpolation (default) = True" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat cfa_out = apply_cfa();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "  Execution time: " << elapsed.count() << "s" << std::endl;
    
    img_ = cfa_out;
    save();
    return img_;
}

// Malvar class implementation
Malvar::Malvar(const cv::Mat& raw_in, const std::array<cv::Mat, 3>& masks)
    : img_(raw_in)
    , masks_(masks) {
}

cv::Mat Malvar::apply_malvar() {
    cv::Mat raw_in;
    img_.convertTo(raw_in, CV_32F);

    // Create output image
    cv::Mat demos_out(raw_in.size(), CV_32FC3);

    // Extract channels using masks
    cv::Mat r_channel = raw_in.mul(masks_[0]);
    cv::Mat g_channel = raw_in.mul(masks_[1]);
    cv::Mat b_channel = raw_in.mul(masks_[2]);

    // Apply filters
    cv::Mat g_at_r_and_b_out;
    cv::filter2D(raw_in, g_at_r_and_b_out, CV_32F, g_at_r_and_b, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    g_channel = cv::Mat::zeros(raw_in.size(), CV_32F);
    cv::bitwise_or(masks_[0], masks_[2], g_channel);
    g_channel = g_channel.mul(g_at_r_and_b_out) + g_channel.mul(raw_in);

    cv::Mat rb_at_g_rbbr, rb_at_g_brrb, rb_at_gr_bbrr;
    cv::filter2D(raw_in, rb_at_g_rbbr, CV_32F, r_at_gr_and_b_at_gb, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(raw_in, rb_at_g_brrb, CV_32F, r_at_gb_and_b_at_gr, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(raw_in, rb_at_gr_bbrr, CV_32F, r_at_b_and_b_at_r, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

    // Create row and column masks
    cv::Mat r_rows, r_col, b_rows, b_col;
    cv::reduce(masks_[0], r_rows, 1, cv::REDUCE_MAX);
    cv::reduce(masks_[0], r_col, 0, cv::REDUCE_MAX);
    cv::reduce(masks_[2], b_rows, 1, cv::REDUCE_MAX);
    cv::reduce(masks_[2], b_col, 0, cv::REDUCE_MAX);

    // Update R channel
    cv::Mat r_update_mask = (r_rows * r_col) > 0;
    r_channel = r_update_mask.mul(rb_at_g_rbbr) + (~r_update_mask).mul(r_channel);
    r_update_mask = (b_rows * r_col) > 0;
    r_channel = r_update_mask.mul(rb_at_g_brrb) + (~r_update_mask).mul(r_channel);
    r_update_mask = (b_rows * b_col) > 0;
    r_channel = r_update_mask.mul(rb_at_gr_bbrr) + (~r_update_mask).mul(r_channel);

    // Update B channel
    cv::Mat b_update_mask = (b_rows * r_col) > 0;
    b_channel = b_update_mask.mul(rb_at_g_rbbr) + (~b_update_mask).mul(b_channel);
    b_update_mask = (r_rows * b_col) > 0;
    b_channel = b_update_mask.mul(rb_at_g_brrb) + (~b_update_mask).mul(b_channel);
    b_update_mask = (r_rows * r_col) > 0;
    b_channel = b_update_mask.mul(rb_at_gr_bbrr) + (~b_update_mask).mul(b_channel);

    // Combine channels
    std::vector<cv::Mat> channels = {r_channel, g_channel, b_channel};
    cv::merge(channels, demos_out);

    return demos_out;
} 
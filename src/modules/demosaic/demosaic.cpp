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
Demosaic::Demosaic(const cv::Mat& img, const std::string& bayer_pattern, int bit_depth, bool is_save, DemosaicAlgorithm algorithm)
    : img_(img)
    , bayer_pattern_(bayer_pattern)
    , bit_depth_(bit_depth)
    , is_save_(is_save)
    , algorithm_(algorithm) {
}

std::array<cv::Mat, 3> Demosaic::masks_cfa_bayer() {
    std::array<cv::Mat, 3> channels;
    for (auto& channel : channels) {
        channel = cv::Mat::zeros(img_.size(), CV_8UC1);
    }

    // Create masks based on bayer pattern
    for (int y = 0; y < img_.rows; y += 2) {
        for (int x = 0; x < img_.cols; x += 2) {
            // Get the 2x2 bayer pattern for this block
            std::string block_pattern = bayer_pattern_.substr(0, 4);
            
            // Set the mask values for each channel in the 2x2 block
            for (int i = 0; i < 4; ++i) {
                int block_y = y + (i / 2);
                int block_x = x + (i % 2);
                if (block_y < img_.rows && block_x < img_.cols) {
                    char channel = block_pattern[i];
                    channels[channel == 'r' ? 0 : (channel == 'g' ? 1 : 2)].at<uchar>(block_y, block_x) = 1;
                }
            }
        }
    }

    return channels;
}

cv::Mat Demosaic::apply_cfa() {
    auto masks = masks_cfa_bayer();
    Malvar mal(img_, masks);
    cv::Mat demos_out = mal.apply_malvar();

    // Scale to full bit depth range
    double scale = (1 << bit_depth_) - 1;
    demos_out *= scale;

    // Convert to 16-bit
    cv::Mat demos_out_16u;
    demos_out.convertTo(demos_out_16u, CV_16UC3);
    
    return demos_out_16u;
}

cv::Mat Demosaic::apply_opencv_demosaic() {
    cv::Mat demos_out;
    
    // Convert bayer pattern string to OpenCV format
    int code;
    if (bayer_pattern_ == "rggb") {
        code = cv::COLOR_BayerBG2BGR;  // OpenCV uses BG as reference
    } else if (bayer_pattern_ == "grbg") {
        code = cv::COLOR_BayerGB2BGR;
    } else if (bayer_pattern_ == "gbrg") {
        code = cv::COLOR_BayerRG2BGR;
    } else if (bayer_pattern_ == "bggr") {
        code = cv::COLOR_BayerGR2BGR;
    } else {
        throw std::runtime_error("Unsupported bayer pattern: " + bayer_pattern_);
    }

    // Convert input to 16-bit for OpenCV demosaic
    cv::Mat img_16u;
    img_.convertTo(img_16u, CV_16U);

    // Apply OpenCV demosaic
    cv::cvtColor(img_16u, demos_out, code);
    
    return demos_out;
}

void Demosaic::save() {
    if (is_save_) {
        fs::path output_dir = "out_frames";
        fs::create_directories(output_dir);
        fs::path output_path = output_dir / ("Out_demosaic_" + bayer_pattern_ + ".png");
        
        // Convert to 8-bit for saving
        cv::Mat save_img;
        double scale = 255.0 / ((1 << bit_depth_) - 1);
        img_.convertTo(save_img, CV_8UC3, scale);
        cv::imwrite(output_path.string(), save_img);
    }
}

cv::Mat Demosaic::execute() {
    std::cout << "Demosaic algorithm: " << (algorithm_ == DemosaicAlgorithm::MALVAR ? "Malvar" : "OpenCV") << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat cfa_out;
    if (algorithm_ == DemosaicAlgorithm::MALVAR) {
        cfa_out = apply_cfa();
    } else {
        cfa_out = apply_opencv_demosaic();
    }
    
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

    // Convert masks to float type for arithmetic operations
    std::array<cv::Mat, 3> float_masks;
    for (int i = 0; i < 3; ++i) {
        masks_[i].convertTo(float_masks[i], CV_32F);
    }

    // Extract channels using masks
    cv::Mat r_channel = raw_in.mul(float_masks[0]);
    cv::Mat g_channel = raw_in.mul(float_masks[1]);
    cv::Mat b_channel = raw_in.mul(float_masks[2]);

    // Apply filters
    cv::Mat g_at_r_and_b_out;
    cv::filter2D(raw_in, g_at_r_and_b_out, CV_32F, g_at_r_and_b, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    
    // Create green channel mask
    cv::Mat g_mask;
    cv::add(float_masks[0], float_masks[2], g_mask);
    g_channel = cv::Mat::zeros(raw_in.size(), CV_32F);
    g_channel = g_mask.mul(g_at_r_and_b_out) + g_mask.mul(raw_in);

    // Apply filters for red and blue channels
    cv::Mat rb_at_g_rbbr, rb_at_g_brrb, rb_at_gr_bbrr;
    cv::filter2D(raw_in, rb_at_g_rbbr, CV_32F, r_at_gr_and_b_at_gb, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(raw_in, rb_at_g_brrb, CV_32F, r_at_gb_and_b_at_gr, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(raw_in, rb_at_gr_bbrr, CV_32F, r_at_b_and_b_at_r, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

    // Create row and column masks
    cv::Mat r_rows, r_col, b_rows, b_col;
    cv::reduce(float_masks[0], r_rows, 1, cv::REDUCE_MAX);
    cv::reduce(float_masks[0], r_col, 0, cv::REDUCE_MAX);
    cv::reduce(float_masks[2], b_rows, 1, cv::REDUCE_MAX);
    cv::reduce(float_masks[2], b_col, 0, cv::REDUCE_MAX);

    // Ensure all masks have the same size as the input image
    cv::resize(r_rows, r_rows, raw_in.size(), 0, 0, cv::INTER_NEAREST);
    cv::resize(r_col, r_col, raw_in.size(), 0, 0, cv::INTER_NEAREST);
    cv::resize(b_rows, b_rows, raw_in.size(), 0, 0, cv::INTER_NEAREST);
    cv::resize(b_col, b_col, raw_in.size(), 0, 0, cv::INTER_NEAREST);

    // Update R channel
    cv::Mat r_update_mask;
    cv::multiply(r_rows, r_col, r_update_mask);
    r_channel = r_update_mask.mul(rb_at_g_rbbr) + (1.0f - r_update_mask).mul(r_channel);
    cv::multiply(b_rows, r_col, r_update_mask);
    r_channel = r_update_mask.mul(rb_at_g_brrb) + (1.0f - r_update_mask).mul(r_channel);
    cv::multiply(b_rows, b_col, r_update_mask);
    r_channel = r_update_mask.mul(rb_at_gr_bbrr) + (1.0f - r_update_mask).mul(r_channel);

    // Update B channel
    cv::Mat b_update_mask;
    cv::multiply(b_rows, r_col, b_update_mask);
    b_channel = b_update_mask.mul(rb_at_g_rbbr) + (1.0f - b_update_mask).mul(b_channel);
    cv::multiply(r_rows, b_col, b_update_mask);
    b_channel = b_update_mask.mul(rb_at_g_brrb) + (1.0f - b_update_mask).mul(b_channel);
    cv::multiply(r_rows, r_col, b_update_mask);
    b_channel = b_update_mask.mul(rb_at_gr_bbrr) + (1.0f - b_update_mask).mul(b_channel);

    // Combine channels
    std::vector<cv::Mat> channels = {r_channel, g_channel, b_channel};
    cv::merge(channels, demos_out);

    return demos_out;
} 
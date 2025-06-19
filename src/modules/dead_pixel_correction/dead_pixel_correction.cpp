#include "dead_pixel_correction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>

namespace fs = std::filesystem;

DeadPixelCorrection::DeadPixelCorrection(const cv::Mat& img, const YAML::Node& platform,
                                       const YAML::Node& sensor_info, const YAML::Node& parm_dpc)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_dpc_(parm_dpc)
    , enable_(parm_dpc["is_enable"].as<bool>())
    , is_debug_(parm_dpc["is_debug"].as<bool>())
    , is_save_(parm_dpc["is_save"].as<bool>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , bayer_pattern_(sensor_info["bayer_pattern"].as<std::string>())
    , use_eigen_(true) // Use Eigen by default, can be made configurable
{
}

cv::Mat DeadPixelCorrection::correct_dead_pixels_opencv() {
    cv::Mat corrected_img = img_.clone();
    int max_value = (1 << bit_depth_) - 1;
    int threshold = parm_dpc_["threshold"].as<int>();

    // Create a mask for dead pixels (pixels with value 0 or max_value)
    cv::Mat dead_pixel_mask = (img_ == 0) | (img_ == max_value);

    // For each dead pixel, replace with the median of its neighbors
    for (int i = 1; i < img_.rows - 1; i++) {
        for (int j = 1; j < img_.cols - 1; j++) {
            if (dead_pixel_mask.at<uchar>(i, j)) {
                // Get 3x3 neighborhood
                cv::Mat neighborhood = img_(cv::Range(i-1, i+2), cv::Range(j-1, j+2));
                
                // Calculate median of non-dead pixels in neighborhood
                std::vector<uchar> values;
                for (int ni = 0; ni < 3; ni++) {
                    for (int nj = 0; nj < 3; nj++) {
                        if (!dead_pixel_mask.at<uchar>(i-1+ni, j-1+nj)) {
                            values.push_back(neighborhood.at<uchar>(ni, nj));
                        }
                    }
                }
                
                if (!values.empty()) {
                    // Sort values and get median
                    std::sort(values.begin(), values.end());
                    corrected_img.at<uchar>(i, j) = values[values.size() / 2];
                }
            }
        }
    }

    if (is_debug_) {
        int num_corrected = cv::countNonZero(dead_pixel_mask);
        std::cout << "   - Number of dead pixels corrected: " << num_corrected << std::endl;
    }

    return corrected_img;
}

hdr_isp::EigenImage DeadPixelCorrection::correct_dead_pixels_eigen(const hdr_isp::EigenImage& img) {
    hdr_isp::EigenImage corrected_img = img;
    float max_value = static_cast<float>((1 << bit_depth_) - 1);
    float threshold = static_cast<float>(parm_dpc_["threshold"].as<int>());

    // Create a mask for dead pixels (pixels with value 0 or max_value)
    hdr_isp::EigenImage dead_pixel_mask(img.rows(), img.cols());
    // Use explicit casting to avoid type mixing issues
    dead_pixel_mask.data() = ((img.data().array() == 0.0f).cast<float>() + 
                              (img.data().array() == max_value).cast<float>()).matrix();

    // Count dead pixels for debug output
    int num_dead_pixels = static_cast<int>(dead_pixel_mask.data().sum());

    // For each dead pixel, replace with the median of its neighbors
    for (int i = 1; i < img.rows() - 1; i++) {
        for (int j = 1; j < img.cols() - 1; j++) {
            if (dead_pixel_mask.data()(i, j) > 0.5f) { // Check if pixel is dead
                // Extract 3x3 neighborhood
                hdr_isp::EigenImage neighborhood(3, 3);
                neighborhood.data() = img.data().block(i-1, j-1, 3, 3);
                
                // Extract corresponding dead pixel mask for neighborhood
                hdr_isp::EigenImage neighborhood_dead_mask(3, 3);
                neighborhood_dead_mask.data() = dead_pixel_mask.data().block(i-1, j-1, 3, 3);
                
                // Calculate median of non-dead pixels in neighborhood
                float median_value = calculate_median_eigen(neighborhood, neighborhood_dead_mask);
                
                if (median_value >= 0.0f) { // Valid median found
                    corrected_img.data()(i, j) = median_value;
                }
            }
        }
    }

    if (is_debug_) {
        std::cout << "   - Number of dead pixels corrected: " << num_dead_pixels << std::endl;
    }

    return corrected_img;
}

float DeadPixelCorrection::calculate_median_eigen(const hdr_isp::EigenImage& neighborhood, const hdr_isp::EigenImage& dead_mask) {
    std::vector<float> values;
    
    // Collect non-dead pixel values from neighborhood
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (dead_mask.data()(i, j) < 0.5f) { // Not a dead pixel
                values.push_back(neighborhood.data()(i, j));
            }
        }
    }
    
    if (values.empty()) {
        return -1.0f; // No valid pixels found
    }
    
    // Sort values and get median
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void DeadPixelCorrection::save(const std::string& filename_tag) {
    if (is_save_) {
        std::string output_path = "out_frames/intermediate/" + filename_tag + 
                                 std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + ".png";
        cv::imwrite(output_path, img_);
    }
}

cv::Mat DeadPixelCorrection::execute() {
    // Save the input of dead pixel correction module
    save("Inpipeline_dead_pixel_correction_");

    // Apply dead pixel correction if enabled
    if (enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (use_eigen_) {
            // Use Eigen implementation
            if (is_debug_) {
                std::cout << "   - Using Eigen implementation for dead pixel correction" << std::endl;
            }
            
            // Convert to Eigen format
            hdr_isp::EigenImage eigen_img = hdr_isp::EigenImage::fromOpenCV(img_);
            
            // Apply dead pixel correction using Eigen
            hdr_isp::EigenImage corrected_eigen = correct_dead_pixels_eigen(eigen_img);
            
            // Convert back to OpenCV format
            img_ = corrected_eigen.toOpenCV(img_.type());
        } else {
            // Use OpenCV implementation (fallback)
            if (is_debug_) {
                std::cout << "   - Using OpenCV implementation for dead pixel correction" << std::endl;
            }
            img_ = correct_dead_pixels_opencv();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    // Save the output of dead pixel correction module
    save("Out_dead_pixel_correction_");
    return img_;
} 
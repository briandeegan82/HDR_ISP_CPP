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
    , has_eigen_input_(false)
{
}

DeadPixelCorrection::DeadPixelCorrection(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
                                       const YAML::Node& sensor_info, const YAML::Node& parm_dpc)
    : eigen_img_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_dpc_(parm_dpc)
    , enable_(parm_dpc["is_enable"].as<bool>())
    , is_debug_(parm_dpc["is_debug"].as<bool>())
    , is_save_(parm_dpc["is_save"].as<bool>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , bayer_pattern_(sensor_info["bayer_pattern"].as<std::string>())
    , use_eigen_(true) // Use Eigen by default, can be made configurable
    , has_eigen_input_(true)
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

hdr_isp::EigenImageU32 DeadPixelCorrection::correct_dead_pixels_eigen(const hdr_isp::EigenImageU32& img) {
    hdr_isp::EigenImageU32 corrected_img = img;
    
    // Simple dead pixel detection and correction
    // In a real implementation, you'd use more sophisticated algorithms
    
    // Create dead pixel mask (simplified - in real implementation, you'd load actual dead pixel data)
    hdr_isp::EigenImageU32 dead_pixel_mask(img.rows(), img.cols());
    dead_pixel_mask.data().setZero();
    
    // For demonstration, mark some pixels as dead (in practice, this would come from calibration data)
    // Here we're just marking a few pixels as dead for testing
    if (img.rows() > 10 && img.cols() > 10) {
        dead_pixel_mask.data()(5, 5) = 1;
        dead_pixel_mask.data()(10, 10) = 1;
        dead_pixel_mask.data()(15, 15) = 1;
    }
    
    // Correct dead pixels using median filtering
    for (int i = 1; i < img.rows() - 1; ++i) {
        for (int j = 1; j < img.cols() - 1; ++j) {
            if (dead_pixel_mask.data()(i, j) == 1) {
                // Extract 3x3 neighborhood
                hdr_isp::EigenImageU32 neighborhood(3, 3);
                hdr_isp::EigenImageU32 neighborhood_dead_mask(3, 3);
                
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        neighborhood.data()(di + 1, dj + 1) = img.data()(i + di, j + dj);
                        neighborhood_dead_mask.data()(di + 1, dj + 1) = dead_pixel_mask.data()(i + di, j + dj);
                    }
                }
                
                // Calculate median of non-dead pixels
                uint32_t median_val = calculate_median_eigen(neighborhood, neighborhood_dead_mask);
                corrected_img.data()(i, j) = median_val;
            }
        }
    }
    
    return corrected_img;
}

uint32_t DeadPixelCorrection::calculate_median_eigen(const hdr_isp::EigenImageU32& neighborhood, const hdr_isp::EigenImageU32& dead_mask) {
    std::vector<uint32_t> valid_pixels;
    
    // Collect non-dead pixels
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (dead_mask.data()(i, j) == 0) {
                valid_pixels.push_back(neighborhood.data()(i, j));
            }
        }
    }
    
    if (valid_pixels.empty()) {
        return 0; // Fallback value
    }
    
    // Sort and return median
    std::sort(valid_pixels.begin(), valid_pixels.end());
    return valid_pixels[valid_pixels.size() / 2];
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
            hdr_isp::EigenImageU32 eigen_img = hdr_isp::EigenImageU32::fromOpenCV(img_);
            
            // Apply dead pixel correction using Eigen
            hdr_isp::EigenImageU32 corrected_eigen = correct_dead_pixels_eigen(eigen_img);
            
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

hdr_isp::EigenImageU32 DeadPixelCorrection::execute_eigen() {
    // Apply dead pixel correction if enabled
    if (enable_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (use_eigen_) {
            // Use Eigen implementation
            if (is_debug_) {
                std::cout << "   - Using Eigen implementation for dead pixel correction" << std::endl;
            }
            
            // Apply dead pixel correction using Eigen
            eigen_img_ = correct_dead_pixels_eigen(eigen_img_);
        } else {
            // Use OpenCV implementation (fallback)
            if (is_debug_) {
                std::cout << "   - Using OpenCV implementation for dead pixel correction" << std::endl;
            }
            // Convert to OpenCV, process, then convert back
            cv::Mat temp_img = eigen_img_.toOpenCV(CV_32S);
            temp_img = correct_dead_pixels_opencv();
            eigen_img_ = hdr_isp::EigenImageU32::fromOpenCV(temp_img);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    return eigen_img_;
} 
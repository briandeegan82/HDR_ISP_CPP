#include "crop.hpp"
#include <chrono>
#include <iostream>
#include <regex>
#include <filesystem>

namespace fs = std::filesystem;

Crop::Crop(const cv::Mat& img, const YAML::Node& platform,
           const YAML::Node& sensor_info, const YAML::Node& parm_cro)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_cro_(parm_cro)
    , old_size_(sensor_info["height"].as<int>(), sensor_info["width"].as<int>())
    , new_size_(parm_cro["new_height"].as<int>(), parm_cro["new_width"].as<int>())
    , enable_(parm_cro["is_enable"].as<bool>())
    , is_debug_(parm_cro["is_debug"].as<bool>())
    , is_save_(parm_cro["is_save"].as<bool>())
    , use_eigen_(true) // Use Eigen by default, can be made configurable
{
    update_sensor_info(sensor_info_);
}

void Crop::update_sensor_info(YAML::Node& dictionary) {
    if (enable_) {
        if (dictionary["height"].as<int>() != new_size_.first ||
            dictionary["width"].as<int>() != new_size_.second) {
            dictionary["height"] = new_size_.first;
            dictionary["width"] = new_size_.second;
            dictionary["orig_size"] = std::to_string(img_.cols) + "x" + std::to_string(img_.rows);
        }
    }
}

hdr_isp::EigenImage32 Crop::crop_eigen(const hdr_isp::EigenImage32& img, int rows_to_crop, int cols_to_crop) {
    if (rows_to_crop || cols_to_crop) {
        if (rows_to_crop % 4 == 0 && cols_to_crop % 4 == 0) {
            int start_row = rows_to_crop / 2;
            int end_row = img.rows() - rows_to_crop / 2;
            int start_col = cols_to_crop / 2;
            int end_col = img.cols() - cols_to_crop / 2;
            
            // Use Eigen's block operation for cropping
            Eigen::MatrixXi cropped_data = img.data().block(start_row, start_col, 
                                                           end_row - start_row, 
                                                           end_col - start_col);
            return hdr_isp::EigenImage32(cropped_data);
        } else {
            std::cout << "   - Input/Output heights are not compatible."
                      << " Bayer pattern will be disturbed if cropped!" << std::endl;
        }
    }
    return img;
}

cv::Mat Crop::crop_opencv(const cv::Mat& img, int rows_to_crop, int cols_to_crop) {
    if (rows_to_crop || cols_to_crop) {
        if (rows_to_crop % 4 == 0 && cols_to_crop % 4 == 0) {
            int start_row = rows_to_crop / 2;
            int end_row = img.rows - rows_to_crop / 2;
            int start_col = cols_to_crop / 2;
            int end_col = img.cols - cols_to_crop / 2;
            return img(cv::Range(start_row, end_row), cv::Range(start_col, end_col));
        } else {
            std::cout << "   - Input/Output heights are not compatible."
                      << " Bayer pattern will be disturbed if cropped!" << std::endl;
        }
    }
    return img;
}

cv::Mat Crop::apply_cropping() {
    if (old_size_ == new_size_) {
        return img_;
    }

    if (old_size_.first < new_size_.first || old_size_.second < new_size_.second) {
        std::cout << "   - Invalid output size " << new_size_.first << "x" << new_size_.second << std::endl;
        std::cout << "   - Make sure output size is smaller than input size!" << std::endl;
        return img_;
    }

    int crop_rows = old_size_.first - new_size_.first;
    int crop_cols = old_size_.second - new_size_.second;
    cv::Mat cropped_img;

    if (use_eigen_) {
        // Use Eigen implementation
        if (is_debug_) {
            std::cout << "   - Using Eigen implementation for cropping" << std::endl;
        }
        
        // Convert to Eigen format
        hdr_isp::EigenImage32 eigen_img = hdr_isp::EigenImage32::fromOpenCV(img_);
        
        // Apply cropping using Eigen
        hdr_isp::EigenImage32 cropped_eigen = crop_eigen(eigen_img, crop_rows, crop_cols);
        
        // Convert back to OpenCV format
        cropped_img = cropped_eigen.toOpenCV(img_.type());
    } else {
        // Use OpenCV implementation (fallback)
        if (is_debug_) {
            std::cout << "   - Using OpenCV implementation for cropping" << std::endl;
        }
        cropped_img = crop_opencv(img_, crop_rows, crop_cols);
    }

    if (is_debug_) {
        std::cout << "   - Number of rows cropped = " << crop_rows << std::endl;
        std::cout << "   - Number of columns cropped = " << crop_cols << std::endl;
        std::cout << "   - Shape of cropped image = " << cropped_img.size() << std::endl;
    }
    return cropped_img;
}

void Crop::save(const std::string& filename_tag) {
    // Update size of array in filename
    std::string in_file = platform_["in_file"].as<std::string>();
    std::regex size_pattern(R"(\d+x\d+)");
    std::string new_size = std::to_string(img_.cols) + "x" + std::to_string(img_.rows);
    in_file = std::regex_replace(in_file, size_pattern, new_size);
    platform_["in_file"] = in_file;

    if (is_save_) {
        std::string output_path = "out_frames/intermediate/" + filename_tag + 
                                 std::to_string(img_.cols) + "x" + std::to_string(img_.rows) + ".png";
        cv::imwrite(output_path, img_);
    }
}

cv::Mat Crop::execute() {
    // Save the input of crop module
    save("Inpipeline_crop_");

    // Crop image if enabled
    if (enable_) {
        std::cout << "line 97" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        img_ = apply_cropping();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (is_debug_) {
            std::cout << "  Execution time: " << duration.count() / 1000.0 << "s" << std::endl;
        }
    }

    // Save the output of crop module
    save("Out_crop_");
    return img_;
} 
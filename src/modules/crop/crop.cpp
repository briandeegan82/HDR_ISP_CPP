#include "crop.hpp"
#include <chrono>
#include <iostream>
#include <regex>
#include <filesystem>

namespace fs = std::filesystem;

Crop::Crop(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
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
{
    update_sensor_info(sensor_info_);
}

void Crop::update_sensor_info(YAML::Node& dictionary) {
    if (enable_) {
        if (dictionary["height"].as<int>() != new_size_.first ||
            dictionary["width"].as<int>() != new_size_.second) {
            dictionary["height"] = new_size_.first;
            dictionary["width"] = new_size_.second;
            dictionary["orig_size"] = std::to_string(img_.cols()) + "x" + std::to_string(img_.rows());
        }
    }
}

hdr_isp::EigenImageU32 Crop::crop_eigen(const hdr_isp::EigenImageU32& img, int rows_to_crop, int cols_to_crop) {
    int new_rows = img.rows() - rows_to_crop;
    int new_cols = img.cols() - cols_to_crop;
    
    if (new_rows <= 0 || new_cols <= 0) {
        throw std::runtime_error("Crop dimensions would result in zero or negative size");
    }
    
    // Extract the cropped region
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> cropped_data = 
        img.data().block(rows_to_crop/2, cols_to_crop/2, new_rows, new_cols);
    
    return hdr_isp::EigenImageU32(cropped_data);
}

hdr_isp::EigenImageU32 Crop::apply_cropping() {
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

    if (is_debug_) {
        std::cout << "   - Using Eigen implementation for cropping" << std::endl;
    }
    
    // Apply cropping using Eigen
    hdr_isp::EigenImageU32 cropped_eigen = crop_eigen(img_, crop_rows, crop_cols);

    if (is_debug_) {
        std::cout << "   - Number of rows cropped = " << crop_rows << std::endl;
        std::cout << "   - Number of columns cropped = " << crop_cols << std::endl;
        std::cout << "   - Shape of cropped image = " << cropped_eigen.cols() << "x" << cropped_eigen.rows() << std::endl;
    }
    return cropped_eigen;
}

void Crop::save(const std::string& filename_tag) {
    // Update size of array in filename
    std::string in_file = platform_["in_file"].as<std::string>();
    std::regex size_pattern(R"(\d+x\d+)");
    std::string new_size = std::to_string(img_.cols()) + "x" + std::to_string(img_.rows());
    in_file = std::regex_replace(in_file, size_pattern, new_size);
    platform_["in_file"] = in_file;

    if (is_save_) {
        std::string output_path = "out_frames/intermediate/" + filename_tag + 
                                 std::to_string(img_.cols()) + "x" + std::to_string(img_.rows()) + ".png";
        // Convert to OpenCV for saving
        cv::Mat save_img = img_.toOpenCV(CV_32S);
        cv::imwrite(output_path, save_img);
    }
}

hdr_isp::EigenImageU32 Crop::execute() {
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
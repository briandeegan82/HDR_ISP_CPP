#include "image_saver.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>

namespace image_saver {

// Helper function to convert Eigen matrix to OpenCV Mat for saving
cv::Mat eigen_to_opencv_mat(const hdr_isp::EigenImage& img) {
    const Eigen::MatrixXf& data = img.data();
    cv::Mat cv_mat(data.rows(), data.cols(), CV_32F);
    
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            cv_mat.at<float>(i, j) = data(i, j);
        }
    }
    
    return cv_mat;
}

// Helper function to convert Eigen 3C image to OpenCV Mat for saving
cv::Mat eigen_3c_to_opencv_mat(const hdr_isp::EigenImage3C& img) {
    int rows = img.rows();
    int cols = img.cols();
    cv::Mat cv_mat(rows, cols, CV_32FC3);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cv::Vec3f& pixel = cv_mat.at<cv::Vec3f>(i, j);
            pixel[0] = img.r()(i, j);
            pixel[1] = img.g()(i, j);
            pixel[2] = img.b()(i, j);
        }
    }
    
    return cv_mat;
}

// Helper function to convert fixed-point Eigen matrix to OpenCV Mat for saving
cv::Mat eigen_fixed_to_opencv_mat(const hdr_isp::EigenImageFixed& img, int fractional_bits) {
    const Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>& data = img.data();
    cv::Mat cv_mat(data.rows(), data.cols(), CV_32F);
    
    float scale = 1.0f / (1 << fractional_bits);
    
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            cv_mat.at<float>(i, j) = static_cast<float>(data(i, j)) * scale;
        }
    }
    
    return cv_mat;
}

// Helper function to convert fixed-point Eigen 3C image to OpenCV Mat for saving
cv::Mat eigen_3c_fixed_to_opencv_mat(const hdr_isp::EigenImage3CFixed& img, int fractional_bits) {
    int rows = img.rows();
    int cols = img.cols();
    cv::Mat cv_mat(rows, cols, CV_32FC3);
    
    float scale = 1.0f / (1 << fractional_bits);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cv::Vec3f& pixel = cv_mat.at<cv::Vec3f>(i, j);
            pixel[0] = static_cast<float>(img.r()(i, j)) * scale;
            pixel[1] = static_cast<float>(img.g()(i, j)) * scale;
            pixel[2] = static_cast<float>(img.b()(i, j)) * scale;
        }
    }
    
    return cv_mat;
}

void save_eigen_image(const hdr_isp::EigenImage& img, const std::string& output_path) {
    cv::Mat cv_mat = eigen_to_opencv_mat(img);
    
    // Normalize to 0-255 range for saving
    cv::Mat normalized;
    cv::normalize(cv_mat, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
    cv::imwrite(output_path, normalized);
}

void save_eigen_image_3c(const hdr_isp::EigenImage3C& img, const std::string& output_path) {
    cv::Mat cv_mat = eigen_3c_to_opencv_mat(img);
    
    // Normalize to 0-255 range for saving
    cv::Mat normalized;
    cv::normalize(cv_mat, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8UC3);
    
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
    cv::imwrite(output_path, normalized);
}

void save_eigen_image_fixed(const hdr_isp::EigenImageFixed& img, const std::string& output_path, int fractional_bits) {
    cv::Mat cv_mat = eigen_fixed_to_opencv_mat(img, fractional_bits);
    
    // Normalize to 0-255 range for saving
    cv::Mat normalized;
    cv::normalize(cv_mat, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
    cv::imwrite(output_path, normalized);
}

void save_eigen_image_3c_fixed(const hdr_isp::EigenImage3CFixed& img, const std::string& output_path, int fractional_bits) {
    cv::Mat cv_mat = eigen_3c_fixed_to_opencv_mat(img, fractional_bits);
    
    // Normalize to 0-255 range for saving
    cv::Mat normalized;
    cv::normalize(cv_mat, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8UC3);
    
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
    cv::imwrite(output_path, normalized);
}

void save_pipeline_output(const std::string& img_name, const hdr_isp::EigenImage3C& output_img, const YAML::Node& config_file) {
    std::string output_dir = "out_frames/";
    if (config_file["output_dir"].IsDefined()) {
        output_dir = config_file["output_dir"].as<std::string>();
    }
    
    std::filesystem::create_directories(output_dir);
    std::string output_path = output_dir + "output_" + img_name + ".png";
    
    save_eigen_image_3c(output_img, output_path);
    std::cout << "Pipeline output saved to: " << output_path << std::endl;
}

void save_pipeline_output_fixed(const std::string& img_name, const hdr_isp::EigenImage3CFixed& output_img, const YAML::Node& config_file, int fractional_bits) {
    std::string output_dir = "out_frames/";
    if (config_file["output_dir"].IsDefined()) {
        output_dir = config_file["output_dir"].as<std::string>();
    }
    
    std::filesystem::create_directories(output_dir);
    std::string output_path = output_dir + "output_" + img_name + ".png";
    
    save_eigen_image_3c_fixed(output_img, output_path, fractional_bits);
    std::cout << "Pipeline output saved to: " << output_path << std::endl;
}

void save_output_array(const std::string& img_name, const hdr_isp::EigenImage& output_array, const std::string& module_name,
                      const YAML::Node& platform, int bitdepth, const std::string& bayer_pattern) {
    std::string output_dir = "./module_output/";
    if (platform["output_dir"].IsDefined()) {
        output_dir = platform["output_dir"].as<std::string>();
    }
    
    std::filesystem::create_directories(output_dir);
    std::string filename = output_dir + img_name + "_" + module_name + "_" + std::to_string(bitdepth) + "bit_" + bayer_pattern;
    
    save_eigen_image(output_array, filename + ".png");
}

void save_output_array_3c(const std::string& img_name, const hdr_isp::EigenImage3C& output_array, const std::string& module_name,
                         const YAML::Node& platform, int bitdepth, const std::string& bayer_pattern) {
    std::string output_dir = "./module_output/";
    if (platform["output_dir"].IsDefined()) {
        output_dir = platform["output_dir"].as<std::string>();
    }
    
    std::filesystem::create_directories(output_dir);
    std::string filename = output_dir + img_name + "_" + module_name + "_" + std::to_string(bitdepth) + "bit_" + bayer_pattern;
    
    save_eigen_image_3c(output_array, filename + ".png");
}

void save_yuv_format(const std::string& img_name, const hdr_isp::EigenImage3C& yuv_array, const std::string& module_name,
                    const YAML::Node& platform, const std::string& conv_std) {
    std::string output_dir = "./module_output/";
    if (platform["output_dir"].IsDefined()) {
        output_dir = platform["output_dir"].as<std::string>();
    }
    
    std::filesystem::create_directories(output_dir);
    std::string filename = output_dir + img_name + "_" + module_name + "_" + conv_std;
    
    save_eigen_image_3c(yuv_array, filename + ".yuv");
}

} // namespace image_saver 
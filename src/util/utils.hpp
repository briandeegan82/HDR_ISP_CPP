#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <regex>
#include <random>
#include <yaml-cpp/yaml.h>

namespace utils {

// Constants
const std::string OUTPUT_DIR = "out_frames/";
const std::string OUTPUT_ARRAY_DIR = "./module_output/";

// Image processing utilities
cv::Mat introduce_defect(const cv::Mat& img, int total_defective_pixels, bool padding);
cv::Mat gauss_kern_raw(int size, float std_dev, float stride);
cv::Mat crop(const cv::Mat& img, int rows_to_crop = 0, int cols_to_crop = 0);
cv::Mat stride_convolve2d(const cv::Mat& matrix, const cv::Mat& kernel);

// YUV conversion utilities
cv::Mat reconstruct_yuv_from_422_custom(const cv::Mat& yuv_422_custom, int width, int height);
cv::Mat reconstruct_yuv_from_444_custom(const cv::Mat& yuv_444_custom, int width, int height);
cv::Mat get_image_from_yuv_format_conversion(const cv::Mat& yuv_img, int height, int width, const std::string& yuv_custom_format);
cv::Mat yuv_to_rgb(const cv::Mat& yuv_img, const std::string& conv_std);

// Bayer pattern utilities
std::vector<cv::Mat> masks_cfa_bayer(const cv::Mat& img, const std::string& bayer);
cv::Mat apply_cfa(const cv::Mat& img, int bit_depth, const std::string& bayer);

// File handling utilities
void save_pipeline_output(const std::string& img_name, const cv::Mat& output_img, const YAML::Node& config_file);
void save_output_array(const std::string& img_name, const cv::Mat& output_array, const std::string& module_name,
                      const YAML::Node& platform, int bitdepth, const std::string& bayer_pattern);
void save_output_array_yuv(const std::string& img_name, const cv::Mat& output_array, const std::string& module_name,
                          const YAML::Node& platform, const std::string& conv_std);
void save_image(const cv::Mat& img, const std::string& output_path);

// Configuration utilities
std::vector<std::string> parse_file_name(const std::string& filename);
std::vector<std::string> extract_raw_metadata(const std::string& filename);

// Display utilities
void display_ae_statistics(float ae_feedback, const std::vector<float>& awb_gains);

} // namespace utils 
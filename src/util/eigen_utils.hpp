#pragma once

#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "../common/eigen_utils.hpp"

namespace eigen_utils {

// Constants
const std::string OUTPUT_DIR = "out_frames/";
const std::string OUTPUT_ARRAY_DIR = "./module_output/";

// Image processing utilities using Eigen
hdr_isp::EigenImage introduce_defect(const hdr_isp::EigenImage& img, int total_defective_pixels, bool padding);
hdr_isp::EigenImage gauss_kern_raw(int size, float std_dev, float stride);
hdr_isp::EigenImage crop(const hdr_isp::EigenImage& img, int rows_to_crop = 0, int cols_to_crop = 0);
hdr_isp::EigenImage stride_convolve2d(const hdr_isp::EigenImage& matrix, const hdr_isp::EigenImage& kernel);

// YUV conversion utilities using Eigen
hdr_isp::EigenImage3C reconstruct_yuv_from_422_custom(const hdr_isp::EigenImage& yuv_422_custom, int width, int height);
hdr_isp::EigenImage3C reconstruct_yuv_from_444_custom(const hdr_isp::EigenImage& yuv_444_custom, int width, int height);
hdr_isp::EigenImage3C get_image_from_yuv_format_conversion(const hdr_isp::EigenImage& yuv_img, int height, int width, const std::string& yuv_custom_format);
hdr_isp::EigenImage3C yuv_to_rgb(const hdr_isp::EigenImage3C& yuv_img, const std::string& conv_std);

// Bayer pattern utilities using Eigen
std::vector<hdr_isp::EigenImage> masks_cfa_bayer(const hdr_isp::EigenImage& img, const std::string& bayer);
hdr_isp::EigenImage apply_cfa(const hdr_isp::EigenImage& img, int bit_depth, const std::string& bayer);

// Configuration utilities
std::vector<std::string> parse_file_name(const std::string& filename);
std::vector<std::string> extract_raw_metadata(const std::string& filename);

// Display utilities
void display_ae_statistics(float ae_feedback, const std::vector<float>& awb_gains);

// Matrix operations
hdr_isp::EigenImage normalize_matrix(const hdr_isp::EigenImage& img, float min_val = 0.0f, float max_val = 255.0f);
hdr_isp::EigenImage3C normalize_matrix_3c(const hdr_isp::EigenImage3C& img, float min_val = 0.0f, float max_val = 255.0f);

// Filtering operations
hdr_isp::EigenImage apply_filter_2d(const hdr_isp::EigenImage& img, const hdr_isp::EigenImage& kernel);
hdr_isp::EigenImage apply_gaussian_filter(const hdr_isp::EigenImage& img, float sigma, int kernel_size = 5);

// Color space conversions
hdr_isp::EigenImage3C rgb_to_yuv(const hdr_isp::EigenImage3C& rgb, const std::string& standard = "BT709");
hdr_isp::EigenImage3C yuv_to_rgb(const hdr_isp::EigenImage3C& yuv, const std::string& standard = "BT709");

} // namespace eigen_utils 
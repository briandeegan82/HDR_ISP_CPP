#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "../common/eigen_utils.hpp"

// Forward declaration - OpenCV headers only included in the implementation
struct cv_Mat;

namespace image_saver {

// Save Eigen-based images using OpenCV (only for saving)
void save_eigen_image(const hdr_isp::EigenImage& img, const std::string& output_path);
void save_eigen_image_3c(const hdr_isp::EigenImage3C& img, const std::string& output_path);
void save_eigen_image_fixed(const hdr_isp::EigenImageFixed& img, const std::string& output_path, int fractional_bits = 8);
void save_eigen_image_3c_fixed(const hdr_isp::EigenImage3CFixed& img, const std::string& output_path, int fractional_bits = 8);

// Save pipeline output with configuration
void save_pipeline_output(const std::string& img_name, const hdr_isp::EigenImage3C& output_img, const YAML::Node& config_file);
void save_pipeline_output_fixed(const std::string& img_name, const hdr_isp::EigenImage3CFixed& output_img, const YAML::Node& config_file, int fractional_bits = 8);

// Save intermediate module outputs
void save_output_array(const std::string& img_name, const hdr_isp::EigenImage& output_array, const std::string& module_name,
                      const YAML::Node& platform, int bitdepth, const std::string& bayer_pattern);
void save_output_array_3c(const std::string& img_name, const hdr_isp::EigenImage3C& output_array, const std::string& module_name,
                         const YAML::Node& platform, int bitdepth, const std::string& bayer_pattern);

// YUV format saving
void save_yuv_format(const std::string& img_name, const hdr_isp::EigenImage3C& yuv_array, const std::string& module_name,
                    const YAML::Node& platform, const std::string& conv_std);

} // namespace image_saver 
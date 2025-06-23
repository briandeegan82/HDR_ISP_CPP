#pragma once

#include <Halide.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "../../common/eigen_utils.hpp"

namespace hdr_isp {

/**
 * @brief Utility functions for Halide integration
 * 
 * This file provides conversion functions between Eigen, OpenCV, and Halide buffers
 * to support the hybrid pipeline integration.
 */

/**
 * @brief Convert EigenImageU32 to Halide Buffer
 * 
 * @param eigen_img Input Eigen image
 * @return Halide::Buffer<uint32_t> Halide buffer
 */
Halide::Buffer<uint32_t> eigenToHalide(const EigenImageU32& eigen_img);

/**
 * @brief Convert Halide Buffer to EigenImageU32
 * 
 * @param halide_buffer Input Halide buffer
 * @return EigenImageU32 Eigen image
 */
EigenImageU32 halideToEigen(const Halide::Buffer<uint32_t>& halide_buffer);

/**
 * @brief Convert EigenImage3C to Halide Buffer
 * 
 * @param eigen_img Input Eigen 3-channel image
 * @return Halide::Buffer<float> Halide buffer (3 channels)
 */
Halide::Buffer<float> eigen3CToHalide(const EigenImage3C& eigen_img);

/**
 * @brief Convert Halide Buffer to EigenImage3C
 * 
 * @param halide_buffer Input Halide buffer (3 channels)
 * @return EigenImage3C Eigen 3-channel image
 */
EigenImage3C halideToEigen3C(const Halide::Buffer<float>& halide_buffer);

/**
 * @brief Convert OpenCV Mat to Halide Buffer
 * 
 * @param cv_mat Input OpenCV matrix
 * @return Halide::Buffer<float> Halide buffer
 */
Halide::Buffer<float> opencvToHalide(const cv::Mat& cv_mat);

/**
 * @brief Convert Halide Buffer to OpenCV Mat
 * 
 * @param halide_buffer Input Halide buffer
 * @return cv::Mat OpenCV matrix
 */
cv::Mat halideToOpencv(const Halide::Buffer<float>& halide_buffer);

/**
 * @brief Convert OpenCV Mat to Halide Buffer (uint32_t)
 * 
 * @param cv_mat Input OpenCV matrix
 * @return Halide::Buffer<uint32_t> Halide buffer
 */
Halide::Buffer<uint32_t> opencvToHalideU32(const cv::Mat& cv_mat);

/**
 * @brief Convert Halide Buffer to OpenCV Mat (uint32_t)
 * 
 * @param halide_buffer Input Halide buffer
 * @return cv::Mat OpenCV matrix
 */
cv::Mat halideToOpencvU32(const Halide::Buffer<uint32_t>& halide_buffer);

/**
 * @brief Create a test Halide buffer with specified size and pattern
 * 
 * @param width Buffer width
 * @param height Buffer height
 * @param pattern Pattern type ("gradient", "constant", "random")
 * @param value Constant value (for "constant" pattern)
 * @return Halide::Buffer<uint32_t> Test buffer
 */
Halide::Buffer<uint32_t> createTestHalideBuffer(int width, int height, 
                                               const std::string& pattern = "gradient",
                                               uint32_t value = 1000);

/**
 * @brief Compare two Halide buffers for equality
 * 
 * @param buffer1 First buffer
 * @param buffer2 Second buffer
 * @param tolerance Tolerance for floating-point comparison
 * @return true if buffers are equal within tolerance
 */
bool compareHalideBuffers(const Halide::Buffer<uint32_t>& buffer1,
                         const Halide::Buffer<uint32_t>& buffer2,
                         uint32_t tolerance = 0);

/**
 * @brief Compare two Halide buffers for equality (float)
 * 
 * @param buffer1 First buffer
 * @param buffer2 Second buffer
 * @param tolerance Tolerance for floating-point comparison
 * @return true if buffers are equal within tolerance
 */
bool compareHalideBuffers(const Halide::Buffer<float>& buffer1,
                         const Halide::Buffer<float>& buffer2,
                         float tolerance = 1e-6f);

/**
 * @brief Print Halide buffer statistics
 * 
 * @param buffer Input buffer
 * @param name Buffer name for output
 */
void printHalideBufferStats(const Halide::Buffer<uint32_t>& buffer, 
                           const std::string& name = "Buffer");

/**
 * @brief Print Halide buffer statistics (float)
 * 
 * @param buffer Input buffer
 * @param name Buffer name for output
 */
void printHalideBufferStats(const Halide::Buffer<float>& buffer, 
                           const std::string& name = "Buffer");

} // namespace hdr_isp 
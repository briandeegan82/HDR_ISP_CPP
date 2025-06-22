#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "src/common/fixed_point_utils.hpp"

// Test program to compare 8-bit vs 16-bit fixed-point performance
void test_fixed_point_performance() {
    std::cout << "=== Fixed-Point Performance Test ===" << std::endl;
    
    // Test CCM matrix values
    std::vector<float> corrected_red = {1.660f, -0.527f, -0.133f};
    std::vector<float> corrected_green = {-0.408f, 1.563f, -0.082f};
    std::vector<float> corrected_blue = {-0.055f, -1.641f, 2.695f};
    
    // Create test CCM matrix
    Eigen::Matrix3f ccm_mat;
    ccm_mat.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data());
    ccm_mat.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data());
    ccm_mat.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data());
    
    std::cout << "CCM Matrix:" << std::endl;
    std::cout << ccm_mat << std::endl;
    
    // Test input RGB values (simulating demosaic output)
    int test_width = 1920;
    int test_height = 1080;
    Eigen::MatrixXf test_r = Eigen::MatrixXf::Random(test_height, test_width) * 0.5f + 0.5f; // 0-1 range
    Eigen::MatrixXf test_g = Eigen::MatrixXf::Random(test_height, test_width) * 0.5f + 0.5f;
    Eigen::MatrixXf test_b = Eigen::MatrixXf::Random(test_height, test_width) * 0.5f + 0.5f;
    
    std::cout << "Test image size: " << test_width << "x" << test_height << std::endl;
    std::cout << "Input range: 0.0 - 1.0" << std::endl;
    
    // 1. Floating-point reference
    std::cout << "\n--- Floating-Point Reference ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    Eigen::MatrixXf result_r_float = ccm_mat(0, 0) * test_r + ccm_mat(0, 1) * test_g + ccm_mat(0, 2) * test_b;
    Eigen::MatrixXf result_g_float = ccm_mat(1, 0) * test_r + ccm_mat(1, 1) * test_g + ccm_mat(1, 2) * test_b;
    Eigen::MatrixXf result_b_float = ccm_mat(2, 0) * test_r + ccm_mat(2, 1) * test_g + ccm_mat(2, 2) * test_b;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_float = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Floating-point execution time: " << duration_float.count() << " microseconds" << std::endl;
    
    // 2. 8-bit fixed-point (6 fractional bits)
    std::cout << "\n--- 8-bit Fixed-Point (6 fractional bits) ---" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    int fractional_bits_8bit = 6;
    Eigen::Matrix3i ccm_mat_8bit = hdr_isp::FixedPointUtils::applyFixedPointScaling<int8_t>(ccm_mat, fractional_bits_8bit);
    
    Eigen::MatrixXf result_r_8bit(test_height, test_width);
    Eigen::MatrixXf result_g_8bit(test_height, test_width);
    Eigen::MatrixXf result_b_8bit(test_height, test_width);
    
    for (int i = 0; i < test_height; ++i) {
        for (int j = 0; j < test_width; ++j) {
            // Convert to fixed-point
            int8_t r_fixed = hdr_isp::FixedPointUtils::floatToFixed<int8_t>(test_r(i, j), fractional_bits_8bit);
            int8_t g_fixed = hdr_isp::FixedPointUtils::floatToFixed<int8_t>(test_g(i, j), fractional_bits_8bit);
            int8_t b_fixed = hdr_isp::FixedPointUtils::floatToFixed<int8_t>(test_b(i, j), fractional_bits_8bit);
            
            // Apply matrix multiplication
            int32_t out_r = ccm_mat_8bit(0, 0) * r_fixed + ccm_mat_8bit(0, 1) * g_fixed + ccm_mat_8bit(0, 2) * b_fixed;
            int32_t out_g = ccm_mat_8bit(1, 0) * r_fixed + ccm_mat_8bit(1, 1) * g_fixed + ccm_mat_8bit(1, 2) * b_fixed;
            int32_t out_b = ccm_mat_8bit(2, 0) * r_fixed + ccm_mat_8bit(2, 1) * g_fixed + ccm_mat_8bit(2, 2) * b_fixed;
            
            // Convert back to float
            result_r_8bit(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int32_t>(out_r, fractional_bits_8bit);
            result_g_8bit(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int32_t>(out_g, fractional_bits_8bit);
            result_b_8bit(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int32_t>(out_b, fractional_bits_8bit);
        }
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto duration_8bit = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "8-bit fixed-point execution time: " << duration_8bit.count() << " microseconds" << std::endl;
    std::cout << "Speedup vs float: " << static_cast<float>(duration_float.count()) / duration_8bit.count() << "x" << std::endl;
    
    // 3. 16-bit fixed-point (12 fractional bits)
    std::cout << "\n--- 16-bit Fixed-Point (12 fractional bits) ---" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    int fractional_bits_16bit = 12;
    Eigen::Matrix3i ccm_mat_16bit = hdr_isp::FixedPointUtils::applyFixedPointScaling<int16_t>(ccm_mat, fractional_bits_16bit);
    
    Eigen::MatrixXf result_r_16bit(test_height, test_width);
    Eigen::MatrixXf result_g_16bit(test_height, test_width);
    Eigen::MatrixXf result_b_16bit(test_height, test_width);
    
    for (int i = 0; i < test_height; ++i) {
        for (int j = 0; j < test_width; ++j) {
            // Convert to fixed-point
            int16_t r_fixed = hdr_isp::FixedPointUtils::floatToFixed<int16_t>(test_r(i, j), fractional_bits_16bit);
            int16_t g_fixed = hdr_isp::FixedPointUtils::floatToFixed<int16_t>(test_g(i, j), fractional_bits_16bit);
            int16_t b_fixed = hdr_isp::FixedPointUtils::floatToFixed<int16_t>(test_b(i, j), fractional_bits_16bit);
            
            // Apply matrix multiplication
            int64_t out_r = ccm_mat_16bit(0, 0) * r_fixed + ccm_mat_16bit(0, 1) * g_fixed + ccm_mat_16bit(0, 2) * b_fixed;
            int64_t out_g = ccm_mat_16bit(1, 0) * r_fixed + ccm_mat_16bit(1, 1) * g_fixed + ccm_mat_16bit(1, 2) * b_fixed;
            int64_t out_b = ccm_mat_16bit(2, 0) * r_fixed + ccm_mat_16bit(2, 1) * g_fixed + ccm_mat_16bit(2, 2) * b_fixed;
            
            // Convert back to float
            result_r_16bit(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int64_t>(out_r, fractional_bits_16bit);
            result_g_16bit(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int64_t>(out_g, fractional_bits_16bit);
            result_b_16bit(i, j) = hdr_isp::FixedPointUtils::fixedToFloat<int64_t>(out_b, fractional_bits_16bit);
        }
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto duration_16bit = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "16-bit fixed-point execution time: " << duration_16bit.count() << " microseconds" << std::endl;
    std::cout << "Speedup vs float: " << static_cast<float>(duration_float.count()) / duration_16bit.count() << "x" << std::endl;
    
    // 4. Accuracy comparison
    std::cout << "\n--- Accuracy Comparison ---" << std::endl;
    
    // Calculate RMS error
    float rms_error_8bit = std::sqrt(((result_r_8bit - result_r_float).array().square() + 
                                     (result_g_8bit - result_g_float).array().square() + 
                                     (result_b_8bit - result_b_float).array().square()).mean());
    
    float rms_error_16bit = std::sqrt(((result_r_16bit - result_r_float).array().square() + 
                                      (result_g_16bit - result_g_float).array().square() + 
                                      (result_b_16bit - result_b_float).array().square()).mean());
    
    std::cout << "8-bit RMS error: " << rms_error_8bit << std::endl;
    std::cout << "16-bit RMS error: " << rms_error_16bit << std::endl;
    std::cout << "16-bit vs 8-bit accuracy improvement: " << rms_error_8bit / rms_error_16bit << "x" << std::endl;
    
    // 5. Memory usage comparison
    std::cout << "\n--- Memory Usage Comparison ---" << std::endl;
    size_t float_memory = test_width * test_height * 3 * sizeof(float);
    size_t int8_memory = test_width * test_height * 3 * sizeof(int8_t);
    size_t int16_memory = test_width * test_height * 3 * sizeof(int16_t);
    
    std::cout << "Float memory: " << float_memory << " bytes" << std::endl;
    std::cout << "8-bit memory: " << int8_memory << " bytes (" << static_cast<float>(int8_memory) / float_memory << "x smaller)" << std::endl;
    std::cout << "16-bit memory: " << int16_memory << " bytes (" << static_cast<float>(int16_memory) / float_memory << "x smaller)" << std::endl;
    
    // 6. Summary
    std::cout << "\n--- Summary ---" << std::endl;
    std::cout << "8-bit fixed-point:" << std::endl;
    std::cout << "  Speedup: " << static_cast<float>(duration_float.count()) / duration_8bit.count() << "x" << std::endl;
    std::cout << "  Memory: " << static_cast<float>(int8_memory) / float_memory << "x smaller" << std::endl;
    std::cout << "  RMS error: " << rms_error_8bit << std::endl;
    
    std::cout << "16-bit fixed-point:" << std::endl;
    std::cout << "  Speedup: " << static_cast<float>(duration_float.count()) / duration_16bit.count() << "x" << std::endl;
    std::cout << "  Memory: " << static_cast<float>(int16_memory) / float_memory << "x smaller" << std::endl;
    std::cout << "  RMS error: " << rms_error_16bit << std::endl;
}

int main() {
    test_fixed_point_performance();
    return 0;
} 
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

// Simple test to verify fixed-point CCM implementation
void test_fixed_point_ccm() {
    std::cout << "=== Fixed-Point CCM Test ===" << std::endl;
    
    // Test CCM matrix values
    std::vector<float> corrected_red = {1.660f, -0.527f, -0.133f};
    std::vector<float> corrected_green = {-0.408f, 1.563f, -0.082f};
    std::vector<float> corrected_blue = {-0.055f, -1.641f, 2.695f};
    
    // Test input RGB values
    float test_r = 100.0f;
    float test_g = 150.0f;
    float test_b = 200.0f;
    
    std::cout << "Test input RGB: [" << test_r << ", " << test_g << ", " << test_b << "]" << std::endl;
    
    // Floating-point calculation
    Eigen::Matrix3f ccm_float;
    ccm_float.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data());
    ccm_float.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data());
    ccm_float.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data());
    
    Eigen::Vector3f input_float(test_r, test_g, test_b);
    Eigen::Vector3f result_float = ccm_float * input_float;
    
    std::cout << "Floating-point result: [" << result_float(0) << ", " << result_float(1) << ", " << result_float(2) << "]" << std::endl;
    
    // Fixed-point calculation (16 bits)
    int fixed_point_bits = 16;
    int scale_factor = 1 << fixed_point_bits;
    
    Eigen::Matrix3i ccm_fixed;
    ccm_fixed.row(0) = Eigen::Map<Eigen::Vector3f>(corrected_red.data()).cast<int>() * scale_factor;
    ccm_fixed.row(1) = Eigen::Map<Eigen::Vector3f>(corrected_green.data()).cast<int>() * scale_factor;
    ccm_fixed.row(2) = Eigen::Map<Eigen::Vector3f>(corrected_blue.data()).cast<int>() * scale_factor;
    
    std::cout << "Fixed-point CCM matrix (scaled by 2^" << fixed_point_bits << "):" << std::endl;
    std::cout << ccm_fixed << std::endl;
    
    // Convert input to fixed-point
    int64_t r_fixed = static_cast<int64_t>(test_r * scale_factor);
    int64_t g_fixed = static_cast<int64_t>(test_g * scale_factor);
    int64_t b_fixed = static_cast<int64_t>(test_b * scale_factor);
    
    // Fixed-point matrix multiplication
    int64_t new_r_fixed = static_cast<int64_t>(ccm_fixed(0, 0)) * r_fixed + 
                          static_cast<int64_t>(ccm_fixed(0, 1)) * g_fixed + 
                          static_cast<int64_t>(ccm_fixed(0, 2)) * b_fixed;
    int64_t new_g_fixed = static_cast<int64_t>(ccm_fixed(1, 0)) * r_fixed + 
                          static_cast<int64_t>(ccm_fixed(1, 1)) * g_fixed + 
                          static_cast<int64_t>(ccm_fixed(1, 2)) * b_fixed;
    int64_t new_b_fixed = static_cast<int64_t>(ccm_fixed(2, 0)) * r_fixed + 
                          static_cast<int64_t>(ccm_fixed(2, 1)) * g_fixed + 
                          static_cast<int64_t>(ccm_fixed(2, 2)) * b_fixed;
    
    // Convert back to floating-point with proper rounding
    int64_t half_scale = static_cast<int64_t>(scale_factor) / 2;
    float result_r_fixed = static_cast<float>(new_r_fixed + half_scale) / (scale_factor * scale_factor);
    float result_g_fixed = static_cast<float>(new_g_fixed + half_scale) / (scale_factor * scale_factor);
    float result_b_fixed = static_cast<float>(new_b_fixed + half_scale) / (scale_factor * scale_factor);
    
    std::cout << "Fixed-point result: [" << result_r_fixed << ", " << result_g_fixed << ", " << result_b_fixed << "]" << std::endl;
    
    // Calculate differences
    float diff_r = std::abs(result_float(0) - result_r_fixed);
    float diff_g = std::abs(result_float(1) - result_g_fixed);
    float diff_b = std::abs(result_float(2) - result_b_fixed);
    
    std::cout << "Differences: [" << diff_r << ", " << diff_g << ", " << diff_b << "]" << std::endl;
    
    // Check if differences are acceptable (should be very small)
    float max_diff = std::max({diff_r, diff_g, diff_b});
    float expected_precision = 1.0f / scale_factor; // Theoretical precision
    
    std::cout << "Maximum difference: " << max_diff << std::endl;
    std::cout << "Expected precision: " << expected_precision << std::endl;
    
    if (max_diff < expected_precision * 10) {
        std::cout << "✓ Fixed-point implementation is working correctly!" << std::endl;
    } else {
        std::cout << "✗ Fixed-point implementation has significant errors!" << std::endl;
    }
    
    std::cout << "==========================" << std::endl;
}

int main() {
    test_fixed_point_ccm();
    return 0;
} 
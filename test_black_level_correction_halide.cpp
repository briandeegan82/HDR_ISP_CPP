#include <iostream>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include "src/modules/black_level_correction/black_level_correction.hpp"
#include "src/modules/black_level_correction/black_level_correction_halide.hpp"
#include "src/common/halide_utils.hpp"
#include "include/module_ab_test.hpp"

void test_black_level_correction_performance() {
    std::cout << "=== Black Level Correction Performance Test ===" << std::endl;
    
    // Test parameters
    const int width = 2592;
    const int height = 1536;
    const int iterations = 100;
    
    // Create test configuration
    YAML::Node sensor_info;
    sensor_info["bit_depth"] = 12;
    sensor_info["bayer_pattern"] = "rggb";
    sensor_info["width"] = width;
    sensor_info["height"] = height;
    
    YAML::Node params;
    params["is_enable"] = true;
    params["is_save"] = false;
    params["r_offset"] = 200;
    params["gr_offset"] = 200;
    params["gb_offset"] = 200;
    params["b_offset"] = 200;
    params["r_sat"] = 4095;
    params["gr_sat"] = 4095;
    params["gb_sat"] = 4095;
    params["b_sat"] = 4095;
    
    // Create test image
    std::cout << "Creating test image..." << std::endl;
    hdr_isp::EigenImageU32 test_image = hdr_isp::EigenImageU32::Constant(height, width, 1000);
    
    // Add some variation to the test image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            test_image.data()(y, x) = static_cast<uint32_t>((x + y) % 4096); // 12-bit range
        }
    }
    
    std::cout << "Test image created: " << width << "x" << height << std::endl;
    std::cout << "Input image stats - Min: " << test_image.min() 
              << ", Max: " << test_image.max() 
              << ", Mean: " << test_image.mean() << std::endl;
    
    // Test Eigen implementation
    std::cout << "\n--- Testing Eigen Implementation ---" << std::endl;
    auto eigen_start = std::chrono::high_resolution_clock::now();
    
    hdr_isp::EigenImageU32 eigen_result;
    for (int i = 0; i < iterations; ++i) {
        BlackLevelCorrection blc_eigen(test_image, sensor_info, params);
        eigen_result = blc_eigen.execute();
    }
    
    auto eigen_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> eigen_duration = eigen_end - eigen_start;
    double eigen_time_ms = eigen_duration.count();
    
    std::cout << "Eigen implementation:" << std::endl;
    std::cout << "  Total time: " << eigen_time_ms << " ms" << std::endl;
    std::cout << "  Average time: " << (eigen_time_ms / iterations) << " ms per iteration" << std::endl;
    std::cout << "  Output stats - Min: " << eigen_result.min() 
              << ", Max: " << eigen_result.max() 
              << ", Mean: " << eigen_result.mean() << std::endl;
    
    // Test Halide implementation
    std::cout << "\n--- Testing Halide Implementation ---" << std::endl;
    
    // Convert Eigen image to Halide buffer
    Halide::Buffer<uint32_t> halide_input = hdr_isp::eigenToHalide(test_image);
    
    auto halide_start = std::chrono::high_resolution_clock::now();
    
    hdr_isp::BlackLevelCorrectionHalide blc_halide(halide_input, sensor_info, params);
    Halide::Buffer<uint32_t> halide_result = blc_halide.execute();
    
    auto halide_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> halide_duration = halide_end - halide_start;
    double halide_time_ms = halide_duration.count();
    
    std::cout << "Halide implementation:" << std::endl;
    std::cout << "  Total time: " << halide_time_ms << " ms" << std::endl;
    std::cout << "  Average time: " << (halide_time_ms / 1) << " ms per iteration" << std::endl;
    std::cout << "  Performance stats: " << blc_halide.getPerformanceStats() << std::endl;
    
    // Convert Halide result back to Eigen for comparison
    hdr_isp::EigenImageU32 halide_eigen_result = hdr_isp::halideToEigen(halide_result);
    
    std::cout << "  Output stats - Min: " << halide_eigen_result.min() 
              << ", Max: " << halide_eigen_result.max() 
              << ", Mean: " << halide_eigen_result.mean() << std::endl;
    
    // Performance comparison
    std::cout << "\n--- Performance Comparison ---" << std::endl;
    double speedup = eigen_time_ms / halide_time_ms;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    std::cout << "Performance improvement: " << ((speedup - 1.0) * 100.0) << "%" << std::endl;
    
    // Quality comparison
    std::cout << "\n--- Quality Comparison ---" << std::endl;
    bool outputs_match = hdr_isp::ModuleABTest::compareOutputs(
        eigen_result.toOpenCV(CV_32S), 
        halide_eigen_result.toOpenCV(CV_32S), 
        0.0  // Exact match for integer values
    );
    
    std::cout << "Outputs match: " << (outputs_match ? "YES" : "NO") << std::endl;
    
    if (!outputs_match) {
        std::cout << "Warning: Outputs do not match exactly!" << std::endl;
        
        // Print some sample values for debugging
        std::cout << "Sample values comparison (top-left 3x3):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << "  [" << i << "," << j << "] Eigen: " << eigen_result.data()(i, j)
                         << ", Halide: " << halide_eigen_result.data()(i, j) << std::endl;
            }
        }
    }
    
    // Memory usage comparison
    std::cout << "\n--- Memory Usage Comparison ---" << std::endl;
    size_t image_size_bytes = width * height * sizeof(uint32_t);
    std::cout << "Image size: " << (image_size_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Memory bandwidth (Eigen): " << (image_size_bytes * iterations / (eigen_time_ms / 1000.0) / (1024 * 1024)) << " MB/s" << std::endl;
    std::cout << "Memory bandwidth (Halide): " << (image_size_bytes / (halide_time_ms / 1000.0) / (1024 * 1024)) << " MB/s" << std::endl;
    
    std::cout << "\n=== Test Complete ===" << std::endl;
}

void test_black_level_correction_quality() {
    std::cout << "=== Black Level Correction Quality Test ===" << std::endl;
    
    // Test with different image sizes
    std::vector<std::pair<int, int>> test_sizes = {
        {512, 512},
        {1024, 1024},
        {2048, 1536},
        {2592, 1536}
    };
    
    for (const auto& size : test_sizes) {
        int width = size.first;
        int height = size.second;
        
        std::cout << "\n--- Testing size " << width << "x" << height << " ---" << std::endl;
        
        // Create test configuration
        YAML::Node sensor_info;
        sensor_info["bit_depth"] = 12;
        sensor_info["bayer_pattern"] = "rggb";
        sensor_info["width"] = width;
        sensor_info["height"] = height;
        
        YAML::Node params;
        params["is_enable"] = true;
        params["is_save"] = false;
        params["r_offset"] = 200;
        params["gr_offset"] = 200;
        params["gb_offset"] = 200;
        params["b_offset"] = 200;
        params["r_sat"] = 4095;
        params["gr_sat"] = 4095;
        params["gb_sat"] = 4095;
        params["b_sat"] = 4095;
        
        // Create test image
        hdr_isp::EigenImageU32 test_image = hdr_isp::EigenImageU32::Constant(height, width, 1000);
        
        // Add some variation to the test image
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                test_image.data()(y, x) = static_cast<uint32_t>((x + y) % 4096);
            }
        }
        
        // Test Eigen implementation
        BlackLevelCorrection blc_eigen(test_image, sensor_info, params);
        hdr_isp::EigenImageU32 eigen_result = blc_eigen.execute();
        
        // Test Halide implementation
        Halide::Buffer<uint32_t> halide_input = hdr_isp::eigenToHalide(test_image);
        hdr_isp::BlackLevelCorrectionHalide blc_halide(halide_input, sensor_info, params);
        Halide::Buffer<uint32_t> halide_result = blc_halide.execute();
        hdr_isp::EigenImageU32 halide_eigen_result = hdr_isp::halideToEigen(halide_result);
        
        // Compare outputs
        bool outputs_match = hdr_isp::ModuleABTest::compareOutputs(
            eigen_result.toOpenCV(CV_32S), 
            halide_eigen_result.toOpenCV(CV_32S), 
            0.0
        );
        
        std::cout << "Outputs match: " << (outputs_match ? "YES" : "NO") << std::endl;
        std::cout << "Eigen result - Min: " << eigen_result.min() 
                  << ", Max: " << eigen_result.max() 
                  << ", Mean: " << eigen_result.mean() << std::endl;
        std::cout << "Halide result - Min: " << halide_eigen_result.min() 
                  << ", Max: " << halide_eigen_result.max() 
                  << ", Mean: " << halide_eigen_result.mean() << std::endl;
    }
    
    std::cout << "\n=== Quality Test Complete ===" << std::endl;
}

int main() {
    try {
        std::cout << "Starting Black Level Correction Halide Test..." << std::endl;
        
        // Test performance
        test_black_level_correction_performance();
        
        // Test quality
        test_black_level_correction_quality();
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 
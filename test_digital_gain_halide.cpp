#include <iostream>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include "src/common/eigen_utils.hpp"
#include "src/modules/digital_gain/digital_gain.hpp"
#include "src/modules/digital_gain/digital_gain_halide.hpp"
#include "src/common/halide_utils.hpp"

int main() {
    std::cout << "=== Digital Gain Halide Test ===" << std::endl;
    
    // Create test configuration
    YAML::Node platform;
    platform["name"] = "test_platform";
    
    YAML::Node sensor_info;
    sensor_info["output_bit_depth"] = 12;
    sensor_info["width"] = 2592;
    sensor_info["height"] = 1536;
    
    YAML::Node parm_dga;
    parm_dga["is_save"] = false;
    parm_dga["is_debug"] = true;
    parm_dga["is_auto"] = false;
    parm_dga["current_gain"] = 1;
    parm_dga["ae_feedback"] = 0.0f;
    
    // Create gain array
    YAML::Node gain_array;
    gain_array.push_back(1.0f);
    gain_array.push_back(1.5f);
    gain_array.push_back(2.0f);
    gain_array.push_back(2.5f);
    gain_array.push_back(3.0f);
    parm_dga["gain_array"] = gain_array;
    
    // Create test image (smaller for testing)
    int width = 512;
    int height = 384;
    
    std::cout << "Creating test image: " << width << "x" << height << std::endl;
    
    // Create test image with gradient pattern
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> test_matrix(height, width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            test_matrix(y, x) = static_cast<uint32_t>((x + y) % 4096); // 12-bit range
        }
    }
    
    hdr_isp::EigenImageU32 test_image(test_matrix);
    
    std::cout << "Test image statistics:" << std::endl;
    std::cout << "  Min: " << test_image.min() << std::endl;
    std::cout << "  Max: " << test_image.max() << std::endl;
    std::cout << "  Mean: " << test_image.mean() << std::endl;
    
    // Test original Digital Gain
    std::cout << "\n--- Testing Original Digital Gain ---" << std::endl;
    DigitalGain original_dg(test_image, platform, sensor_info, parm_dga);
    
    auto start_original = std::chrono::high_resolution_clock::now();
    auto result_original = original_dg.execute();
    auto end_original = std::chrono::high_resolution_clock::now();
    
    auto duration_original = std::chrono::duration_cast<std::chrono::microseconds>(end_original - start_original);
    
    std::cout << "Original Digital Gain:" << std::endl;
    std::cout << "  Applied gain: " << result_original.second << std::endl;
    std::cout << "  Execution time: " << duration_original.count() << " microseconds" << std::endl;
    std::cout << "  Output min: " << result_original.first.min() << std::endl;
    std::cout << "  Output max: " << result_original.first.max() << std::endl;
    std::cout << "  Output mean: " << result_original.first.mean() << std::endl;
    
    // Test Halide Digital Gain
    std::cout << "\n--- Testing Halide Digital Gain ---" << std::endl;
    DigitalGainHalide halide_dg(test_image, platform, sensor_info, parm_dga);
    
    auto start_halide = std::chrono::high_resolution_clock::now();
    auto result_halide = halide_dg.execute();
    auto end_halide = std::chrono::high_resolution_clock::now();
    
    auto duration_halide = std::chrono::duration_cast<std::chrono::microseconds>(end_halide - start_halide);
    
    std::cout << "Halide Digital Gain:" << std::endl;
    std::cout << "  Applied gain: " << result_halide.second << std::endl;
    std::cout << "  Execution time: " << duration_halide.count() << " microseconds" << std::endl;
    std::cout << "  Output min: " << result_halide.first.min() << std::endl;
    std::cout << "  Output max: " << result_halide.first.max() << std::endl;
    std::cout << "  Output mean: " << result_halide.first.mean() << std::endl;
    
    // Compare results
    std::cout << "\n--- Performance Comparison ---" << std::endl;
    double speedup = static_cast<double>(duration_original.count()) / duration_halide.count();
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    // Verify output quality
    std::cout << "\n--- Output Quality Verification ---" << std::endl;
    
    // Convert both results to Halide buffers for comparison
    Halide::Buffer<uint32_t> original_buffer = hdr_isp::eigenToHalide(result_original.first);
    Halide::Buffer<uint32_t> halide_buffer = hdr_isp::eigenToHalide(result_halide.first);
    
    bool outputs_match = hdr_isp::compareHalideBuffers(original_buffer, halide_buffer, 1);
    
    if (outputs_match) {
        std::cout << "✅ Outputs match within tolerance!" << std::endl;
    } else {
        std::cout << "❌ Outputs do not match!" << std::endl;
        
        // Print detailed comparison
        std::cout << "Detailed comparison:" << std::endl;
        std::cout << "  Original min/max: " << original_buffer.min() << "/" << original_buffer.max() << std::endl;
        std::cout << "  Halide min/max: " << halide_buffer.min() << "/" << halide_buffer.max() << std::endl;
        
        // Check first few pixels
        for (int y = 0; y < std::min(5, height); ++y) {
            for (int x = 0; x < std::min(5, width); ++x) {
                if (original_buffer(x, y) != halide_buffer(x, y)) {
                    std::cout << "  Mismatch at (" << x << "," << y << "): " 
                              << original_buffer(x, y) << " vs " << halide_buffer(x, y) << std::endl;
                }
            }
        }
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    
    return 0;
} 
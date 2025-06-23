#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "include/module_ab_test.hpp"
#include "src/modules/lens_shading_correction/lens_shading_correction.hpp"
#include "src/modules/lens_shading_correction/lens_shading_correction_halide.hpp"
#include "src/common/eigen_utils.hpp"

int main() {
    std::cout << "=== Lens Shading Correction Halide Test ===" << std::endl;
    
    // Create test configuration
    YAML::Node platform;
    platform["name"] = "test_platform";
    
    YAML::Node sensor_info;
    sensor_info["width"] = 2592;
    sensor_info["height"] = 1536;
    sensor_info["bayer_pattern"] = "RGGB";
    
    YAML::Node parm_lsc;
    parm_lsc["is_enable"] = true;
    parm_lsc["is_save"] = false;
    parm_lsc["is_debug"] = true;
    
    // Create test image (simulate raw sensor data)
    int width = 2592;
    int height = 1536;
    
    // Create a test image with some radial shading pattern
    hdr_isp::EigenImageU32 test_image = hdr_isp::EigenImageU32::Random(height, width);
    
    // Apply some radial shading to make the test more realistic
    float center_x = width / 2.0f;
    float center_y = height / 2.0f;
    float max_distance = std::sqrt(center_x * center_x + center_y * center_y);
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float dx = j - center_x;
            float dy = i - center_y;
            float distance = std::sqrt(dx * dx + dy * dy);
            float shading_factor = 1.0f - 0.2f * (distance / max_distance);
            
            // Apply shading to create realistic test data
            test_image.data()(i, j) = static_cast<uint32_t>(
                std::max(0.0f, std::min(4294967295.0f, 
                    test_image.data()(i, j) * shading_factor))
            );
        }
    }
    
    std::cout << "Test image created: " << width << "x" << height << std::endl;
    std::cout << "Image statistics - Min: " << test_image.min() 
              << ", Max: " << test_image.max() 
              << ", Mean: " << test_image.mean() << std::endl;
    
    // Test original implementation
    std::cout << "\n--- Testing Original Implementation ---" << std::endl;
    auto start_original = std::chrono::high_resolution_clock::now();
    
    LensShadingCorrection lsc_original(test_image, platform, sensor_info, parm_lsc);
    hdr_isp::EigenImageU32 result_original = lsc_original.execute();
    
    auto end_original = std::chrono::high_resolution_clock::now();
    auto duration_original = std::chrono::duration_cast<std::chrono::milliseconds>(end_original - start_original);
    
    std::cout << "Original implementation time: " << duration_original.count() << "ms" << std::endl;
    std::cout << "Original result - Min: " << result_original.min() 
              << ", Max: " << result_original.max() 
              << ", Mean: " << result_original.mean() << std::endl;
    
    // Test Halide implementation
    std::cout << "\n--- Testing Halide Implementation ---" << std::endl;
    auto start_halide = std::chrono::high_resolution_clock::now();
    
    LensShadingCorrectionHalide lsc_halide(test_image, platform, sensor_info, parm_lsc);
    hdr_isp::EigenImageU32 result_halide = lsc_halide.execute();
    
    auto end_halide = std::chrono::high_resolution_clock::now();
    auto duration_halide = std::chrono::duration_cast<std::chrono::milliseconds>(end_halide - start_halide);
    
    std::cout << "Halide implementation time: " << duration_halide.count() << "ms" << std::endl;
    std::cout << "Halide result - Min: " << result_halide.min() 
              << ", Max: " << result_halide.max() 
              << ", Mean: " << result_halide.mean() << std::endl;
    
    // Compare results
    std::cout << "\n--- Performance Comparison ---" << std::endl;
    double speedup = static_cast<double>(duration_original.count()) / duration_halide.count();
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    // Convert to OpenCV for comparison
    cv::Mat cv_original = result_original.toOpenCV(CV_32S);
    cv::Mat cv_halide = result_halide.toOpenCV(CV_32S);
    
    // Use A/B testing framework to compare outputs
    bool outputs_match = hdr_isp::ModuleABTest::compareOutputs(cv_original, cv_halide, 1e-3);
    
    std::cout << "\n--- Output Quality Check ---" << std::endl;
    if (outputs_match) {
        std::cout << "✅ Outputs match within tolerance!" << std::endl;
    } else {
        std::cout << "❌ Outputs differ significantly!" << std::endl;
        
        // Calculate difference statistics
        cv::Mat diff;
        cv::absdiff(cv_original, cv_halide, diff);
        double max_diff, mean_diff;
        cv::minMaxLoc(diff, nullptr, &max_diff);
        mean_diff = cv::mean(diff)[0];
        
        std::cout << "Max difference: " << max_diff << std::endl;
        std::cout << "Mean difference: " << mean_diff << std::endl;
    }
    
    // Benchmark with multiple iterations
    std::cout << "\n--- Benchmarking (100 iterations) ---" << std::endl;
    hdr_isp::ModuleABTest::benchmarkModule("Lens Shading Correction", cv_original, 100);
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
} 
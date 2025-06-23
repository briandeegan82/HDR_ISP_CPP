#include <iostream>
#include <chrono>
#include <random>
#include <yaml-cpp/yaml.h>
#include "src/modules/bayer_noise_reduction/bayer_noise_reduction_halide.hpp"
#include "src/common/eigen_utils.hpp"

using namespace hdr_isp;

// Helper function to create test image with Bayer pattern
Halide::Buffer<uint32_t> createTestImage(int width, int height, const std::string& bayer_pattern) {
    Halide::Buffer<uint32_t> image(width, height);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(1000, 4000); // Realistic sensor values
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint32_t value = dis(gen);
            
            // Apply Bayer pattern
            if (bayer_pattern == "rggb") {
                if ((y % 2 == 0 && x % 2 == 0)) {
                    // Red pixel
                    value = value + 500; // Red channel typically brighter
                } else if ((y % 2 == 1 && x % 2 == 1)) {
                    // Blue pixel
                    value = value + 300; // Blue channel
                } else {
                    // Green pixel
                    value = value + 400; // Green channel
                }
            } else if (bayer_pattern == "bggr") {
                if ((y % 2 == 0 && x % 2 == 0)) {
                    // Blue pixel
                    value = value + 300;
                } else if ((y % 2 == 1 && x % 2 == 1)) {
                    // Red pixel
                    value = value + 500;
                } else {
                    // Green pixel
                    value = value + 400;
                }
            } else if (bayer_pattern == "grbg") {
                if ((y % 2 == 0 && x % 2 == 1)) {
                    // Red pixel
                    value = value + 500;
                } else if ((y % 2 == 1 && x % 2 == 0)) {
                    // Blue pixel
                    value = value + 300;
                } else {
                    // Green pixel
                    value = value + 400;
                }
            } else if (bayer_pattern == "gbrg") {
                if ((y % 2 == 0 && x % 2 == 1)) {
                    // Blue pixel
                    value = value + 300;
                } else if ((y % 2 == 1 && x % 2 == 0)) {
                    // Red pixel
                    value = value + 500;
                } else {
                    // Green pixel
                    value = value + 400;
                }
            }
            
            image(x, y) = value;
        }
    }
    
    return image;
}

// Helper function to validate output
bool validateOutput(const Halide::Buffer<uint32_t>& output, int width, int height) {
    // Check that output has correct dimensions
    if (output.width() != width || output.height() != height) {
        std::cerr << "Output dimensions mismatch: expected " << width << "x" << height 
                  << ", got " << output.width() << "x" << output.height() << std::endl;
        return false;
    }
    
    // Check that output contains valid values (not all zeros)
    uint32_t min_val = std::numeric_limits<uint32_t>::max();
    uint32_t max_val = 0;
    uint64_t sum = 0;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint32_t val = output(x, y);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
        }
    }
    
    double mean_val = static_cast<double>(sum) / (width * height);
    
    std::cout << "Output validation:" << std::endl;
    std::cout << "  Min value: " << min_val << std::endl;
    std::cout << "  Max value: " << max_val << std::endl;
    std::cout << "  Mean value: " << mean_val << std::endl;
    
    // Basic sanity checks
    if (min_val == 0 && max_val == 0) {
        std::cerr << "Output is all zeros!" << std::endl;
        return false;
    }
    
    if (max_val > 65535) {
        std::cout << "Warning: Output contains values > 65535 (16-bit range)" << std::endl;
    }
    
    return true;
}

// Test function for a specific Bayer pattern
void testBayerPattern(const std::string& bayer_pattern, int width, int height) {
    std::cout << "\n=== Testing Bayer Pattern: " << bayer_pattern << " ===" << std::endl;
    std::cout << "Image size: " << width << "x" << height << std::endl;
    
    // Create test image
    Halide::Buffer<uint32_t> test_image = createTestImage(width, height, bayer_pattern);
    
    // Create sensor info
    YAML::Node sensor_info;
    sensor_info["bit_depth"] = 16;
    sensor_info["bayer_pattern"] = bayer_pattern;
    
    // Create parameters
    YAML::Node params;
    params["is_enable"] = true;
    params["is_debug"] = true;
    params["is_save"] = false;
    
    try {
        // Create and execute Bayer Noise Reduction Halide
        BayerNoiseReductionHalide bnr_halide(test_image, sensor_info, params);
        
        auto start = std::chrono::high_resolution_clock::now();
        Halide::Buffer<uint32_t> result = bnr_halide.execute();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        // Print performance stats
        std::cout << bnr_halide.getPerformanceStats();
        std::cout << "  Wall clock time: " << elapsed.count() << "ms" << std::endl;
        
        // Validate output
        if (validateOutput(result, width, height)) {
            std::cout << "✓ Test PASSED for " << bayer_pattern << std::endl;
        } else {
            std::cout << "✗ Test FAILED for " << bayer_pattern << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Test FAILED for " << bayer_pattern << ": " << e.what() << std::endl;
    }
}

// Performance benchmark function
void benchmarkPerformance(const std::string& bayer_pattern, int width, int height, int iterations = 100) {
    std::cout << "\n=== Performance Benchmark: " << bayer_pattern << " ===" << std::endl;
    std::cout << "Image size: " << width << "x" << height << ", Iterations: " << iterations << std::endl;
    
    // Create test image
    Halide::Buffer<uint32_t> test_image = createTestImage(width, height, bayer_pattern);
    
    // Create sensor info and parameters
    YAML::Node sensor_info;
    sensor_info["bit_depth"] = 16;
    sensor_info["bayer_pattern"] = bayer_pattern;
    
    YAML::Node params;
    params["is_enable"] = true;
    params["is_debug"] = false;
    params["is_save"] = false;
    
    try {
        BayerNoiseReductionHalide bnr_halide(test_image, sensor_info, params);
        
        // Warm-up run
        bnr_halide.execute();
        
        // Benchmark runs
        std::vector<double> times;
        times.reserve(iterations);
        
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            bnr_halide.execute();
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> elapsed = end - start;
            times.push_back(elapsed.count());
        }
        
        // Calculate statistics
        double total_time = 0.0;
        double min_time = std::numeric_limits<double>::max();
        double max_time = 0.0;
        
        for (double time : times) {
            total_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        
        double avg_time = total_time / iterations;
        
        std::cout << "Performance Results:" << std::endl;
        std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time << "ms" << std::endl;
        std::cout << "  Min time: " << std::fixed << std::setprecision(3) << min_time << "ms" << std::endl;
        std::cout << "  Max time: " << std::fixed << std::setprecision(3) << max_time << "ms" << std::endl;
        std::cout << "  Total time: " << std::fixed << std::setprecision(3) << total_time << "ms" << std::endl;
        
        // Calculate throughput
        double pixels_per_second = (width * height * iterations) / (total_time / 1000.0);
        std::cout << "  Throughput: " << std::fixed << std::setprecision(0) 
                  << pixels_per_second / 1000000.0 << " MPixels/sec" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark FAILED: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== Bayer Noise Reduction Halide Test ===" << std::endl;
    
    // Test different Bayer patterns
    std::vector<std::string> bayer_patterns = {"rggb", "bggr", "grbg", "gbrg"};
    std::vector<std::pair<int, int>> image_sizes = {
        {64, 64},      // Small test
        {256, 256},    // Medium test
        {512, 512}     // Large test
    };
    
    // Basic functionality tests
    for (const auto& pattern : bayer_patterns) {
        for (const auto& size : image_sizes) {
            testBayerPattern(pattern, size.first, size.second);
        }
    }
    
    // Performance benchmarks
    for (const auto& pattern : bayer_patterns) {
        benchmarkPerformance(pattern, 512, 512, 50); // 50 iterations for benchmark
    }
    
    std::cout << "\n=== All Tests Completed ===" << std::endl;
    return 0;
} 
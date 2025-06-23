#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace hdr_isp {

struct ABTestResult {
    std::string module_name;
    std::string test_name;
    double original_time_ms;
    double hybrid_time_ms;
    double speedup_factor;
    double output_difference;
    bool output_match;
    int image_width;
    int image_height;
    int iterations;
};

class ModuleABTest {
public:
    ModuleABTest();
    ~ModuleABTest() = default;

    // Compare outputs of original vs hybrid implementations
    static bool compareOutputs(const cv::Mat& original, const cv::Mat& hybrid, 
                              double tolerance = 1e-6);
    
    // Benchmark a module with both original and hybrid implementations
    static ABTestResult benchmarkModule(const std::string& module_name, 
                                       const cv::Mat& test_input, 
                                       int iterations = 100);
    
    // Run comprehensive A/B test suite
    static std::vector<ABTestResult> runFullABTest(const cv::Mat& test_input, 
                                                   int iterations = 100);
    
    // Generate A/B test report
    static void generateReport(const std::vector<ABTestResult>& results, 
                              const std::string& output_file = "");
    
    // Get performance summary
    static void printPerformanceSummary(const std::vector<ABTestResult>& results);

private:
    // Helper functions
    static double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2);
    static double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2);
    static double calculateMeanAbsoluteError(const cv::Mat& img1, const cv::Mat& img2);
    static std::string formatTime(double time_ms);
    static std::string formatSpeedup(double speedup);
};

// Template for easy A/B testing of any module
template<typename OriginalModule, typename HybridModule>
class ModuleABTestTemplate {
public:
    static ABTestResult testModule(const std::string& module_name,
                                  const cv::Mat& test_input,
                                  const typename OriginalModule::Params& params,
                                  int iterations = 100) {
        ABTestResult result;
        result.module_name = module_name;
        result.test_name = "A/B Test";
        result.image_width = test_input.cols;
        result.image_height = test_input.rows;
        result.iterations = iterations;
        
        // Test original implementation
        auto start_original = std::chrono::high_resolution_clock::now();
        cv::Mat original_output;
        for (int i = 0; i < iterations; ++i) {
            OriginalModule original_module(test_input, params);
            original_output = original_module.execute();
        }
        auto end_original = std::chrono::high_resolution_clock::now();
        result.original_time_ms = std::chrono::duration<double, std::milli>(end_original - start_original).count();
        
        // Test hybrid implementation
        auto start_hybrid = std::chrono::high_resolution_clock::now();
        cv::Mat hybrid_output;
        for (int i = 0; i < iterations; ++i) {
            HybridModule hybrid_module(test_input, params);
            hybrid_output = hybrid_module.execute();
        }
        auto end_hybrid = std::chrono::high_resolution_clock::now();
        result.hybrid_time_ms = std::chrono::duration<double, std::milli>(end_hybrid - start_hybrid).count();
        
        // Calculate metrics
        result.speedup_factor = result.original_time_ms / result.hybrid_time_ms;
        result.output_difference = ModuleABTest::calculateMeanAbsoluteError(original_output, hybrid_output);
        result.output_match = ModuleABTest::compareOutputs(original_output, hybrid_output);
        
        return result;
    }
};

// Convenience function for quick A/B test
template<typename OriginalModule, typename HybridModule>
ABTestResult quickABTest(const std::string& module_name,
                        const cv::Mat& test_input,
                        const typename OriginalModule::Params& params) {
    return ModuleABTestTemplate<OriginalModule, HybridModule>::testModule(
        module_name, test_input, params, 10);
}

} // namespace hdr_isp 
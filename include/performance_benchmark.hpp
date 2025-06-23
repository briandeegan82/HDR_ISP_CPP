#pragma once

#include "hybrid_backend.hpp"
#include <vector>
#include <string>
#include <map>

namespace hdr_isp {

struct BenchmarkResult {
    std::string operation_name;
    std::string backend_name;
    double execution_time_ms;
    double throughput_mpixels_per_sec;
    int image_width;
    int image_height;
    int iterations;
};

class PerformanceBenchmark {
public:
    PerformanceBenchmark();
    ~PerformanceBenchmark() = default;

    // Run benchmark for a specific operation across all available backends
    std::vector<BenchmarkResult> benchmarkOperation(
        const std::string& operation_name,
        const cv::Mat& test_image,
        int iterations = 100
    );

    // Run comprehensive benchmark suite
    std::vector<BenchmarkResult> runFullBenchmark(
        const cv::Mat& test_image,
        int iterations = 100
    );

    // Generate benchmark report
    void generateReport(const std::vector<BenchmarkResult>& results, 
                       const std::string& output_file = "");

    // Get the fastest backend for a specific operation
    std::string getFastestBackend(const std::string& operation_name,
                                 const std::vector<BenchmarkResult>& results);

private:
    // Test operations
    cv::Mat testGaussianBlur(const cv::Mat& input);
    cv::Mat testColorConversion(const cv::Mat& input);
    cv::Mat testResize(const cv::Mat& input);
    cv::Mat testConvolution(const cv::Mat& input);
    
    // Helper functions
    double calculateThroughput(double time_ms, int width, int height);
    std::string formatTime(double time_ms);
    std::string formatThroughput(double throughput);
};

// Convenience function to run quick benchmark
std::vector<BenchmarkResult> quickBenchmark(const cv::Mat& test_image);

} // namespace hdr_isp 
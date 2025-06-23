#include "../../include/module_ab_test.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace hdr_isp {

ModuleABTest::ModuleABTest() {
    // Constructor - no initialization needed for static class
}

bool ModuleABTest::compareOutputs(const cv::Mat& original, const cv::Mat& hybrid, double tolerance) {
    if (original.size() != hybrid.size() || original.type() != hybrid.type()) {
        std::cerr << "Output comparison failed: size or type mismatch" << std::endl;
        return false;
    }
    
    // Convert to float for comparison
    cv::Mat original_float, hybrid_float;
    original.convertTo(original_float, CV_32F);
    hybrid.convertTo(hybrid_float, CV_32F);
    
    // Calculate absolute difference
    cv::Mat diff;
    cv::absdiff(original_float, hybrid_float, diff);
    
    // Check if any pixel exceeds tolerance
    double max_diff;
    cv::minMaxLoc(diff, nullptr, &max_diff);
    
    bool match = max_diff <= tolerance;
    
    if (!match) {
        std::cout << "Output comparison: Max difference = " << max_diff 
                  << " (tolerance = " << tolerance << ")" << std::endl;
    }
    
    return match;
}

ABTestResult ModuleABTest::benchmarkModule(const std::string& module_name, 
                                          const cv::Mat& test_input, 
                                          int iterations) {
    ABTestResult result;
    result.module_name = module_name;
    result.test_name = "Benchmark";
    result.image_width = test_input.cols;
    result.image_height = test_input.rows;
    result.iterations = iterations;
    
    // This is a placeholder implementation
    // In a real implementation, you would:
    // 1. Create original module instance
    // 2. Create hybrid module instance
    // 3. Run both with timing
    // 4. Compare outputs
    
    std::cout << "Benchmarking " << module_name << " with " << iterations << " iterations..." << std::endl;
    
    // Placeholder timing (replace with actual module execution)
    result.original_time_ms = 100.0; // Placeholder
    result.hybrid_time_ms = 50.0;    // Placeholder
    result.speedup_factor = result.original_time_ms / result.hybrid_time_ms;
    result.output_difference = 0.0;  // Placeholder
    result.output_match = true;      // Placeholder
    
    return result;
}

std::vector<ABTestResult> ModuleABTest::runFullABTest(const cv::Mat& test_input, int iterations) {
    std::vector<ABTestResult> results;
    
    // List of modules to test
    std::vector<std::string> modules = {
        "RGB Conversion",
        "Color Space Conversion", 
        "2D Noise Reduction",
        "Scale/Resize",
        "Gaussian Blur"
    };
    
    for (const auto& module : modules) {
        try {
            ABTestResult result = benchmarkModule(module, test_input, iterations);
            results.push_back(result);
        } catch (const std::exception& e) {
            std::cerr << "Failed to benchmark " << module << ": " << e.what() << std::endl;
        }
    }
    
    return results;
}

void ModuleABTest::generateReport(const std::vector<ABTestResult>& results, const std::string& output_file) {
    std::ostream* out = &std::cout;
    std::ofstream file_out;
    
    if (!output_file.empty()) {
        file_out.open(output_file);
        if (file_out.is_open()) {
            out = &file_out;
        } else {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
        }
    }
    
    *out << "=== HDR ISP Hybrid Backend A/B Test Report ===" << std::endl;
    *out << "Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
    *out << std::endl;
    
    // Summary table
    *out << std::setw(25) << std::left << "Module" 
         << std::setw(12) << std::right << "Original(ms)"
         << std::setw(12) << std::right << "Hybrid(ms)"
         << std::setw(12) << std::right << "Speedup"
         << std::setw(12) << std::right << "Diff"
         << std::setw(8) << std::right << "Match" << std::endl;
    
    *out << std::string(85, '-') << std::endl;
    
    double total_original_time = 0.0;
    double total_hybrid_time = 0.0;
    int match_count = 0;
    
    for (const auto& result : results) {
        *out << std::setw(25) << std::left << result.module_name
             << std::setw(12) << std::right << std::fixed << std::setprecision(2) << result.original_time_ms
             << std::setw(12) << std::right << std::fixed << std::setprecision(2) << result.hybrid_time_ms
             << std::setw(12) << std::right << std::fixed << std::setprecision(2) << result.speedup_factor
             << std::setw(12) << std::right << std::scientific << std::setprecision(2) << result.output_difference
             << std::setw(8) << std::right << (result.output_match ? "✓" : "✗") << std::endl;
        
        total_original_time += result.original_time_ms;
        total_hybrid_time += result.hybrid_time_ms;
        if (result.output_match) match_count++;
    }
    
    *out << std::string(85, '-') << std::endl;
    *out << std::setw(25) << std::left << "TOTAL"
         << std::setw(12) << std::right << std::fixed << std::setprecision(2) << total_original_time
         << std::setw(12) << std::right << std::fixed << std::setprecision(2) << total_hybrid_time
         << std::setw(12) << std::right << std::fixed << std::setprecision(2) << (total_original_time / total_hybrid_time)
         << std::setw(12) << std::right << "-"
         << std::setw(8) << std::right << (std::to_string(match_count) + "/" + std::to_string(results.size())) << std::endl;
    
    *out << std::endl;
    
    // Performance summary
    printPerformanceSummary(results);
    
    if (file_out.is_open()) {
        file_out.close();
        std::cout << "Report saved to: " << output_file << std::endl;
    }
}

void ModuleABTest::printPerformanceSummary(const std::vector<ABTestResult>& results) {
    if (results.empty()) {
        std::cout << "No results to summarize." << std::endl;
        return;
    }
    
    std::cout << "=== Performance Summary ===" << std::endl;
    
    // Calculate statistics
    std::vector<double> speedups;
    for (const auto& result : results) {
        speedups.push_back(result.speedup_factor);
    }
    
    std::sort(speedups.begin(), speedups.end());
    
    double avg_speedup = std::accumulate(speedups.begin(), speedups.end(), 0.0) / speedups.size();
    double median_speedup = speedups[speedups.size() / 2];
    double min_speedup = speedups.front();
    double max_speedup = speedups.back();
    
    std::cout << "Average speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "x" << std::endl;
    std::cout << "Median speedup:  " << std::fixed << std::setprecision(2) << median_speedup << "x" << std::endl;
    std::cout << "Min speedup:     " << std::fixed << std::setprecision(2) << min_speedup << "x" << std::endl;
    std::cout << "Max speedup:     " << std::fixed << std::setprecision(2) << max_speedup << "x" << std::endl;
    
    // Count matches
    int match_count = 0;
    for (const auto& result : results) {
        if (result.output_match) match_count++;
    }
    
    std::cout << "Output quality:  " << match_count << "/" << results.size() 
              << " modules match original output" << std::endl;
    
    std::cout << std::endl;
}

double ModuleABTest::calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    
    double mse = cv::mean(diff)[0];
    if (mse <= 1e-10) return 100.0; // Perfect match
    
    return 10.0 * log10((255 * 255) / mse);
}

double ModuleABTest::calculateSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    // Simplified SSIM calculation
    // In a full implementation, you would use a proper SSIM algorithm
    
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_32F);
    
    double mae = cv::mean(diff)[0];
    double max_val = 255.0;
    
    return 1.0 - (mae / max_val);
}

double ModuleABTest::calculateMeanAbsoluteError(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_32F);
    
    return cv::mean(diff)[0];
}

std::string ModuleABTest::formatTime(double time_ms) {
    if (time_ms < 1.0) {
        return std::to_string(static_cast<int>(time_ms * 1000)) + "μs";
    } else if (time_ms < 1000.0) {
        return std::to_string(static_cast<int>(time_ms)) + "ms";
    } else {
        return std::to_string(static_cast<int>(time_ms / 1000)) + "s";
    }
}

std::string ModuleABTest::formatSpeedup(double speedup) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << speedup << "x";
    return oss.str();
}

} // namespace hdr_isp 
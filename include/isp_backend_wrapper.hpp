#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

// Optional hybrid backend inclusion
#ifdef USE_HYBRID_BACKEND
    #include "hybrid_backend.hpp"
#endif

namespace hdr_isp {

/**
 * @brief Backend-aware wrapper for ISP operations
 * 
 * This class provides optimized versions of common ISP operations
 * while maintaining compatibility with existing code.
 */
class ISPBackendWrapper {
public:
    ISPBackendWrapper();
    ~ISPBackendWrapper() = default;

    // Initialize the backend (optional)
    static bool initializeBackend();

    // Core ISP operations with backend optimization
    static cv::Mat gaussianBlur(const cv::Mat& input, const cv::Size& kernel_size, double sigma = 0);
    static cv::Mat colorSpaceConversion(const cv::Mat& input, int code);
    static cv::Mat resize(const cv::Mat& input, const cv::Size& size, int interpolation = cv::INTER_LINEAR);
    static cv::Mat matrixMultiplication(const cv::Mat& input, const cv::Mat& matrix);
    static cv::Mat convolution(const cv::Mat& input, const cv::Mat& kernel);
    
    // Performance measurement
    static void startTimer();
    static double endTimer();
    
    // Backend information
    static std::string getCurrentBackend();
    static bool isOptimizedBackendAvailable();

private:
    // Internal implementation helpers
    static cv::Mat fallbackGaussianBlur(const cv::Mat& input, const cv::Size& kernel_size, double sigma);
    static cv::Mat fallbackColorSpaceConversion(const cv::Mat& input, int code);
    static cv::Mat fallbackResize(const cv::Mat& input, const cv::Size& size, int interpolation);
    static cv::Mat fallbackMatrixMultiplication(const cv::Mat& input, const cv::Mat& matrix);
    static cv::Mat fallbackConvolution(const cv::Mat& input, const cv::Mat& kernel);
};

// Convenience functions for direct use
cv::Mat optimizedGaussianBlur(const cv::Mat& input, const cv::Size& kernel_size, double sigma = 0);
cv::Mat optimizedColorSpaceConversion(const cv::Mat& input, int code);
cv::Mat optimizedResize(const cv::Mat& input, const cv::Size& size, int interpolation = cv::INTER_LINEAR);
cv::Mat optimizedMatrixMultiplication(const cv::Mat& input, const cv::Mat& matrix);
cv::Mat optimizedConvolution(const cv::Mat& input, const cv::Mat& kernel);

} // namespace hdr_isp 
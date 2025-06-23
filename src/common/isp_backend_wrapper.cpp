#include "../../include/isp_backend_wrapper.hpp"
#include <chrono>
#include <iostream>

namespace hdr_isp {

// Static timer for performance measurement
static std::chrono::high_resolution_clock::time_point timer_start_;

ISPBackendWrapper::ISPBackendWrapper() {
    // Initialize backend if hybrid backend is available
#ifdef USE_HYBRID_BACKEND
    initializeBackend();
#endif
}

bool ISPBackendWrapper::initializeBackend() {
#ifdef USE_HYBRID_BACKEND
    if (g_backend) {
        return g_backend->initialize(BackendType::AUTO);
    }
#endif
    return false;
}

cv::Mat ISPBackendWrapper::gaussianBlur(const cv::Mat& input, const cv::Size& kernel_size, double sigma) {
#ifdef USE_HYBRID_BACKEND
    if (isOptimizedBackendAvailable()) {
        try {
            // Try optimized backend first
            if (g_backend->getCurrentBackend() == BackendType::OPENCV_OPENCL) {
                // Use OpenCV OpenCL for Gaussian blur
                cv::UMat input_umat, output_umat;
                input.copyTo(input_umat);
                cv::GaussianBlur(input_umat, output_umat, kernel_size, sigma);
                cv::Mat result;
                output_umat.copyTo(result);
                return result;
            } else if (g_backend->getCurrentBackend() == BackendType::HALIDE_CPU || 
                       g_backend->getCurrentBackend() == BackendType::HALIDE_OPENCL) {
                // Use Halide for Gaussian blur (simplified implementation)
                // In a full implementation, this would use Halide's Gaussian blur
                return fallbackGaussianBlur(input, kernel_size, sigma);
            }
        } catch (const std::exception& e) {
            std::cerr << "Optimized Gaussian blur failed, falling back to OpenCV: " << e.what() << std::endl;
        }
    }
#endif
    return fallbackGaussianBlur(input, kernel_size, sigma);
}

cv::Mat ISPBackendWrapper::colorSpaceConversion(const cv::Mat& input, int code) {
#ifdef USE_HYBRID_BACKEND
    if (isOptimizedBackendAvailable()) {
        try {
            // Try optimized backend first
            if (g_backend->getCurrentBackend() == BackendType::OPENCV_OPENCL) {
                // Use OpenCV OpenCL for color space conversion
                cv::UMat input_umat, output_umat;
                input.copyTo(input_umat);
                cv::cvtColor(input_umat, output_umat, code);
                cv::Mat result;
                output_umat.copyTo(result);
                return result;
            }
        } catch (const std::exception& e) {
            std::cerr << "Optimized color space conversion failed, falling back to OpenCV: " << e.what() << std::endl;
        }
    }
#endif
    return fallbackColorSpaceConversion(input, code);
}

cv::Mat ISPBackendWrapper::resize(const cv::Mat& input, const cv::Size& size, int interpolation) {
#ifdef USE_HYBRID_BACKEND
    if (isOptimizedBackendAvailable()) {
        try {
            // Try optimized backend first
            if (g_backend->getCurrentBackend() == BackendType::OPENCV_OPENCL) {
                // Use OpenCV OpenCL for resize
                cv::UMat input_umat, output_umat;
                input.copyTo(input_umat);
                cv::resize(input_umat, output_umat, size, 0, 0, interpolation);
                cv::Mat result;
                output_umat.copyTo(result);
                return result;
            }
        } catch (const std::exception& e) {
            std::cerr << "Optimized resize failed, falling back to OpenCV: " << e.what() << std::endl;
        }
    }
#endif
    return fallbackResize(input, size, interpolation);
}

cv::Mat ISPBackendWrapper::matrixMultiplication(const cv::Mat& input, const cv::Mat& matrix) {
#ifdef USE_HYBRID_BACKEND
    if (isOptimizedBackendAvailable()) {
        try {
            // Try optimized backend first
            if (g_backend->getCurrentBackend() == BackendType::OPENCV_OPENCL) {
                // Use OpenCV OpenCL for matrix multiplication
                cv::UMat input_umat, matrix_umat, output_umat;
                input.copyTo(input_umat);
                matrix.copyTo(matrix_umat);
                cv::gemm(input_umat, matrix_umat, 1.0, cv::UMat(), 0.0, output_umat);
                cv::Mat result;
                output_umat.copyTo(result);
                return result;
            }
        } catch (const std::exception& e) {
            std::cerr << "Optimized matrix multiplication failed, falling back to OpenCV: " << e.what() << std::endl;
        }
    }
#endif
    return fallbackMatrixMultiplication(input, matrix);
}

cv::Mat ISPBackendWrapper::convolution(const cv::Mat& input, const cv::Mat& kernel) {
#ifdef USE_HYBRID_BACKEND
    if (isOptimizedBackendAvailable()) {
        try {
            // Try optimized backend first
            if (g_backend->getCurrentBackend() == BackendType::OPENCV_OPENCL) {
                // Use OpenCV OpenCL for convolution
                cv::UMat input_umat, kernel_umat, output_umat;
                input.copyTo(input_umat);
                kernel.copyTo(kernel_umat);
                cv::filter2D(input_umat, output_umat, -1, kernel_umat);
                cv::Mat result;
                output_umat.copyTo(result);
                return result;
            }
        } catch (const std::exception& e) {
            std::cerr << "Optimized convolution failed, falling back to OpenCV: " << e.what() << std::endl;
        }
    }
#endif
    return fallbackConvolution(input, kernel);
}

void ISPBackendWrapper::startTimer() {
    timer_start_ = std::chrono::high_resolution_clock::now();
}

double ISPBackendWrapper::endTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - timer_start_);
    return duration.count() / 1000.0; // Convert to milliseconds
}

std::string ISPBackendWrapper::getCurrentBackend() {
#ifdef USE_HYBRID_BACKEND
    if (g_backend) {
        return g_backend->getBackendName();
    }
#endif
    return "OpenCV CPU (Fallback)";
}

bool ISPBackendWrapper::isOptimizedBackendAvailable() {
#ifdef USE_HYBRID_BACKEND
    return g_backend && g_backend->getCurrentBackend() != BackendType::OPENCV_CPU;
#else
    return false;
#endif
}

// Fallback implementations using standard OpenCV
cv::Mat ISPBackendWrapper::fallbackGaussianBlur(const cv::Mat& input, const cv::Size& kernel_size, double sigma) {
    cv::Mat output;
    cv::GaussianBlur(input, output, kernel_size, sigma);
    return output;
}

cv::Mat ISPBackendWrapper::fallbackColorSpaceConversion(const cv::Mat& input, int code) {
    cv::Mat output;
    cv::cvtColor(input, output, code);
    return output;
}

cv::Mat ISPBackendWrapper::fallbackResize(const cv::Mat& input, const cv::Size& size, int interpolation) {
    cv::Mat output;
    cv::resize(input, output, size, 0, 0, interpolation);
    return output;
}

cv::Mat ISPBackendWrapper::fallbackMatrixMultiplication(const cv::Mat& input, const cv::Mat& matrix) {
    cv::Mat output;
    cv::gemm(input, matrix, 1.0, cv::Mat(), 0.0, output);
    return output;
}

cv::Mat ISPBackendWrapper::fallbackConvolution(const cv::Mat& input, const cv::Mat& kernel) {
    cv::Mat output;
    cv::filter2D(input, output, -1, kernel);
    return output;
}

// Convenience functions for direct use
cv::Mat optimizedGaussianBlur(const cv::Mat& input, const cv::Size& kernel_size, double sigma) {
    return ISPBackendWrapper::gaussianBlur(input, kernel_size, sigma);
}

cv::Mat optimizedColorSpaceConversion(const cv::Mat& input, int code) {
    return ISPBackendWrapper::colorSpaceConversion(input, code);
}

cv::Mat optimizedResize(const cv::Mat& input, const cv::Size& size, int interpolation) {
    return ISPBackendWrapper::resize(input, size, interpolation);
}

cv::Mat optimizedMatrixMultiplication(const cv::Mat& input, const cv::Mat& matrix) {
    return ISPBackendWrapper::matrixMultiplication(input, matrix);
}

cv::Mat optimizedConvolution(const cv::Mat& input, const cv::Mat& kernel) {
    return ISPBackendWrapper::convolution(input, kernel);
}

} // namespace hdr_isp 
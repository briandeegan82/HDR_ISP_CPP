#pragma once

#include <opencv2/opencv.hpp>
#include <Halide.h>
#include <memory>
#include <string>

namespace hdr_isp {

enum class BackendType {
    OPENCV_CPU,
    OPENCV_OPENCL,
    HALIDE_CPU,
    HALIDE_OPENCL,
    AUTO  // Automatically select best available backend
};

class HybridBackend {
public:
    HybridBackend();
    ~HybridBackend() = default;

    // Initialize the backend with preferred type
    bool initialize(BackendType preferred = BackendType::AUTO);
    
    // Get current backend type
    BackendType getCurrentBackend() const { return current_backend_; }
    
    // Check if specific backend is available
    bool isBackendAvailable(BackendType backend) const;
    
    // Switch to a specific backend
    bool switchBackend(BackendType backend);
    
    // Get backend name as string
    std::string getBackendName() const;
    
    // Performance measurement utilities
    void startTimer();
    double endTimer() const; // Returns elapsed time in milliseconds
    
    // OpenCV utilities with backend awareness
    cv::Mat processWithOpenCV(const cv::Mat& input, const std::string& operation);
    
    // Halide utilities
    Halide::Buffer<float> processWithHalide(const Halide::Buffer<float>& input, const std::string& operation);
    
    // Conversion utilities
    Halide::Buffer<float> cvMatToHalide(const cv::Mat& cv_mat);
    cv::Mat halideToCvMat(const Halide::Buffer<float>& halide_buffer);

private:
    BackendType current_backend_;
    bool opencv_opencl_available_;
    bool halide_opencl_available_;
    std::chrono::high_resolution_clock::time_point timer_start_;
    
    // Auto-select best available backend
    BackendType selectBestBackend() const;
    
    // Initialize OpenCV OpenCL
    bool initializeOpenCVOpenCL();
    
    // Initialize Halide OpenCL
    bool initializeHalideOpenCL();
};

// Global backend instance
extern std::unique_ptr<HybridBackend> g_backend;

// Convenience macros for backend switching
#define USE_OPENCV_CPU() g_backend->switchBackend(BackendType::OPENCV_CPU)
#define USE_OPENCV_OPENCL() g_backend->switchBackend(BackendType::OPENCV_OPENCL)
#define USE_HALIDE_CPU() g_backend->switchBackend(BackendType::HALIDE_CPU)
#define USE_HALIDE_OPENCL() g_backend->switchBackend(BackendType::HALIDE_OPENCL)
#define USE_AUTO() g_backend->switchBackend(BackendType::AUTO)

// Performance measurement macros
#define START_TIMER() g_backend->startTimer()
#define END_TIMER() g_backend->endTimer()

} // namespace hdr_isp 
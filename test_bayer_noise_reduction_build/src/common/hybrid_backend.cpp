#include "../../include/hybrid_backend.hpp"
#include <chrono>
#include <iostream>

namespace hdr_isp {

// Global backend instance
std::unique_ptr<HybridBackend> g_backend = std::make_unique<HybridBackend>();

HybridBackend::HybridBackend() 
    : current_backend_(BackendType::OPENCV_CPU)
    , opencv_opencl_available_(false)
    , halide_opencl_available_(false) {
}

bool HybridBackend::initialize(BackendType preferred) {
    std::cout << "Initializing Hybrid Backend..." << std::endl;
    
    // Check OpenCV OpenCL availability
    opencv_opencl_available_ = initializeOpenCVOpenCL();
    if (opencv_opencl_available_) {
        std::cout << "OpenCV OpenCL: Available" << std::endl;
    } else {
        std::cout << "OpenCV OpenCL: Not available" << std::endl;
    }
    
    // Check Halide OpenCL availability
    halide_opencl_available_ = initializeHalideOpenCL();
    if (halide_opencl_available_) {
        std::cout << "Halide OpenCL: Available" << std::endl;
    } else {
        std::cout << "Halide OpenCL: Not available" << std::endl;
    }
    
    // Set initial backend
    if (preferred == BackendType::AUTO) {
        current_backend_ = selectBestBackend();
    } else {
        current_backend_ = preferred;
    }
    
    std::cout << "Selected backend: " << getBackendName() << std::endl;
    return true;
}

bool HybridBackend::isBackendAvailable(BackendType backend) const {
    switch (backend) {
        case BackendType::OPENCV_CPU:
            return true; // OpenCV CPU is always available
        case BackendType::OPENCV_OPENCL:
            return opencv_opencl_available_;
        case BackendType::HALIDE_CPU:
            return true; // Halide CPU is always available
        case BackendType::HALIDE_OPENCL:
            return halide_opencl_available_;
        case BackendType::AUTO:
            return true;
        default:
            return false;
    }
}

bool HybridBackend::switchBackend(BackendType backend) {
    if (backend == BackendType::AUTO) {
        backend = selectBestBackend();
    }
    
    if (!isBackendAvailable(backend)) {
        std::cerr << "Backend not available: " << static_cast<int>(backend) << std::endl;
        return false;
    }
    
    current_backend_ = backend;
    std::cout << "Switched to backend: " << getBackendName() << std::endl;
    return true;
}

std::string HybridBackend::getBackendName() const {
    switch (current_backend_) {
        case BackendType::OPENCV_CPU: return "OpenCV CPU";
        case BackendType::OPENCV_OPENCL: return "OpenCV OpenCL";
        case BackendType::HALIDE_CPU: return "Halide CPU";
        case BackendType::HALIDE_OPENCL: return "Halide OpenCL";
        case BackendType::AUTO: return "Auto";
        default: return "Unknown";
    }
}

void HybridBackend::startTimer() {
    timer_start_ = std::chrono::high_resolution_clock::now();
}

double HybridBackend::endTimer() const {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - timer_start_);
    return duration.count() / 1000.0; // Convert to milliseconds
}

cv::Mat HybridBackend::processWithOpenCV(const cv::Mat& input, const std::string& operation) {
    // This is a placeholder - implement specific operations as needed
    cv::Mat output;
    
    if (current_backend_ == BackendType::OPENCV_OPENCL && opencv_opencl_available_) {
        // Use OpenCV OpenCL
        cv::UMat u_input = input.getUMat(cv::ACCESS_READ);
        cv::UMat u_output;
        
        // Apply operation (placeholder)
        u_input.copyTo(u_output);
        
        u_output.copyTo(output);
    } else {
        // Use OpenCV CPU
        input.copyTo(output);
    }
    
    return output;
}

Halide::Buffer<float> HybridBackend::processWithHalide(const Halide::Buffer<float>& input, const std::string& operation) {
    // This is a placeholder - implement specific operations as needed
    Halide::Buffer<float> output(input.width(), input.height(), input.channels());
    
    if (current_backend_ == BackendType::HALIDE_OPENCL && halide_opencl_available_) {
        // Use Halide OpenCL
        output.set_host_dirty();
        output.copy_to_device(Halide::Target::OpenCL);
        // Apply operation here
        output.copy_to_host();
    } else {
        // Use Halide CPU
        // Apply operation here
    }
    
    return output;
}

Halide::Buffer<float> HybridBackend::cvMatToHalide(const cv::Mat& cv_mat) {
    // Convert OpenCV Mat to Halide Buffer
    int width = cv_mat.cols;
    int height = cv_mat.rows;
    int channels = cv_mat.channels();
    
    Halide::Buffer<float> halide_buffer(width, height, channels);
    
    // Copy data
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                halide_buffer(x, y, c) = cv_mat.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    
    return halide_buffer;
}

cv::Mat HybridBackend::halideToCvMat(const Halide::Buffer<float>& halide_buffer) {
    // Convert Halide Buffer to OpenCV Mat
    int width = halide_buffer.width();
    int height = halide_buffer.height();
    int channels = halide_buffer.channels();
    
    cv::Mat cv_mat(height, width, CV_32FC(channels));
    
    // Copy data
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cv_mat.at<cv::Vec3f>(y, x)[c] = halide_buffer(x, y, c);
            }
        }
    }
    
    return cv_mat;
}

BackendType HybridBackend::selectBestBackend() const {
    // Priority order: Halide OpenCL > OpenCV OpenCL > Halide CPU > OpenCV CPU
    if (halide_opencl_available_) {
        return BackendType::HALIDE_OPENCL;
    } else if (opencv_opencl_available_) {
        return BackendType::OPENCV_OPENCL;
    } else {
        return BackendType::HALIDE_CPU; // Halide CPU is generally faster than OpenCV CPU
    }
}

bool HybridBackend::initializeOpenCVOpenCL() {
    try {
        // Check if OpenCV was built with OpenCL support
        if (!cv::ocl::haveOpenCL()) {
            return false;
        }
        
        // Check if OpenCL device is available
        cv::ocl::Context context;
        if (context.empty()) {
            return false;
        }
        
        // Enable OpenCL for OpenCV
        cv::ocl::setUseOpenCL(true);
        
        return true;
    } catch (...) {
        return false;
    }
}

bool HybridBackend::initializeHalideOpenCL() {
    try {
        // Check if Halide OpenCL target is available
        Halide::Target target = Halide::Target::OpenCL;
        return target.supported();
    } catch (...) {
        return false;
    }
}

} // namespace hdr_isp 
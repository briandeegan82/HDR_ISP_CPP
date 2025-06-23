#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>

int main() {
    std::cout << "Testing OpenCV OpenCL support..." << std::endl;
    
    // Check if OpenCV was built with OpenCL support
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    // Check if OpenCL is available
    if (cv::ocl::haveOpenCL()) {
        std::cout << "✓ OpenCV OpenCL support: AVAILABLE" << std::endl;
        
        // Get OpenCL context
        cv::ocl::Context context;
        if (!context.empty()) {
            std::cout << "✓ OpenCL context created successfully" << std::endl;
            
            // Get device info
            cv::ocl::Device device = context.device(0);
            std::cout << "✓ OpenCL device: " << device.name() << std::endl;
            std::cout << "  - Compute units: " << device.maxComputeUnits() << std::endl;
            std::cout << "  - Global memory: " << device.globalMemSize() / (1024*1024) << " MB" << std::endl;
            
            // Enable OpenCL for OpenCV
            cv::ocl::setUseOpenCL(true);
            std::cout << "✓ OpenCL enabled for OpenCV operations" << std::endl;
            
            // Test a simple operation
            cv::Mat test_image = cv::Mat::ones(100, 100, CV_8UC1) * 128;
            cv::Mat result;
            
            // This should use OpenCL if available
            cv::GaussianBlur(test_image, result, cv::Size(5, 5), 0);
            std::cout << "✓ Gaussian blur test completed" << std::endl;
            
        } else {
            std::cout << "✗ Failed to create OpenCL context" << std::endl;
        }
    } else {
        std::cout << "✗ OpenCV OpenCL support: NOT AVAILABLE" << std::endl;
        std::cout << "  This means OpenCV was not built with OpenCL support" << std::endl;
    }
    
    return 0;
} 
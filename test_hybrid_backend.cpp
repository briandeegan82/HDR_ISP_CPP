#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/isp_backend_wrapper.hpp"
#include "include/module_ab_test.hpp"
#include "include/hybrid_backend.hpp"

int main() {
    std::cout << "=== HDR ISP Hybrid Backend Test ===" << std::endl;
    
    // Create a test image
    cv::Mat test_image = cv::Mat::zeros(1024, 1024, CV_8UC3);
    cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    std::cout << "Test image created: " << test_image.cols << "x" << test_image.rows << std::endl;
    
    // Initialize hybrid backend
#ifdef USE_HYBRID_BACKEND
    std::cout << "\n--- Hybrid Backend Initialization ---" << std::endl;
    if (hdr_isp::g_backend) {
        bool success = hdr_isp::g_backend->initialize(hdr_isp::BackendType::AUTO);
        if (success) {
            std::cout << "Hybrid backend initialized successfully" << std::endl;
            std::cout << "Current backend: " << hdr_isp::g_backend->getBackendName() << std::endl;
        } else {
            std::cout << "Failed to initialize hybrid backend" << std::endl;
        }
    } else {
        std::cout << "Hybrid backend not available" << std::endl;
    }
#else
    std::cout << "Hybrid backend not enabled (USE_HYBRID_BACKEND=OFF)" << std::endl;
#endif
    
    // Test ISPBackendWrapper functionality
    std::cout << "\n--- ISPBackendWrapper Tests ---" << std::endl;
    
    // Test Gaussian Blur
    std::cout << "Testing Gaussian Blur..." << std::endl;
    hdr_isp::ISPBackendWrapper::startTimer();
    cv::Mat blurred = hdr_isp::ISPBackendWrapper::gaussianBlur(test_image, cv::Size(15, 15), 2.0);
    double blur_time = hdr_isp::ISPBackendWrapper::endTimer();
    std::cout << "Gaussian blur time: " << blur_time << "ms" << std::endl;
    
    // Test Color Space Conversion
    std::cout << "Testing Color Space Conversion..." << std::endl;
    hdr_isp::ISPBackendWrapper::startTimer();
    cv::Mat converted = hdr_isp::ISPBackendWrapper::colorSpaceConversion(test_image, cv::COLOR_BGR2YUV);
    double conv_time = hdr_isp::ISPBackendWrapper::endTimer();
    std::cout << "Color space conversion time: " << conv_time << "ms" << std::endl;
    
    // Test Resize
    std::cout << "Testing Resize..." << std::endl;
    hdr_isp::ISPBackendWrapper::startTimer();
    cv::Mat resized = hdr_isp::ISPBackendWrapper::resize(test_image, cv::Size(512, 512));
    double resize_time = hdr_isp::ISPBackendWrapper::endTimer();
    std::cout << "Resize time: " << resize_time << "ms" << std::endl;
    
    // Test Matrix Multiplication
    std::cout << "Testing Matrix Multiplication..." << std::endl;
    cv::Mat matrix = (cv::Mat_<float>(3, 3) << 0.299, 0.587, 0.114,
                                               0.596, -0.274, -0.321,
                                               0.211, -0.523, 0.312);
    hdr_isp::ISPBackendWrapper::startTimer();
    cv::Mat multiplied = hdr_isp::ISPBackendWrapper::matrixMultiplication(test_image.reshape(1, test_image.total()), matrix);
    double mult_time = hdr_isp::ISPBackendWrapper::endTimer();
    std::cout << "Matrix multiplication time: " << mult_time << "ms" << std::endl;
    
    // Test Convolution
    std::cout << "Testing Convolution..." << std::endl;
    cv::Mat kernel = cv::getGaussianKernel(5, 1.0) * cv::getGaussianKernel(5, 1.0).t();
    hdr_isp::ISPBackendWrapper::startTimer();
    cv::Mat convolved = hdr_isp::ISPBackendWrapper::convolution(test_image, kernel);
    double conv_time2 = hdr_isp::ISPBackendWrapper::endTimer();
    std::cout << "Convolution time: " << conv_time2 << "ms" << std::endl;
    
    // Backend information
    std::cout << "\n--- Backend Information ---" << std::endl;
    std::cout << "Current backend: " << hdr_isp::ISPBackendWrapper::getCurrentBackend() << std::endl;
    std::cout << "Optimized backend available: " << (hdr_isp::ISPBackendWrapper::isOptimizedBackendAvailable() ? "Yes" : "No") << std::endl;
    
    // Test A/B testing framework
    std::cout << "\n--- A/B Testing Framework ---" << std::endl;
    
    // Create a simple test case
    cv::Mat test_input = cv::Mat::zeros(512, 512, CV_8UC3);
    cv::randu(test_input, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    // Test output comparison
    cv::Mat original_output = test_input.clone();
    cv::Mat hybrid_output = test_input.clone();
    
    bool comparison_result = hdr_isp::ModuleABTest::compareOutputs(original_output, hybrid_output);
    std::cout << "Output comparison test: " << (comparison_result ? "PASSED" : "FAILED") << std::endl;
    
    // Test performance summary
    std::vector<hdr_isp::ABTestResult> test_results;
    hdr_isp::ABTestResult test_result;
    test_result.module_name = "Test Module";
    test_result.original_time_ms = 100.0;
    test_result.hybrid_time_ms = 50.0;
    test_result.speedup_factor = 2.0;
    test_result.output_match = true;
    test_results.push_back(test_result);
    
    hdr_isp::ModuleABTest::printPerformanceSummary(test_results);
    
    std::cout << "\n=== Test Completed Successfully ===" << std::endl;
    
    return 0;
} 
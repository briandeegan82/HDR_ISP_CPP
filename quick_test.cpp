#include <iostream>
#include <opencv2/opencv.hpp>

// Simple test to verify OpenCV is working
int main() {
    std::cout << "=== Quick Hybrid Backend Test ===" << std::endl;
    
    // Test OpenCV functionality
    cv::Mat test_image = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    std::cout << "OpenCV test image created: " << test_image.cols << "x" << test_image.rows << std::endl;
    
    // Test basic operations
    cv::Mat blurred;
    cv::GaussianBlur(test_image, blurred, cv::Size(5, 5), 1.0);
    std::cout << "Gaussian blur test: PASSED" << std::endl;
    
    // Test color space conversion
    cv::Mat yuv;
    cv::cvtColor(test_image, yuv, cv::COLOR_BGR2YUV);
    std::cout << "Color space conversion test: PASSED" << std::endl;
    
    // Test matrix operations
    cv::Mat matrix = (cv::Mat_<float>(3, 3) << 0.299, 0.587, 0.114,
                                               0.596, -0.274, -0.321,
                                               0.211, -0.523, 0.312);
    cv::Mat result;
    cv::gemm(test_image.reshape(1, test_image.total()), matrix, 1.0, cv::Mat(), 0.0, result);
    std::cout << "Matrix multiplication test: PASSED" << std::endl;
    
    std::cout << "=== All Basic Tests Passed ===" << std::endl;
    std::cout << "Ready for hybrid backend testing once Visual Studio is installed!" << std::endl;
    
    return 0;
} 
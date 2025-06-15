#include "gray_world.hpp"
#include <numeric>

GrayWorld::GrayWorld(const cv::Mat& flatten_img)
    : flatten_img_(flatten_img) {
}

std::tuple<double, double> GrayWorld::calculate_gains() {
    // Calculate mean values for each channel
    cv::Scalar avg_rgb = cv::mean(flatten_img_);
    
    // Calculate white balance gains G/R and G/B
    double rgain = std::isnan(avg_rgb[1] / avg_rgb[0]) ? 0.0 : avg_rgb[1] / avg_rgb[0];
    double bgain = std::isnan(avg_rgb[1] / avg_rgb[2]) ? 0.0 : avg_rgb[1] / avg_rgb[2];
    
    return {rgain, bgain};
} 
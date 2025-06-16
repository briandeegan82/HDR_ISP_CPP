#include "gray_world.hpp"
#include <numeric>

GrayWorld::GrayWorld(const cv::Mat& flatten_img)
    : flatten_img_(flatten_img) {
}

std::tuple<double, double> GrayWorld::calculate_gains() {
    // Calculate mean values for each channel
    cv::Scalar avg_rgb = cv::mean(flatten_img_);
    
    if (avg_rgb[0] == 0 || avg_rgb[1] == 0 || avg_rgb[2] == 0) {
        return {1.0, 1.0};  // Return neutral gains if any channel is zero
    }
    
    // Calculate white balance gains G/R and G/B
    double rgain = avg_rgb[1] / avg_rgb[0];
    double bgain = avg_rgb[1] / avg_rgb[2];
    
    // Ensure gains are reasonable
    rgain = std::min(std::max(rgain, 0.5), 2.0);
    bgain = std::min(std::max(bgain, 0.5), 2.0);
    
    return {rgain, bgain};
} 
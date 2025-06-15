#include "norm_gray_world.hpp"
#include <numeric>
#include <cmath>

NormGrayWorld::NormGrayWorld(const cv::Mat& flatten_img)
    : flatten_img_(flatten_img) {
}

std::tuple<double, double> NormGrayWorld::calculate_gains() {
    // Calculate norm-2 values for each channel
    std::vector<double> avg_rgb(3);
    for (int c = 0; c < 3; ++c) {
        cv::Mat channel = flatten_img_.col(c);
        avg_rgb[c] = std::sqrt(cv::sum(channel.mul(channel))[0]);
    }
    
    // Calculate white balance gains G/R and G/B
    double rgain = std::isnan(avg_rgb[1] / avg_rgb[0]) ? 0.0 : avg_rgb[1] / avg_rgb[0];
    double bgain = std::isnan(avg_rgb[1] / avg_rgb[2]) ? 0.0 : avg_rgb[1] / avg_rgb[2];
    
    return {rgain, bgain};
} 
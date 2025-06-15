#pragma once

#include <opencv2/opencv.hpp>
#include <tuple>

class PCAIlluminEstimation {
public:
    PCAIlluminEstimation(const cv::Mat& flatten_img, float pixel_percentage);
    std::tuple<double, double> calculate_gains();

private:
    cv::Mat flatten_img_;
    float pixel_percentage_;
}; 
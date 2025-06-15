#pragma once

#include <opencv2/opencv.hpp>
#include <tuple>

class NormGrayWorld {
public:
    NormGrayWorld(const cv::Mat& flatten_img);
    std::tuple<double, double> calculate_gains();

private:
    cv::Mat flatten_img_;
}; 
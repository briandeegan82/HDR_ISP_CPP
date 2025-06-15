#pragma once

#include <opencv2/opencv.hpp>

namespace hdr_isp {

class HDRDurand {
public:
    HDRDurand(float sigma = 0.4f, float base_contrast = 5.0f);
    ~HDRDurand() = default;

    // Process the HDR image using Durand's algorithm
    cv::Mat process(const cv::Mat& hdr_image);

private:
    float sigma_;
    float base_contrast_;
    
    // Helper functions
    cv::Mat compute_luminance(const cv::Mat& input);
    cv::Mat compute_log_luminance(const cv::Mat& luminance);
    cv::Mat compute_bilateral_filter(const cv::Mat& input);
    cv::Mat compute_detail_layer(const cv::Mat& log_luminance, const cv::Mat& base_layer);
};

} // namespace hdr_isp 
#pragma once

#include <opencv2/opencv.hpp>

namespace hdr_isp {

class LDCI {
public:
    LDCI(int clip_limit = 2, int grid_size = 8);
    ~LDCI() = default;

    // Process the image using Contrast Limited Adaptive Histogram Equalization
    cv::Mat process(const cv::Mat& input);

private:
    int clip_limit_;
    int grid_size_;
    
    // Helper functions
    cv::Mat compute_histogram(const cv::Mat& tile);
    cv::Mat apply_clahe_tile(const cv::Mat& tile, const cv::Mat& histogram);
};

} // namespace hdr_isp 
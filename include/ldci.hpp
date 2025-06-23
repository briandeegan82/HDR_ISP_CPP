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
    int clip_limit_;      // CLAHE clip limit parameter
    int grid_size_;       // CLAHE tile grid size
    
    // Helper functions for CLAHE processing
    cv::Mat apply_clahe_to_lab(const cv::Mat& input);
    cv::Mat normalize_for_clahe(const cv::Mat& input);
    cv::Mat denormalize_from_clahe(const cv::Mat& input, double original_max);
};

} // namespace hdr_isp 
#pragma once

#include <opencv2/opencv.hpp>

namespace hdr_isp {

class ColorSpaceConversion {
public:
    enum class Standard {
        BT601,
        BT709
    };

    ColorSpaceConversion(Standard standard = Standard::BT709);
    ~ColorSpaceConversion() = default;

    // Convert between different color spaces
    cv::Mat rgb_to_yuv(const cv::Mat& rgb);
    cv::Mat yuv_to_rgb(const cv::Mat& yuv);
    cv::Mat rgb_to_ycbcr(const cv::Mat& rgb);
    cv::Mat ycbcr_to_rgb(const cv::Mat& ycbcr);

private:
    Standard standard_;
    
    // Conversion matrices
    cv::Mat rgb_to_yuv_matrix_;
    cv::Mat yuv_to_rgb_matrix_;
    cv::Mat rgb_to_ycbcr_matrix_;
    cv::Mat ycbcr_to_rgb_matrix_;

    void initialize_matrices();
};

} // namespace hdr_isp 
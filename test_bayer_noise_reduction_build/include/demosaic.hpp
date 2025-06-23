#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <array>

namespace hdr_isp {

class Demosaic {
public:
    Demosaic(const cv::Mat& img, const std::string& bayer_pattern, int bit_depth, bool is_save);
    ~Demosaic() = default;

    cv::Mat execute();

private:
    // Member variables
    cv::Mat img_;
    std::string bayer_pattern_;
    int bit_depth_;
    bool is_save_;

    // Helper methods
    std::array<cv::Mat, 3> masks_cfa_bayer();
    cv::Mat apply_cfa();
    void save();
};

class Malvar {
public:
    Malvar(const cv::Mat& raw_in, const std::array<cv::Mat, 3>& masks);
    ~Malvar() = default;

    cv::Mat apply_malvar();

private:
    // Member variables
    cv::Mat img_;
    std::array<cv::Mat, 3> masks_;

    // Filter coefficients
    static const cv::Mat g_at_r_and_b;
    static const cv::Mat r_at_gr_and_b_at_gb;
    static const cv::Mat r_at_gb_and_b_at_gr;
    static const cv::Mat r_at_b_and_b_at_r;
};

} // namespace hdr_isp 
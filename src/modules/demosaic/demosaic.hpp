#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <array>

enum class DemosaicAlgorithm {
    MALVAR,
    OPENCV
};

class Malvar {
public:
    Malvar(const cv::Mat& raw_in, const std::array<cv::Mat, 3>& masks);
    cv::Mat apply_malvar();

private:
    cv::Mat img_;
    std::array<cv::Mat, 3> masks_;

    // Filter coefficients
    static const cv::Mat g_at_r_and_b;
    static const cv::Mat r_at_gr_and_b_at_gb;
    static const cv::Mat r_at_gb_and_b_at_gr;
    static const cv::Mat r_at_b_and_b_at_r;
};

class Demosaic {
public:
    Demosaic(const cv::Mat& img, const std::string& bayer_pattern, int bit_depth = 16, bool is_save = true, DemosaicAlgorithm algorithm = DemosaicAlgorithm::MALVAR);
    cv::Mat execute();

private:
    cv::Mat img_;
    std::string bayer_pattern_;
    int bit_depth_;
    bool is_save_;
    DemosaicAlgorithm algorithm_;

    std::array<cv::Mat, 3> masks_cfa_bayer();
    cv::Mat apply_cfa();
    cv::Mat apply_opencv_demosaic();
    void save();
}; 
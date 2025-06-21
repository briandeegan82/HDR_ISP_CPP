#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <array>
#include "../../common/eigen_utils.hpp"

enum class DemosaicAlgorithm {
    MALVAR,
    OPENCV
};

class Malvar {
public:
    Malvar(const cv::Mat& raw_in, const std::array<cv::Mat, 3>& masks);
    cv::Mat apply_malvar();
    
    // Eigen version
    Malvar(const hdr_isp::EigenImage& raw_in, const std::array<hdr_isp::EigenImage, 3>& masks);
    hdr_isp::EigenImage apply_malvar_eigen();

private:
    cv::Mat img_;
    std::array<cv::Mat, 3> masks_;
    
    // Eigen members
    hdr_isp::EigenImage eigen_img_;
    std::array<hdr_isp::EigenImage, 3> eigen_masks_;
    bool use_eigen_;

    // Filter coefficients
    static const cv::Mat g_at_r_and_b;
    static const cv::Mat r_at_gr_and_b_at_gb;
    static const cv::Mat r_at_gb_and_b_at_gr;
    static const cv::Mat r_at_b_and_b_at_r;
    
    // Eigen filter coefficients
    static const Eigen::MatrixXf eigen_g_at_r_and_b;
    static const Eigen::MatrixXf eigen_r_at_gr_and_b_at_gb;
    static const Eigen::MatrixXf eigen_r_at_gb_and_b_at_gr;
    static const Eigen::MatrixXf eigen_r_at_b_and_b_at_r;
    
    // Helper function for Eigen convolution
    hdr_isp::EigenImage apply_convolution_eigen(const hdr_isp::EigenImage& img, const Eigen::MatrixXf& kernel);
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
    bool is_debug_;
    bool is_enable_;
    DemosaicAlgorithm algorithm_;
    bool use_eigen_; // Use Eigen by default

    std::array<cv::Mat, 3> masks_cfa_bayer();
    std::array<hdr_isp::EigenImage, 3> masks_cfa_bayer_eigen();
    cv::Mat apply_cfa();
    hdr_isp::EigenImage apply_cfa_eigen();
    cv::Mat apply_opencv_demosaic();
    void save();
}; 
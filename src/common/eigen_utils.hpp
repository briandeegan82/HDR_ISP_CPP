#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

namespace hdr_isp {

// Forward declarations
class EigenImage;
class EigenImage3C;

/**
 * @brief Single-channel Eigen-based image class
 * 
 * This class provides a wrapper around Eigen::MatrixXf for single-channel images,
 * with utilities for conversion to/from OpenCV Mat.
 */
class EigenImage {
public:
    EigenImage() = default;
    EigenImage(int rows, int cols) : data_(rows, cols) {}
    EigenImage(const Eigen::MatrixXf& matrix) : data_(matrix) {}
    
    // Conversion from OpenCV Mat
    static EigenImage fromOpenCV(const cv::Mat& mat);
    
    // Conversion to OpenCV Mat
    cv::Mat toOpenCV(int opencv_type = CV_32F) const;
    
    // Access to underlying data
    Eigen::MatrixXf& data() { return data_; }
    const Eigen::MatrixXf& data() const { return data_; }
    
    // Basic operations
    int rows() const { return data_.rows(); }
    int cols() const { return data_.cols(); }
    int size() const { return data_.size(); }
    
    // Static factory methods
    static EigenImage Zero(int rows, int cols) { return EigenImage(Eigen::MatrixXf::Zero(rows, cols)); }
    
    // Element-wise operations
    EigenImage operator*(float scalar) const { return EigenImage(data_ * scalar); }
    EigenImage operator+(const EigenImage& other) const { return EigenImage(data_ + other.data_); }
    EigenImage operator-(const EigenImage& other) const { return EigenImage(data_ - other.data_); }
    
    // Clipping operations
    EigenImage clip(float min_val, float max_val) const;
    EigenImage cwiseMax(float val) const { return EigenImage(data_.cwiseMax(val)); }
    EigenImage cwiseMin(float val) const { return EigenImage(data_.cwiseMin(val)); }
    
    // Block operations
    EigenImage block(int startRow, int startCol, int numRows, int numCols) const {
        return EigenImage(data_.block(startRow, startCol, numRows, numCols));
    }
    
    // Reshape operations
    EigenImage reshaped(int rows, int cols) const {
        return EigenImage(Eigen::Map<const Eigen::MatrixXf>(data_.data(), rows, cols));
    }
    
    // Statistics
    float min() const { return data_.minCoeff(); }
    float max() const { return data_.maxCoeff(); }
    float mean() const { return data_.mean(); }
    
    // Assignment operators
    EigenImage& operator=(const EigenImage& other) {
        data_ = other.data_;
        return *this;
    }
    
    // Element access
    float& operator()(int i, int j) { return data_(i, j); }
    const float& operator()(int i, int j) const { return data_(i, j); }
    
private:
    Eigen::MatrixXf data_;
};

/**
 * @brief Three-channel Eigen-based image class
 * 
 * This class provides a wrapper around three Eigen::MatrixXf for RGB/YUV images,
 * with utilities for conversion to/from OpenCV Mat.
 */
class EigenImage3C {
public:
    EigenImage3C() = default;
    EigenImage3C(int rows, int cols) 
        : r_(rows, cols), g_(rows, cols), b_(rows, cols) {}
    
    // Conversion from OpenCV Mat
    static EigenImage3C fromOpenCV(const cv::Mat& mat);
    
    // Conversion to OpenCV Mat
    cv::Mat toOpenCV(int opencv_type = CV_32FC3) const;
    
    // Access to channels
    EigenImage& r() { return r_; }
    EigenImage& g() { return g_; }
    EigenImage& b() { return b_; }
    const EigenImage& r() const { return r_; }
    const EigenImage& g() const { return g_; }
    const EigenImage& b() const { return b_; }
    
    // Basic operations
    int rows() const { return r_.rows(); }
    int cols() const { return r_.cols(); }
    
    // Element-wise operations
    EigenImage3C operator*(float scalar) const;
    EigenImage3C operator*(const Eigen::Vector3f& gains) const;
    
    // Clipping operations
    EigenImage3C clip(float min_val, float max_val) const;
    
    // Matrix multiplication (for color space conversion)
    EigenImage3C operator*(const Eigen::Matrix3f& matrix) const;
    
private:
    EigenImage r_, g_, b_;
};

// Utility functions
namespace eigen_utils {
    
    /**
     * @brief Convert OpenCV Mat to Eigen::MatrixXf
     */
    Eigen::MatrixXf matToEigen(const cv::Mat& mat);
    
    /**
     * @brief Convert Eigen::MatrixXf to OpenCV Mat
     */
    cv::Mat eigenToMat(const Eigen::MatrixXf& eigen_mat, int opencv_type = CV_32F);
    
    /**
     * @brief Convert OpenCV Mat to Eigen::Matrix3f
     */
    Eigen::Matrix3f mat3x3ToEigen(const cv::Mat& mat);
    
    /**
     * @brief Convert Eigen::Matrix3f to OpenCV Mat
     */
    cv::Mat eigenToMat3x3(const Eigen::Matrix3f& eigen_mat);
    
    /**
     * @brief Reshape Eigen matrix to different dimensions
     */
    Eigen::MatrixXf reshape(const Eigen::MatrixXf& mat, int rows, int cols);
    
    /**
     * @brief Apply Bayer pattern mask to image
     */
    EigenImage applyBayerMask(const EigenImage& img, const std::string& bayer_pattern, int channel);
    
    /**
     * @brief Create Bayer pattern masks
     */
    std::vector<EigenImage> createBayerMasks(int rows, int cols, const std::string& bayer_pattern);
}

// Global utility functions for backward compatibility
/**
 * @brief Convert OpenCV Mat to EigenImage
 */
EigenImage opencv_to_eigen(const cv::Mat& mat);

/**
 * @brief Convert EigenImage to OpenCV Mat
 */
cv::Mat eigen_to_opencv(const EigenImage& eigen_img);

} // namespace hdr_isp 
#pragma once

/**
 * @file eigen_utils.hpp
 * @brief Eigen-based image utilities for HDR ISP pipeline
 * 
 * This file provides wrapper classes around Eigen matrices for image processing,
 * with utilities for conversion to/from OpenCV Mat. The classes handle Eigen::Index
 * to int conversions with bounds checking to prevent data loss and suppress
 * compiler warnings about potential truncation.
 */

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <limits>

namespace hdr_isp {

// Forward declarations
class EigenImage;
class EigenImage3C;
class EigenImageU32;

// Type aliases for backward compatibility
using EigenImage32 = EigenImageU32;

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
    int rows() const { 
        Eigen::Index rows = data_.rows();
        if (rows > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Matrix rows too large for int conversion");
        }
        return static_cast<int>(rows); 
    }
    int cols() const { 
        Eigen::Index cols = data_.cols();
        if (cols > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Matrix cols too large for int conversion");
        }
        return static_cast<int>(cols); 
    }
    int size() const { 
        Eigen::Index size = data_.size();
        if (size > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Matrix size too large for int conversion");
        }
        return static_cast<int>(size); 
    }
    
    // Static factory methods
    static EigenImage Zero(int rows, int cols) { return EigenImage(Eigen::MatrixXf::Zero(rows, cols)); }
    static EigenImage Ones(int rows, int cols) { return EigenImage(Eigen::MatrixXf::Ones(rows, cols)); }
    static EigenImage Constant(int rows, int cols, float value) { return EigenImage(Eigen::MatrixXf::Constant(rows, cols, value)); }
    
    // Element-wise operations
    EigenImage operator*(float scalar) const { return EigenImage(data_ * scalar); }
    EigenImage operator+(const EigenImage& other) const { return EigenImage(data_ + other.data_); }
    EigenImage operator-(const EigenImage& other) const { return EigenImage(data_ - other.data_); }
    EigenImage operator+(float scalar) const { return EigenImage(data_.array() + scalar); }
    EigenImage operator-(float scalar) const { return EigenImage(data_.array() - scalar); }
    EigenImage operator/(float scalar) const { return EigenImage(data_ / scalar); }
    
    // Element-wise product
    EigenImage cwiseProduct(const EigenImage& other) const { return EigenImage(data_.cwiseProduct(other.data_)); }
    
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
    
    EigenImage& operator=(float scalar) {
        data_.setConstant(scalar);
        return *this;
    }
    
    EigenImage& operator+=(const EigenImage& other) {
        data_ += other.data_;
        return *this;
    }
    
    EigenImage& operator-=(const EigenImage& other) {
        data_ -= other.data_;
        return *this;
    }
    
    EigenImage& operator*=(float scalar) {
        data_ *= scalar;
        return *this;
    }
    
    EigenImage& operator/=(float scalar) {
        data_ /= scalar;
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
    EigenImage3C(const EigenImage& r, const EigenImage& g, const EigenImage& b)
        : r_(r), g_(g), b_(b) {}
    
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
    int rows() const { 
        Eigen::Index rows = r_.rows();
        if (rows > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Matrix rows too large for int conversion");
        }
        return static_cast<int>(rows); 
    }
    int cols() const { 
        Eigen::Index cols = r_.cols();
        if (cols > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Matrix cols too large for int conversion");
        }
        return static_cast<int>(cols); 
    }
    
    // Element-wise operations
    EigenImage3C operator*(float scalar) const;
    EigenImage3C operator*(const Eigen::Vector3f& gains) const;
    
    // Clipping operations
    EigenImage3C clip(float min_val, float max_val) const;
    
    // Matrix multiplication (for color space conversion)
    EigenImage3C operator*(const Eigen::Matrix3f& matrix) const;
    
    // Clone method
    EigenImage3C clone() const { return EigenImage3C(r_, g_, b_); }
    
    // Statistics methods (to resolve compiler errors in LDCI)
    float min() const { 
        return std::min({r_.min(), g_.min(), b_.min()}); 
    }
    float max() const { 
        return std::max({r_.max(), g_.max(), b_.max()}); 
    }
    float mean() const { 
        return (r_.mean() + g_.mean() + b_.mean()) / 3.0f; 
    }
    
private:
    EigenImage r_, g_, b_;
};

/**
 * @brief Single-channel Eigen-based image class for 32-bit unsigned integer processing
 * 
 * This class provides a wrapper around Eigen::Matrix<uint32_t> for single-channel images,
 * optimized for HDR pipeline stages that need unsigned integer precision.
 */
class EigenImageU32 {
public:
    EigenImageU32() = default;
    EigenImageU32(int rows, int cols) : data_(rows, cols) {}
    EigenImageU32(const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>& matrix) : data_(matrix) {}
    
    // Conversion from OpenCV Mat
    static EigenImageU32 fromOpenCV(const cv::Mat& mat);
    
    // Conversion to OpenCV Mat
    cv::Mat toOpenCV(int opencv_type = CV_32SC1) const;
    
    // Access to underlying data
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>& data() { return data_; }
    const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>& data() const { return data_; }
    
    // Basic operations
    int rows() const { 
        Eigen::Index rows = data_.rows();
        if (rows > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Matrix rows too large for int conversion");
        }
        return static_cast<int>(rows); 
    }
    int cols() const { 
        Eigen::Index cols = data_.cols();
        if (cols > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Matrix cols too large for int conversion");
        }
        return static_cast<int>(cols); 
    }
    int size() const { 
        Eigen::Index size = data_.size();
        if (size > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Matrix size too large for int conversion");
        }
        return static_cast<int>(size); 
    }
    
    // Static factory methods
    static EigenImageU32 Zero(int rows, int cols) { 
        return EigenImageU32(Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols)); 
    }
    static EigenImageU32 Ones(int rows, int cols) { 
        return EigenImageU32(Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(rows, cols)); 
    }
    static EigenImageU32 Constant(int rows, int cols, uint32_t value) { 
        return EigenImageU32(Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>::Constant(rows, cols, value)); 
    }
    
    // Element-wise operations
    EigenImageU32 operator*(uint32_t scalar) const { return EigenImageU32(data_ * scalar); }
    EigenImageU32 operator+(const EigenImageU32& other) const { return EigenImageU32(data_ + other.data_); }
    EigenImageU32 operator-(const EigenImageU32& other) const { 
        // Handle underflow by clamping to 0 using element-wise max
        // Use array operations to avoid Eigen's binary operator issues
        Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> result = data_.array().max(other.data_.array()) - other.data_.array();
        return EigenImageU32(result);
    }
    EigenImageU32 operator+(uint32_t scalar) const { return EigenImageU32(data_.array() + scalar); }
    EigenImageU32 operator-(uint32_t scalar) const { 
        // Handle underflow by clamping to 0
        return EigenImageU32(data_.array().max(scalar) - scalar);
    }
    EigenImageU32 operator/(uint32_t scalar) const { return EigenImageU32(data_ / scalar); }
    
    // Element-wise product
    EigenImageU32 cwiseProduct(const EigenImageU32& other) const { return EigenImageU32(data_.cwiseProduct(other.data_)); }
    
    // Clipping operations
    EigenImageU32 clip(uint32_t min_val, uint32_t max_val) const;
    EigenImageU32 cwiseMax(uint32_t val) const { return EigenImageU32(data_.cwiseMax(val)); }
    EigenImageU32 cwiseMin(uint32_t val) const { return EigenImageU32(data_.cwiseMin(val)); }
    
    // Block operations
    EigenImageU32 block(int startRow, int startCol, int numRows, int numCols) const {
        return EigenImageU32(data_.block(startRow, startCol, numRows, numCols));
    }
    
    // Statistics
    uint32_t min() const { return data_.minCoeff(); }
    uint32_t max() const { return data_.maxCoeff(); }
    float mean() const { 
        // Use double precision to avoid overflow in large matrices
        double sum = data_.cast<double>().sum();
        return static_cast<float>(sum / data_.size());
    }
    
    // Assignment operators
    EigenImageU32& operator=(const EigenImageU32& other) {
        data_ = other.data_;
        return *this;
    }
    
    EigenImageU32& operator=(uint32_t scalar) {
        data_.setConstant(scalar);
        return *this;
    }
    
    EigenImageU32& operator+=(const EigenImageU32& other) {
        data_ += other.data_;
        return *this;
    }
    
    EigenImageU32& operator-=(const EigenImageU32& other) {
        // Handle underflow by clamping to 0
        data_ = data_.array().max(other.data_.array()) - other.data_.array();
        return *this;
    }
    
    EigenImageU32& operator*=(uint32_t scalar) {
        data_ *= scalar;
        return *this;
    }
    
    EigenImageU32& operator/=(uint32_t scalar) {
        data_ /= scalar;
        return *this;
    }
    
    // Element access
    uint32_t& operator()(int i, int j) { return data_(i, j); }
    const uint32_t& operator()(int i, int j) const { return data_(i, j); }
    
    // Clone method
    EigenImageU32 clone() const { return EigenImageU32(data_); }
    
    // Bayer pattern operations
    EigenImageU32 extractBayerChannel(const std::string& bayer_pattern, char channel) const;
    
private:
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> data_;
};

/**
 * @brief Three-channel Eigen-based image class for fixed-point processing
 * 
 * This class provides a wrapper around three Eigen::Matrix<int16_t> for RGB/YUV images
 * with fixed-point arithmetic, optimized for modules after demosaic.
 */
class EigenImage3CFixed {
public:
    EigenImage3CFixed() = default;
    EigenImage3CFixed(int rows, int cols) 
        : r_(rows, cols), g_(rows, cols), b_(rows, cols) {}
    EigenImage3CFixed(const Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>& r,
                      const Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>& g,
                      const Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>& b)
        : r_(r), g_(g), b_(b) {}
    
    // Conversion from floating-point EigenImage3C
    static EigenImage3CFixed fromEigenImage3C(const EigenImage3C& img, int fractional_bits);
    
    // Conversion to floating-point EigenImage3C
    EigenImage3C toEigenImage3C(int fractional_bits) const;
    
    // Conversion to OpenCV Mat (for saving/display)
    cv::Mat toOpenCV(int fractional_bits, int opencv_type = CV_32FC3) const;
    
    // Access to channels
    Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>& r() { return r_; }
    Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>& g() { return g_; }
    Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>& b() { return b_; }
    const Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>& r() const { return r_; }
    const Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>& g() const { return g_; }
    const Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>& b() const { return b_; }
    
    // Basic operations
    int rows() const { 
        Eigen::Index rows = r_.rows();
        if (rows > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Matrix rows too large for int conversion");
        }
        return static_cast<int>(rows); 
    }
    int cols() const { 
        Eigen::Index cols = r_.cols();
        if (cols > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Matrix cols too large for int conversion");
        }
        return static_cast<int>(cols); 
    }
    
    // Element-wise operations
    EigenImage3CFixed operator*(int16_t scalar) const;
    EigenImage3CFixed operator*(const Eigen::Vector3i& gains) const;
    
    // Clipping operations
    EigenImage3CFixed clip(int16_t min_val, int16_t max_val) const;
    
    // Matrix multiplication (for color space conversion)
    EigenImage3CFixed operator*(const Eigen::Matrix3i& matrix) const;
    
    // Clone method
    EigenImage3CFixed clone() const { return EigenImage3CFixed(r_, g_, b_); }
    
    // Statistics
    int16_t min() const { 
        return std::min({r_.minCoeff(), g_.minCoeff(), b_.minCoeff()}); 
    }
    int16_t max() const { 
        return std::max({r_.maxCoeff(), g_.maxCoeff(), b_.maxCoeff()}); 
    }
    float mean() const { 
        return (r_.cast<float>().mean() + g_.cast<float>().mean() + b_.cast<float>().mean()) / 3.0f; 
    }
    
private:
    Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic> r_, g_, b_;
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
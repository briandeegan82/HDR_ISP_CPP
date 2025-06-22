#include "eigen_utils.hpp"
#include <iostream>
#include <algorithm>

namespace hdr_isp {

// EigenImage implementation
EigenImage EigenImage::fromOpenCV(const cv::Mat& mat) {
    if (mat.channels() != 1) {
        throw std::runtime_error("EigenImage::fromOpenCV: Input must be single-channel");
    }
    
    cv::Mat float_mat;
    mat.convertTo(float_mat, CV_32F);
    
    Eigen::MatrixXf eigen_mat(float_mat.rows, float_mat.cols);
    for (int i = 0; i < float_mat.rows; ++i) {
        for (int j = 0; j < float_mat.cols; ++j) {
            eigen_mat(i, j) = float_mat.at<float>(i, j);
        }
    }
    
    return EigenImage(eigen_mat);
}

cv::Mat EigenImage::toOpenCV(int opencv_type) const {
    cv::Mat mat(data_.rows(), data_.cols(), opencv_type);
    
    if (opencv_type == CV_32F) {
        for (int i = 0; i < data_.rows(); ++i) {
            for (int j = 0; j < data_.cols(); ++j) {
                mat.at<float>(i, j) = data_(i, j);
            }
        }
    } else if (opencv_type == CV_8U) {
        for (int i = 0; i < data_.rows(); ++i) {
            for (int j = 0; j < data_.cols(); ++j) {
                mat.at<uchar>(i, j) = static_cast<uchar>(std::max(0.0f, std::min(255.0f, data_(i, j))));
            }
        }
    } else if (opencv_type == CV_16U) {
        for (int i = 0; i < data_.rows(); ++i) {
            for (int j = 0; j < data_.cols(); ++j) {
                mat.at<ushort>(i, j) = static_cast<ushort>(std::max(0.0f, std::min(65535.0f, data_(i, j))));
            }
        }
    } else {
        throw std::runtime_error("EigenImage::toOpenCV: Unsupported OpenCV type");
    }
    
    return mat;
}

EigenImage EigenImage::clip(float min_val, float max_val) const {
    Eigen::MatrixXf clipped = data_.cwiseMax(min_val).cwiseMin(max_val);
    return EigenImage(clipped);
}

// EigenImage3C implementation
EigenImage3C EigenImage3C::fromOpenCV(const cv::Mat& mat) {
    if (mat.channels() != 3) {
        throw std::runtime_error("EigenImage3C::fromOpenCV: Input must be three-channel");
    }
    
    std::vector<cv::Mat> channels;
    cv::split(mat, channels);
    
    EigenImage3C result(mat.rows, mat.cols);
    
    result.r_ = EigenImage::fromOpenCV(channels[0]);
    result.g_ = EigenImage::fromOpenCV(channels[1]);
    result.b_ = EigenImage::fromOpenCV(channels[2]);
    
    return result;
}

cv::Mat EigenImage3C::toOpenCV(int opencv_type) const {
    cv::Mat r_mat = r_.toOpenCV(CV_32F);
    cv::Mat g_mat = g_.toOpenCV(CV_32F);
    cv::Mat b_mat = b_.toOpenCV(CV_32F);
    
    std::vector<cv::Mat> channels = {r_mat, g_mat, b_mat};
    cv::Mat result;
    cv::merge(channels, result);
    
    if (opencv_type != CV_32FC3) {
        cv::Mat converted;
        result.convertTo(converted, opencv_type);
        return converted;
    }
    
    return result;
}

EigenImage3C EigenImage3C::operator*(float scalar) const {
    EigenImage3C result(rows(), cols());
    result.r_ = r_ * scalar;
    result.g_ = g_ * scalar;
    result.b_ = b_ * scalar;
    return result;
}

EigenImage3C EigenImage3C::operator*(const Eigen::Vector3f& gains) const {
    EigenImage3C result(rows(), cols());
    result.r_ = r_ * gains(0);
    result.g_ = g_ * gains(1);
    result.b_ = b_ * gains(2);
    return result;
}

EigenImage3C EigenImage3C::clip(float min_val, float max_val) const {
    EigenImage3C result(rows(), cols());
    result.r_ = r_.clip(min_val, max_val);
    result.g_ = g_.clip(min_val, max_val);
    result.b_ = b_.clip(min_val, max_val);
    return result;
}

EigenImage3C EigenImage3C::operator*(const Eigen::Matrix3f& matrix) const {
    EigenImage3C result(rows(), cols());
    
    // Reshape channels to column vectors for matrix multiplication
    Eigen::Map<const Eigen::VectorXf> r_vec(r_.data().data(), r_.size());
    Eigen::Map<const Eigen::VectorXf> g_vec(g_.data().data(), g_.size());
    Eigen::Map<const Eigen::VectorXf> b_vec(b_.data().data(), b_.size());
    
    // Apply color space transformation
    Eigen::VectorXf new_r = matrix(0, 0) * r_vec + matrix(0, 1) * g_vec + matrix(0, 2) * b_vec;
    Eigen::VectorXf new_g = matrix(1, 0) * r_vec + matrix(1, 1) * g_vec + matrix(1, 2) * b_vec;
    Eigen::VectorXf new_b = matrix(2, 0) * r_vec + matrix(2, 1) * g_vec + matrix(2, 2) * b_vec;
    
    // Reshape back to original dimensions
    result.r_ = EigenImage(Eigen::Map<const Eigen::MatrixXf>(new_r.data(), rows(), cols()));
    result.g_ = EigenImage(Eigen::Map<const Eigen::MatrixXf>(new_g.data(), rows(), cols()));
    result.b_ = EigenImage(Eigen::Map<const Eigen::MatrixXf>(new_b.data(), rows(), cols()));
    
    return result;
}

// Utility functions implementation
namespace eigen_utils {

Eigen::MatrixXf matToEigen(const cv::Mat& mat) {
    cv::Mat float_mat;
    mat.convertTo(float_mat, CV_32F);
    
    Eigen::MatrixXf eigen_mat(float_mat.rows, float_mat.cols);
    for (int i = 0; i < float_mat.rows; ++i) {
        for (int j = 0; j < float_mat.cols; ++j) {
            eigen_mat(i, j) = float_mat.at<float>(i, j);
        }
    }
    
    return eigen_mat;
}

cv::Mat eigenToMat(const Eigen::MatrixXf& eigen_mat, int opencv_type) {
    cv::Mat mat(eigen_mat.rows(), eigen_mat.cols(), opencv_type);
    
    if (opencv_type == CV_32F) {
        for (int i = 0; i < eigen_mat.rows(); ++i) {
            for (int j = 0; j < eigen_mat.cols(); ++j) {
                mat.at<float>(i, j) = eigen_mat(i, j);
            }
        }
    } else if (opencv_type == CV_8U) {
        for (int i = 0; i < eigen_mat.rows(); ++i) {
            for (int j = 0; j < eigen_mat.cols(); ++j) {
                mat.at<uchar>(i, j) = static_cast<uchar>(std::max(0.0f, std::min(255.0f, eigen_mat(i, j))));
            }
        }
    } else if (opencv_type == CV_16U) {
        for (int i = 0; i < eigen_mat.rows(); ++i) {
            for (int j = 0; j < eigen_mat.cols(); ++j) {
                mat.at<ushort>(i, j) = static_cast<ushort>(std::max(0.0f, std::min(65535.0f, eigen_mat(i, j))));
            }
        }
    } else {
        throw std::runtime_error("eigenToMat: Unsupported OpenCV type");
    }
    
    return mat;
}

Eigen::Matrix3f mat3x3ToEigen(const cv::Mat& mat) {
    if (mat.rows != 3 || mat.cols != 3) {
        throw std::runtime_error("mat3x3ToEigen: Input must be 3x3 matrix");
    }
    
    cv::Mat float_mat;
    mat.convertTo(float_mat, CV_32F);
    
    Eigen::Matrix3f eigen_mat;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            eigen_mat(i, j) = float_mat.at<float>(i, j);
        }
    }
    
    return eigen_mat;
}

cv::Mat eigenToMat3x3(const Eigen::Matrix3f& eigen_mat) {
    cv::Mat mat(3, 3, CV_32F);
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            mat.at<float>(i, j) = eigen_mat(i, j);
        }
    }
    
    return mat;
}

Eigen::MatrixXf reshape(const Eigen::MatrixXf& mat, int rows, int cols) {
    if (mat.size() != rows * cols) {
        throw std::runtime_error("reshape: Size mismatch");
    }
    
    Eigen::Map<const Eigen::MatrixXf> reshaped(mat.data(), rows, cols);
    return reshaped;
}

EigenImage applyBayerMask(const EigenImage& img, const std::string& bayer_pattern, int channel) {
    EigenImage mask(img.rows(), img.cols());
    mask.data().setZero();
    
    // Create mask based on Bayer pattern
    for (int i = 0; i < img.rows(); i += 2) {
        for (int j = 0; j < img.cols(); j += 2) {
            // Set mask values based on Bayer pattern and channel
            // This is a simplified implementation - would need to be expanded for all patterns
            if (bayer_pattern == "rggb") {
                if (channel == 0) { // R
                    mask.data()(i, j) = 1.0f;
                } else if (channel == 1) { // G
                    mask.data()(i, j+1) = 1.0f;
                    mask.data()(i+1, j) = 1.0f;
                } else if (channel == 2) { // B
                    mask.data()(i+1, j+1) = 1.0f;
                }
            }
        }
    }
    
    return mask;
}

std::vector<EigenImage> createBayerMasks(int rows, int cols, const std::string& bayer_pattern) {
    std::vector<EigenImage> masks(4, EigenImage::Zero(rows, cols));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int channel = 0;
            if (bayer_pattern == "RGGB") {
                channel = (i % 2 == 0) ? (j % 2 == 0 ? 0 : 1) : (j % 2 == 0 ? 2 : 3);
            } else if (bayer_pattern == "GRBG") {
                channel = (i % 2 == 0) ? (j % 2 == 0 ? 1 : 0) : (j % 2 == 0 ? 3 : 2);
            } else if (bayer_pattern == "GBRG") {
                channel = (i % 2 == 0) ? (j % 2 == 0 ? 2 : 3) : (j % 2 == 0 ? 0 : 1);
            } else if (bayer_pattern == "BGGR") {
                channel = (i % 2 == 0) ? (j % 2 == 0 ? 3 : 2) : (j % 2 == 0 ? 1 : 0);
            }
            masks[channel].data()(i, j) = 1.0f;
        }
    }
    
    return masks;
}

} // namespace eigen_utils

// Global utility functions implementation
EigenImage opencv_to_eigen(const cv::Mat& mat) {
    return EigenImage::fromOpenCV(mat);
}

cv::Mat eigen_to_opencv(const EigenImage& eigen_img) {
    return eigen_img.toOpenCV();
}

// EigenImageU32 implementation
EigenImageU32 EigenImageU32::fromOpenCV(const cv::Mat& mat) {
    if (mat.channels() != 1) {
        throw std::runtime_error("EigenImageU32::fromOpenCV: Input must be single-channel");
    }
    
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> eigen_mat(mat.rows, mat.cols);
    
    // Handle different input types properly
    if (mat.type() == CV_16UC1) {
        // Handle 16-bit unsigned input correctly
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                eigen_mat(i, j) = static_cast<uint32_t>(mat.at<uint16_t>(i, j));
            }
        }
    } else if (mat.type() == CV_8UC1) {
        // Handle 8-bit unsigned input
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                eigen_mat(i, j) = static_cast<uint32_t>(mat.at<uint8_t>(i, j));
            }
        }
    } else if (mat.type() == CV_32SC1) {
        // Handle 32-bit signed input - clamp negative values to 0
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                int32_t val = mat.at<int32_t>(i, j);
                eigen_mat(i, j) = static_cast<uint32_t>(std::max(0, val));
            }
        }
    } else if (mat.type() == CV_32FC1) {
        // Handle 32-bit float input - clamp to valid range
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                float val = mat.at<float>(i, j);
                eigen_mat(i, j) = static_cast<uint32_t>(std::max(0.0f, std::min(4294967295.0f, val)));
            }
        }
    } else {
        // Fallback: convert to 32-bit unsigned
        cv::Mat uint_mat;
        mat.convertTo(uint_mat, CV_32S);
        for (int i = 0; i < uint_mat.rows; ++i) {
            for (int j = 0; j < uint_mat.cols; ++j) {
                int32_t val = uint_mat.at<int32_t>(i, j);
                eigen_mat(i, j) = static_cast<uint32_t>(std::max(0, val));
            }
        }
    }
    
    return EigenImageU32(eigen_mat);
}

cv::Mat EigenImageU32::toOpenCV(int opencv_type) const {
    cv::Mat mat(data_.rows(), data_.cols(), opencv_type);
    
    if (opencv_type == CV_32S) {
        // Convert to signed 32-bit - check for overflow
        uint32_t max_signed = 2147483647; // 2^31 - 1
        for (int i = 0; i < data_.rows(); ++i) {
            for (int j = 0; j < data_.cols(); ++j) {
                uint32_t val = data_(i, j);
                if (val > max_signed) {
                    std::cerr << "Warning: Value " << val << " exceeds signed 32-bit range, clamping to " << max_signed << std::endl;
                    val = max_signed;
                }
                mat.at<int32_t>(i, j) = static_cast<int32_t>(val);
            }
        }
    } else if (opencv_type == CV_16U) {
        // Convert to 16-bit unsigned - check for overflow
        for (int i = 0; i < data_.rows(); ++i) {
            for (int j = 0; j < data_.cols(); ++j) {
                uint32_t val = data_(i, j);
                if (val > 65535) {
                    std::cerr << "Warning: Value " << val << " exceeds 16-bit range, clamping to 65535" << std::endl;
                    val = 65535;
                }
                mat.at<uint16_t>(i, j) = static_cast<uint16_t>(val);
            }
        }
    } else if (opencv_type == CV_8U) {
        // Convert to 8-bit unsigned - check for overflow
        for (int i = 0; i < data_.rows(); ++i) {
            for (int j = 0; j < data_.cols(); ++j) {
                uint32_t val = data_(i, j);
                if (val > 255) {
                    std::cerr << "Warning: Value " << val << " exceeds 8-bit range, clamping to 255" << std::endl;
                    val = 255;
                }
                mat.at<uchar>(i, j) = static_cast<uchar>(val);
            }
        }
    } else if (opencv_type == CV_32F) {
        // Convert to 32-bit float
        for (int i = 0; i < data_.rows(); ++i) {
            for (int j = 0; j < data_.cols(); ++j) {
                mat.at<float>(i, j) = static_cast<float>(data_(i, j));
            }
        }
    } else {
        throw std::runtime_error("EigenImageU32::toOpenCV: Unsupported OpenCV type");
    }
    
    return mat;
}

EigenImageU32 EigenImageU32::clip(uint32_t min_val, uint32_t max_val) const {
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> clipped = data_.cwiseMax(min_val).cwiseMin(max_val);
    return EigenImageU32(clipped);
}

EigenImageU32 EigenImageU32::extractBayerChannel(const std::string& bayer_pattern, char channel) const {
    EigenImageU32 result(rows(), cols());
    result.data().setZero();
    
    if (bayer_pattern == "rggb") {
        if (channel == 'r') {
            for (int i = 0; i < rows(); i += 2) {
                for (int j = 0; j < cols(); j += 2) {
                    result.data()(i, j) = data_(i, j);
                }
            }
        } else if (channel == 'g') {
            for (int i = 0; i < rows(); i += 2) {
                for (int j = 1; j < cols(); j += 2) {
                    result.data()(i, j) = data_(i, j);
                }
            }
            for (int i = 1; i < rows(); i += 2) {
                for (int j = 0; j < cols(); j += 2) {
                    result.data()(i, j) = data_(i, j);
                }
            }
        } else if (channel == 'b') {
            for (int i = 1; i < rows(); i += 2) {
                for (int j = 1; j < cols(); j += 2) {
                    result.data()(i, j) = data_(i, j);
                }
            }
        }
    }
    // Add other Bayer patterns as needed
    
    return result;
}

} // namespace hdr_isp 
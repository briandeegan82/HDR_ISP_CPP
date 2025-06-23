#include "halide_utils.hpp"
#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>

namespace hdr_isp {

Halide::Buffer<uint32_t> eigenToHalide(const EigenImageU32& eigen_img) {
    int width = eigen_img.cols();
    int height = eigen_img.rows();
    
    // Create Halide buffer with the same dimensions
    Halide::Buffer<uint32_t> halide_buffer(width, height);
    
    // Copy data from Eigen to Halide
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            halide_buffer(x, y) = eigen_img.data()(y, x);
        }
    }
    
    return halide_buffer;
}

EigenImageU32 halideToEigen(const Halide::Buffer<uint32_t>& halide_buffer) {
    int width = halide_buffer.width();
    int height = halide_buffer.height();
    
    // Create Eigen matrix with the same dimensions
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix(height, width);
    
    // Copy data from Halide to Eigen
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            eigen_matrix(y, x) = halide_buffer(x, y);
        }
    }
    
    return EigenImageU32(eigen_matrix);
}

Halide::Buffer<float> eigen3CToHalide(const EigenImage3C& eigen_img) {
    int width = eigen_img.cols();
    int height = eigen_img.rows();
    
    // Create Halide buffer with 3 channels
    Halide::Buffer<float> halide_buffer(width, height, 3);
    
    // Copy data from Eigen to Halide
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            halide_buffer(x, y, 0) = eigen_img.r().data()(y, x);
            halide_buffer(x, y, 1) = eigen_img.g().data()(y, x);
            halide_buffer(x, y, 2) = eigen_img.b().data()(y, x);
        }
    }
    
    return halide_buffer;
}

EigenImage3C halideToEigen3C(const Halide::Buffer<float>& halide_buffer) {
    int width = halide_buffer.width();
    int height = halide_buffer.height();
    
    // Create Eigen matrices for each channel
    Eigen::MatrixXf r_matrix(height, width);
    Eigen::MatrixXf g_matrix(height, width);
    Eigen::MatrixXf b_matrix(height, width);
    
    // Copy data from Halide to Eigen
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            r_matrix(y, x) = halide_buffer(x, y, 0);
            g_matrix(y, x) = halide_buffer(x, y, 1);
            b_matrix(y, x) = halide_buffer(x, y, 2);
        }
    }
    
    return EigenImage3C(EigenImage(r_matrix), EigenImage(g_matrix), EigenImage(b_matrix));
}

Halide::Buffer<float> opencvToHalide(const cv::Mat& cv_mat) {
    int width = cv_mat.cols;
    int height = cv_mat.rows;
    int channels = cv_mat.channels();
    
    // Create Halide buffer
    Halide::Buffer<float> halide_buffer(width, height, channels);
    
    // Copy data from OpenCV to Halide
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (cv_mat.type() == CV_8UC1 || cv_mat.type() == CV_8UC3) {
                    halide_buffer(x, y, c) = static_cast<float>(cv_mat.at<uchar>(y, x * channels + c));
                } else if (cv_mat.type() == CV_16UC1 || cv_mat.type() == CV_16UC3) {
                    halide_buffer(x, y, c) = static_cast<float>(cv_mat.at<ushort>(y, x * channels + c));
                } else if (cv_mat.type() == CV_32FC1 || cv_mat.type() == CV_32FC3) {
                    halide_buffer(x, y, c) = cv_mat.at<float>(y, x * channels + c);
                } else {
                    halide_buffer(x, y, c) = static_cast<float>(cv_mat.at<uchar>(y, x * channels + c));
                }
            }
        }
    }
    
    return halide_buffer;
}

cv::Mat halideToOpencv(const Halide::Buffer<float>& halide_buffer) {
    int width = halide_buffer.width();
    int height = halide_buffer.height();
    int channels = halide_buffer.channels();
    
    // Create OpenCV matrix
    cv::Mat cv_mat(height, width, CV_32FC(channels));
    
    // Copy data from Halide to OpenCV
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                cv_mat.at<float>(y, x * channels + c) = halide_buffer(x, y, c);
            }
        }
    }
    
    return cv_mat;
}

Halide::Buffer<uint32_t> opencvToHalideU32(const cv::Mat& cv_mat) {
    int width = cv_mat.cols;
    int height = cv_mat.rows;
    
    // Create Halide buffer
    Halide::Buffer<uint32_t> halide_buffer(width, height);
    
    // Copy data from OpenCV to Halide
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (cv_mat.type() == CV_8UC1) {
                halide_buffer(x, y) = static_cast<uint32_t>(cv_mat.at<uchar>(y, x));
            } else if (cv_mat.type() == CV_16UC1) {
                halide_buffer(x, y) = static_cast<uint32_t>(cv_mat.at<ushort>(y, x));
            } else if (cv_mat.type() == CV_32SC1) {
                halide_buffer(x, y) = static_cast<uint32_t>(cv_mat.at<int>(y, x));
            } else if (cv_mat.type() == CV_32FC1) {
                halide_buffer(x, y) = static_cast<uint32_t>(cv_mat.at<float>(y, x));
            } else {
                halide_buffer(x, y) = static_cast<uint32_t>(cv_mat.at<uchar>(y, x));
            }
        }
    }
    
    return halide_buffer;
}

cv::Mat halideToOpencvU32(const Halide::Buffer<uint32_t>& halide_buffer) {
    int width = halide_buffer.width();
    int height = halide_buffer.height();
    
    // Create OpenCV matrix
    cv::Mat cv_mat(height, width, CV_32SC1);
    
    // Copy data from Halide to OpenCV
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cv_mat.at<int>(y, x) = static_cast<int>(halide_buffer(x, y));
        }
    }
    
    return cv_mat;
}

Halide::Buffer<uint32_t> createTestHalideBuffer(int width, int height, 
                                               const std::string& pattern,
                                               uint32_t value) {
    Halide::Buffer<uint32_t> buffer(width, height);
    
    if (pattern == "constant") {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                buffer(x, y) = value;
            }
        }
    } else if (pattern == "gradient") {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                buffer(x, y) = static_cast<uint32_t>((x + y) % 4096); // 12-bit range
            }
        }
    } else if (pattern == "random") {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, 4095); // 12-bit range
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                buffer(x, y) = dis(gen);
            }
        }
    } else {
        // Default to gradient
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                buffer(x, y) = static_cast<uint32_t>((x + y) % 4096);
            }
        }
    }
    
    return buffer;
}

bool compareHalideBuffers(const Halide::Buffer<uint32_t>& buffer1,
                         const Halide::Buffer<uint32_t>& buffer2,
                         uint32_t tolerance) {
    if (buffer1.width() != buffer2.width() || buffer1.height() != buffer2.height()) {
        return false;
    }
    
    for (int y = 0; y < buffer1.height(); ++y) {
        for (int x = 0; x < buffer1.width(); ++x) {
            uint32_t diff = (buffer1(x, y) > buffer2(x, y)) ? 
                           (buffer1(x, y) - buffer2(x, y)) : 
                           (buffer2(x, y) - buffer1(x, y));
            if (diff > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}

bool compareHalideBuffers(const Halide::Buffer<float>& buffer1,
                         const Halide::Buffer<float>& buffer2,
                         float tolerance) {
    if (buffer1.width() != buffer2.width() || buffer1.height() != buffer2.height() ||
        buffer1.channels() != buffer2.channels()) {
        return false;
    }
    
    for (int c = 0; c < buffer1.channels(); ++c) {
        for (int y = 0; y < buffer1.height(); ++y) {
            for (int x = 0; x < buffer1.width(); ++x) {
                float diff = std::abs(buffer1(x, y, c) - buffer2(x, y, c));
                if (diff > tolerance) {
                    return false;
                }
            }
        }
    }
    
    return true;
}

void printHalideBufferStats(const Halide::Buffer<uint32_t>& buffer, 
                           const std::string& name) {
    uint32_t min_val = std::numeric_limits<uint32_t>::max();
    uint32_t max_val = 0;
    uint64_t sum = 0;
    int total_pixels = buffer.width() * buffer.height();
    
    for (int y = 0; y < buffer.height(); ++y) {
        for (int x = 0; x < buffer.width(); ++x) {
            uint32_t val = buffer(x, y);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
        }
    }
    
    double mean = static_cast<double>(sum) / total_pixels;
    
    std::cout << name << " Statistics:" << std::endl;
    std::cout << "  Size: " << buffer.width() << "x" << buffer.height() << std::endl;
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Total pixels: " << total_pixels << std::endl;
}

void printHalideBufferStats(const Halide::Buffer<float>& buffer, 
                           const std::string& name) {
    float min_val = std::numeric_limits<float>::max();
    float max_val = -std::numeric_limits<float>::max();
    double sum = 0.0;
    int total_pixels = buffer.width() * buffer.height() * buffer.channels();
    
    for (int c = 0; c < buffer.channels(); ++c) {
        for (int y = 0; y < buffer.height(); ++y) {
            for (int x = 0; x < buffer.width(); ++x) {
                float val = buffer(x, y, c);
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
            }
        }
    }
    
    double mean = sum / total_pixels;
    
    std::cout << name << " Statistics:" << std::endl;
    std::cout << "  Size: " << buffer.width() << "x" << buffer.height() << "x" << buffer.channels() << std::endl;
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Total pixels: " << total_pixels << std::endl;
}

} // namespace hdr_isp 
#include "eigen_utils.hpp"
#include <iostream>
#include <regex>
#include <filesystem>

namespace eigen_utils {

hdr_isp::EigenImage introduce_defect(const hdr_isp::EigenImage& img, int total_defective_pixels, bool padding) {
    hdr_isp::EigenImage result = img;
    
    if (padding) {
        // Add padding (simplified version - just extend the matrix)
        int rows = img.rows();
        int cols = img.cols();
        hdr_isp::EigenImage padded(rows + 4, cols + 4);
        
        // Copy original data to center
        padded.data().block(2, 2, rows, cols) = img.data();
        
        // Reflect borders
        padded.data().block(0, 2, 2, cols) = img.data().block(1, 0, 2, cols).colwise().reverse();
        padded.data().block(rows + 2, 2, 2, cols) = img.data().block(rows - 2, 0, 2, cols).colwise().reverse();
        padded.data().block(2, 0, rows, 2) = img.data().block(0, 1, rows, 2).rowwise().reverse();
        padded.data().block(2, cols + 2, rows, 2) = img.data().block(0, cols - 2, rows, 2).rowwise().reverse();
        
        result = padded;
    }
    
    // Introduce defective pixels
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> row_dist(0, result.rows() - 1);
    std::uniform_int_distribution<> col_dist(0, result.cols() - 1);
    std::uniform_int_distribution<> defect_type(0, 1); // 0: stuck at 0, 1: stuck at max
    
    for (int i = 0; i < total_defective_pixels; ++i) {
        int row = row_dist(gen);
        int col = col_dist(gen);
        
        if (defect_type(gen) == 0) {
            result.data()(row, col) = 0.0f;
        } else {
            result.data()(row, col) = 255.0f; // Assuming 8-bit max
        }
    }
    
    return result;
}

hdr_isp::EigenImage gauss_kern_raw(int size, float std_dev, float stride) {
    hdr_isp::EigenImage kernel(size, size);
    
    float sum = 0.0f;
    int center = size / 2;
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float x = (i - center) * stride;
            float y = (j - center) * stride;
            float value = std::exp(-(x*x + y*y) / (2.0f * std_dev * std_dev));
            kernel.data()(i, j) = value;
            sum += value;
        }
    }
    
    // Normalize
    if (sum > 0) {
        kernel.data() /= sum;
    }
    
    return kernel;
}

hdr_isp::EigenImage crop(const hdr_isp::EigenImage& img, int rows_to_crop, int cols_to_crop) {
    hdr_isp::EigenImage result = img;
    
    if (rows_to_crop > 0) {
        int crop_start = rows_to_crop / 2;
        int crop_end = img.rows() - (rows_to_crop - crop_start);
        result = hdr_isp::EigenImage(result.data().block(crop_start, 0, crop_end - crop_start, img.cols()));
    }
    
    if (cols_to_crop > 0) {
        int crop_start = cols_to_crop / 2;
        int crop_end = result.cols() - (cols_to_crop - crop_start);
        result = hdr_isp::EigenImage(result.data().block(0, crop_start, result.rows(), crop_end - crop_start));
    }
    
    return result;
}

hdr_isp::EigenImage stride_convolve2d(const hdr_isp::EigenImage& matrix, const hdr_isp::EigenImage& kernel) {
    // Simple 2D convolution with stride
    int kernel_rows = kernel.rows();
    int kernel_cols = kernel.cols();
    int stride = kernel_rows; // Assuming stride equals kernel size
    
    int output_rows = (matrix.rows() - kernel_rows) / stride + 1;
    int output_cols = (matrix.cols() - kernel_cols) / stride + 1;
    
    hdr_isp::EigenImage result(output_rows, output_cols);
    
    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernel_rows; ++ki) {
                for (int kj = 0; kj < kernel_cols; ++kj) {
                    sum += matrix.data()(i * stride + ki, j * stride + kj) * kernel.data()(ki, kj);
                }
            }
            result.data()(i, j) = sum;
        }
    }
    
    return result;
}

hdr_isp::EigenImage3C reconstruct_yuv_from_422_custom(const hdr_isp::EigenImage& yuv_422_custom, int width, int height) {
    hdr_isp::EigenImage3C result(height, width);
    
    int half_width = width / 2;
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < half_width; ++j) {
            // Y0, U, Y1, V format
            int y0_idx = i * half_width + j;
            int u_idx = height * half_width + i * half_width + j;
            int y1_idx = height * half_width * 2 + i * half_width + j;
            int v_idx = height * half_width * 3 + i * half_width + j;
            
            result.r()(i, j * 2) = yuv_422_custom.data()(0, y0_idx);
            result.g()(i, j * 2) = yuv_422_custom.data()(0, u_idx);
            result.r()(i, j * 2 + 1) = yuv_422_custom.data()(0, y1_idx);
            result.b()(i, j * 2) = yuv_422_custom.data()(0, v_idx);
            
            // Interpolate U and V for odd columns
            if (j * 2 + 1 < width) {
                result.g()(i, j * 2 + 1) = result.g()(i, j * 2);
                result.b()(i, j * 2 + 1) = result.b()(i, j * 2);
            }
        }
    }
    
    return result;
}

hdr_isp::EigenImage3C reconstruct_yuv_from_444_custom(const hdr_isp::EigenImage& yuv_444_custom, int width, int height) {
    hdr_isp::EigenImage3C result(height, width);
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pixel_idx = i * width + j;
            result.r()(i, j) = yuv_444_custom.data()(0, pixel_idx);
            result.g()(i, j) = yuv_444_custom.data()(0, pixel_idx + height * width);
            result.b()(i, j) = yuv_444_custom.data()(0, pixel_idx + height * width * 2);
        }
    }
    
    return result;
}

hdr_isp::EigenImage3C get_image_from_yuv_format_conversion(const hdr_isp::EigenImage& yuv_img, int height, int width, const std::string& yuv_custom_format) {
    if (yuv_custom_format == "422") {
        return reconstruct_yuv_from_422_custom(yuv_img, width, height);
    } else if (yuv_custom_format == "444") {
        return reconstruct_yuv_from_444_custom(yuv_img, width, height);
    } else {
        throw std::runtime_error("Unsupported YUV format: " + yuv_custom_format);
    }
}

hdr_isp::EigenImage3C yuv_to_rgb(const hdr_isp::EigenImage3C& yuv_img, const std::string& conv_std) {
    // Simple YUV to RGB conversion
    hdr_isp::EigenImage3C result(yuv_img.rows(), yuv_img.cols());
    
    for (int i = 0; i < yuv_img.rows(); ++i) {
        for (int j = 0; j < yuv_img.cols(); ++j) {
            float y = yuv_img.r()(i, j);
            float u = yuv_img.g()(i, j);
            float v = yuv_img.b()(i, j);
            
            // YUV to RGB conversion (simplified)
            result.r()(i, j) = y + 1.402f * (v - 128.0f);
            result.g()(i, j) = y - 0.344f * (u - 128.0f) - 0.714f * (v - 128.0f);
            result.b()(i, j) = y + 1.772f * (u - 128.0f);
        }
    }
    
    return result;
}

std::vector<hdr_isp::EigenImage> masks_cfa_bayer(const hdr_isp::EigenImage& img, const std::string& bayer) {
    std::vector<hdr_isp::EigenImage> masks(4);
    int rows = img.rows();
    int cols = img.cols();
    
    for (int i = 0; i < 4; ++i) {
        masks[i] = hdr_isp::EigenImage::Zero(rows, cols);
    }
    
    // Create Bayer pattern masks
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int pattern_idx = 0;
            if (bayer == "RGGB") {
                pattern_idx = ((i % 2) << 1) | (j % 2);
            } else if (bayer == "GRBG") {
                pattern_idx = ((i % 2) << 1) | ((j + 1) % 2);
            } else if (bayer == "GBRG") {
                pattern_idx = (((i + 1) % 2) << 1) | (j % 2);
            } else if (bayer == "BGGR") {
                pattern_idx = (((i + 1) % 2) << 1) | ((j + 1) % 2);
            }
            masks[pattern_idx].data()(i, j) = 1.0f;
        }
    }
    
    return masks;
}

hdr_isp::EigenImage apply_cfa(const hdr_isp::EigenImage& img, int bit_depth, const std::string& bayer) {
    std::vector<hdr_isp::EigenImage> masks = masks_cfa_bayer(img, bayer);
    hdr_isp::EigenImage result = hdr_isp::EigenImage::Zero(img.rows(), img.cols());
    
    for (int i = 0; i < 4; ++i) {
        result.data() += img.data().cwiseProduct(masks[i].data());
    }
    
    return result;
}

std::vector<std::string> parse_file_name(const std::string& filename) {
    std::vector<std::string> parts;
    std::regex pattern("([^_]+)_([^_]+)_([^_]+)_([^_]+)");
    std::smatch matches;
    
    if (std::regex_match(filename, matches, pattern)) {
        for (size_t i = 1; i < matches.size(); ++i) {
            parts.push_back(matches[i].str());
        }
    }
    
    return parts;
}

std::vector<std::string> extract_raw_metadata(const std::string& filename) {
    return parse_file_name(filename);
}

void display_ae_statistics(float ae_feedback, const std::vector<float>& awb_gains) {
    std::cout << "AE Feedback: " << ae_feedback << std::endl;
    std::cout << "AWB Gains: R=" << awb_gains[0] << ", G=" << awb_gains[1] << ", B=" << awb_gains[2] << std::endl;
}

hdr_isp::EigenImage normalize_matrix(const hdr_isp::EigenImage& img, float min_val, float max_val) {
    hdr_isp::EigenImage result = img;
    float current_min = img.data().minCoeff();
    float current_max = img.data().maxCoeff();
    
    if (current_max > current_min) {
        result.data() = (img.data() - current_min) / (current_max - current_min) * (max_val - min_val) + min_val;
    }
    
    return result;
}

hdr_isp::EigenImage3C normalize_matrix_3c(const hdr_isp::EigenImage3C& img, float min_val, float max_val) {
    hdr_isp::EigenImage3C result(img.rows(), img.cols());
    
    result.r() = normalize_matrix(img.r(), min_val, max_val);
    result.g() = normalize_matrix(img.g(), min_val, max_val);
    result.b() = normalize_matrix(img.b(), min_val, max_val);
    
    return result;
}

hdr_isp::EigenImage apply_filter_2d(const hdr_isp::EigenImage& img, const hdr_isp::EigenImage& kernel) {
    // Simple 2D convolution
    int kernel_rows = kernel.rows();
    int kernel_cols = kernel.cols();
    int pad_rows = kernel_rows / 2;
    int pad_cols = kernel_cols / 2;
    
    hdr_isp::EigenImage result = hdr_isp::EigenImage::Zero(img.rows(), img.cols());
    
    for (int i = 0; i < img.rows(); ++i) {
        for (int j = 0; j < img.cols(); ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernel_rows; ++ki) {
                for (int kj = 0; kj < kernel_cols; ++kj) {
                    int img_i = i + ki - pad_rows;
                    int img_j = j + kj - pad_cols;
                    
                    if (img_i >= 0 && img_i < img.rows() && img_j >= 0 && img_j < img.cols()) {
                        sum += img.data()(img_i, img_j) * kernel.data()(ki, kj);
                    }
                }
            }
            result.data()(i, j) = sum;
        }
    }
    
    return result;
}

hdr_isp::EigenImage apply_gaussian_filter(const hdr_isp::EigenImage& img, float sigma, int kernel_size) {
    hdr_isp::EigenImage kernel = gauss_kern_raw(kernel_size, sigma, 1.0f);
    return apply_filter_2d(img, kernel);
}

hdr_isp::EigenImage3C rgb_to_yuv(const hdr_isp::EigenImage3C& rgb, const std::string& standard) {
    hdr_isp::EigenImage3C result(rgb.rows(), rgb.cols());
    
    // BT709 conversion matrix
    Eigen::Matrix3f conversion_matrix;
    if (standard == "BT709") {
        conversion_matrix << 0.2126f, 0.7152f, 0.0722f,
                           -0.1146f, -0.3854f, 0.5000f,
                            0.5000f, -0.4542f, -0.0458f;
    } else {
        // BT601
        conversion_matrix << 0.299f, 0.587f, 0.114f,
                           -0.169f, -0.331f, 0.500f,
                            0.500f, -0.419f, -0.081f;
    }
    
    for (int i = 0; i < rgb.rows(); ++i) {
        for (int j = 0; j < rgb.cols(); ++j) {
            Eigen::Vector3f rgb_pixel(rgb.r()(i, j), rgb.g()(i, j), rgb.b()(i, j));
            Eigen::Vector3f yuv_pixel = conversion_matrix * rgb_pixel;
            
            result.r()(i, j) = yuv_pixel(0);
            result.g()(i, j) = yuv_pixel(1) + 128.0f;
            result.b()(i, j) = yuv_pixel(2) + 128.0f;
        }
    }
    
    return result;
}

hdr_isp::EigenImage3C yuv_to_rgb(const hdr_isp::EigenImage3C& yuv, const std::string& standard) {
    hdr_isp::EigenImage3C result(yuv.rows(), yuv.cols());
    
    // BT709 inverse conversion matrix
    Eigen::Matrix3f conversion_matrix;
    if (standard == "BT709") {
        conversion_matrix << 1.0f, 0.0f, 1.5748f,
                           1.0f, -0.1873f, -0.4681f,
                           1.0f, 1.8556f, 0.0f;
    } else {
        // BT601
        conversion_matrix << 1.0f, 0.0f, 1.402f,
                           1.0f, -0.344f, -0.714f,
                           1.0f, 1.772f, 0.0f;
    }
    
    for (int i = 0; i < yuv.rows(); ++i) {
        for (int j = 0; j < yuv.cols(); ++j) {
            Eigen::Vector3f yuv_pixel(yuv.r()(i, j), yuv.g()(i, j) - 128.0f, yuv.b()(i, j) - 128.0f);
            Eigen::Vector3f rgb_pixel = conversion_matrix * yuv_pixel;
            
            result.r()(i, j) = std::max(0.0f, std::min(255.0f, rgb_pixel(0)));
            result.g()(i, j) = std::max(0.0f, std::min(255.0f, rgb_pixel(1)));
            result.b()(i, j) = std::max(0.0f, std::min(255.0f, rgb_pixel(2)));
        }
    }
    
    return result;
}

} // namespace eigen_utils 
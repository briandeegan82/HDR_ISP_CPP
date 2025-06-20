#include "black_level_correction.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>

BlackLevelCorrection::BlackLevelCorrection(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_blc)
    : raw_(img)
    , sensor_info_(sensor_info)
    , parm_blc_(parm_blc)
    , enable_(parm_blc["is_enable"].as<bool>())
    , is_linearize_(parm_blc["is_linear"].as<bool>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , bayer_pattern_(sensor_info["bayer_pattern"].as<std::string>())
    , is_save_(parm_blc["is_save"].as<bool>())
    , use_eigen_(true) // Use Eigen by default
{
}

cv::Mat BlackLevelCorrection::execute() {
    if (!enable_) {
        return raw_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result;
    if (use_eigen_) {
        // Use EigenImage32 for integer processing
        hdr_isp::EigenImage32 eigen_img = hdr_isp::EigenImage32::fromOpenCV(raw_);
        hdr_isp::EigenImage32 corrected = apply_blc_parameters_eigen(eigen_img);
        result = corrected.toOpenCV(raw_.type());
    } else {
        result = apply_blc_parameters_opencv();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Black Level Correction execution time: " << duration.count() << " seconds" << std::endl;

    return result;
}

cv::Mat BlackLevelCorrection::apply_blc_parameters_opencv() {
    // Get parameters from config
    double r_offset = parm_blc_["r_offset"].as<double>();
    double gb_offset = parm_blc_["gb_offset"].as<double>();
    double gr_offset = parm_blc_["gr_offset"].as<double>();
    double b_offset = parm_blc_["b_offset"].as<double>();

    double r_sat = parm_blc_["r_sat"].as<double>();
    double gr_sat = parm_blc_["gr_sat"].as<double>();
    double gb_sat = parm_blc_["gb_sat"].as<double>();
    double b_sat = parm_blc_["b_sat"].as<double>();

    std::cout << "Black Level Correction Parameters:" << std::endl;
    std::cout << "  R offset: " << r_offset << ", saturation: " << r_sat << std::endl;
    std::cout << "  GR offset: " << gr_offset << ", saturation: " << gr_sat << std::endl;
    std::cout << "  GB offset: " << gb_offset << ", saturation: " << gb_sat << std::endl;
    std::cout << "  B offset: " << b_offset << ", saturation: " << b_sat << std::endl;
    std::cout << "  Bit depth: " << bit_depth_ << std::endl;
    std::cout << "  Bayer pattern: " << bayer_pattern_ << std::endl;
    std::cout << "  Linearize: " << (is_linearize_ ? "true" : "false") << std::endl;

    // Check if we can use optimized integer implementation
    bool use_optimized = true;
    
    // Verify all parameters are integers
    if (r_offset != static_cast<int>(r_offset) || 
        gb_offset != static_cast<int>(gb_offset) ||
        gr_offset != static_cast<int>(gr_offset) ||
        b_offset != static_cast<int>(b_offset) ||
        r_sat != static_cast<int>(r_sat) ||
        gr_sat != static_cast<int>(gr_sat) ||
        gb_sat != static_cast<int>(gb_sat) ||
        b_sat != static_cast<int>(b_sat)) {
        use_optimized = false;
    }
    
    // Check if offsets are small enough for integer arithmetic
    if (r_offset > 32767 || gb_offset > 32767 || gr_offset > 32767 || b_offset > 32767) {
        use_optimized = false;
    }

    if (use_optimized && !is_linearize_) {
        // Optimized integer-only implementation for offset-only correction
        return apply_blc_parameters_opencv_optimized();
    } else {
        // Original float-based implementation for complex cases
        return apply_blc_parameters_opencv_float();
    }
}

cv::Mat BlackLevelCorrection::apply_blc_parameters_opencv_optimized() {
    // Get parameters as integers
    int r_offset = static_cast<int>(parm_blc_["r_offset"].as<double>());
    int gb_offset = static_cast<int>(parm_blc_["gb_offset"].as<double>());
    int gr_offset = static_cast<int>(parm_blc_["gr_offset"].as<double>());
    int b_offset = static_cast<int>(parm_blc_["b_offset"].as<double>());

    std::cout << "Using optimized integer implementation" << std::endl;

    // Work directly on input data type (8-bit, 16-bit, or 32-bit)
    cv::Mat result = raw_.clone();
    
    // Determine input type and handle accordingly
    int input_type = raw_.type();
    
    if (input_type == CV_8UC1) {
        // 8-bit processing
        if (bayer_pattern_ == "rggb") {
            for (int i = 0; i < result.rows; i += 2) {
                for (int j = 0; j < result.cols; j += 2) {
                    int pixel = result.at<uint8_t>(i, j);
                    pixel = std::max(0, pixel - r_offset);
                    result.at<uint8_t>(i, j) = static_cast<uint8_t>(pixel);
                }
            }
            // Add other channels...
        }
    }
    else if (input_type == CV_16UC1) {
        // 16-bit processing
        if (bayer_pattern_ == "rggb") {
            for (int i = 0; i < result.rows; i += 2) {
                for (int j = 0; j < result.cols; j += 2) {
                    int pixel = result.at<uint16_t>(i, j);
                    pixel = std::max(0, pixel - r_offset);
                    result.at<uint16_t>(i, j) = static_cast<uint16_t>(pixel);
                }
            }
            // GR channel
            for (int i = 0; i < result.rows; i += 2) {
                for (int j = 1; j < result.cols; j += 2) {
                    int pixel = result.at<uint16_t>(i, j);
                    pixel = std::max(0, pixel - gr_offset);
                    result.at<uint16_t>(i, j) = static_cast<uint16_t>(pixel);
                }
            }
            // GB channel
            for (int i = 1; i < result.rows; i += 2) {
                for (int j = 0; j < result.cols; j += 2) {
                    int pixel = result.at<uint16_t>(i, j);
                    pixel = std::max(0, pixel - gb_offset);
                    result.at<uint16_t>(i, j) = static_cast<uint16_t>(pixel);
                }
            }
            // B channel
            for (int i = 1; i < result.rows; i += 2) {
                for (int j = 1; j < result.cols; j += 2) {
                    int pixel = result.at<uint16_t>(i, j);
                    pixel = std::max(0, pixel - b_offset);
                    result.at<uint16_t>(i, j) = static_cast<uint16_t>(pixel);
                }
            }
        }
    }
    else if (input_type == CV_32SC1) {
        // 32-bit processing for HDR images
        if (bayer_pattern_ == "rggb") {
            for (int i = 0; i < result.rows; i += 2) {
                for (int j = 0; j < result.cols; j += 2) {
                    int pixel = result.at<int32_t>(i, j);
                    pixel = std::max(0, pixel - r_offset);
                    result.at<int32_t>(i, j) = pixel;
                }
            }
            // GR channel
            for (int i = 0; i < result.rows; i += 2) {
                for (int j = 1; j < result.cols; j += 2) {
                    int pixel = result.at<int32_t>(i, j);
                    pixel = std::max(0, pixel - gr_offset);
                    result.at<int32_t>(i, j) = pixel;
                }
            }
            // GB channel
            for (int i = 1; i < result.rows; i += 2) {
                for (int j = 0; j < result.cols; j += 2) {
                    int pixel = result.at<int32_t>(i, j);
                    pixel = std::max(0, pixel - gb_offset);
                    result.at<int32_t>(i, j) = pixel;
                }
            }
            // B channel
            for (int i = 1; i < result.rows; i += 2) {
                for (int j = 1; j < result.cols; j += 2) {
                    int pixel = result.at<int32_t>(i, j);
                    pixel = std::max(0, pixel - b_offset);
                    result.at<int32_t>(i, j) = pixel;
                }
            }
        }
    }
    else {
        // Fallback to float implementation for unsupported types
        std::cout << "Unsupported input type, falling back to float implementation" << std::endl;
        return apply_blc_parameters_opencv_float();
    }

    return result;
}

cv::Mat BlackLevelCorrection::apply_blc_parameters_opencv_float() {
    // Get parameters from config
    double r_offset = parm_blc_["r_offset"].as<double>();
    double gb_offset = parm_blc_["gb_offset"].as<double>();
    double gr_offset = parm_blc_["gr_offset"].as<double>();
    double b_offset = parm_blc_["b_offset"].as<double>();

    double r_sat = parm_blc_["r_sat"].as<double>();
    double gr_sat = parm_blc_["gr_sat"].as<double>();
    double gb_sat = parm_blc_["gb_sat"].as<double>();
    double b_sat = parm_blc_["b_sat"].as<double>();

    std::cout << "Using float implementation" << std::endl;

    // Convert to float32 for processing
    cv::Mat raw;
    raw_.convertTo(raw, CV_32F);

    // Apply black level correction based on bayer pattern
    if (bayer_pattern_ == "rggb") {
        // R channel
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= r_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (r_sat - r_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // GR channel
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 1; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= gr_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (gr_sat - gr_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // GB channel
        for (int i = 1; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= gb_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (gb_sat - gb_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // B channel
        for (int i = 1; i < raw.rows; i += 2) {
            for (int j = 1; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= b_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (b_sat - b_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
    }
    else if (bayer_pattern_ == "bggr") {
        // B channel
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= b_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (b_sat - b_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // GB channel
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 1; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= gb_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (gb_sat - gb_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // GR channel
        for (int i = 1; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= gr_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (gr_sat - gr_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // R channel
        for (int i = 1; i < raw.rows; i += 2) {
            for (int j = 1; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= r_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (r_sat - r_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
    }
    else if (bayer_pattern_ == "grbg") {
        // GR channel
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= gr_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (gr_sat - gr_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // R channel
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 1; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= r_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (r_sat - r_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // B channel
        for (int i = 1; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= b_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (b_sat - b_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // GB channel
        for (int i = 1; i < raw.rows; i += 2) {
            for (int j = 1; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= gb_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (gb_sat - gb_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
    }
    else if (bayer_pattern_ == "gbrg") {
        // GB channel
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= gb_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (gb_sat - gb_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // B channel
        for (int i = 0; i < raw.rows; i += 2) {
            for (int j = 1; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= b_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (b_sat - b_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // R channel
        for (int i = 1; i < raw.rows; i += 2) {
            for (int j = 0; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= r_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (r_sat - r_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
        // GR channel
        for (int i = 1; i < raw.rows; i += 2) {
            for (int j = 1; j < raw.cols; j += 2) {
                raw.at<float>(i, j) -= gr_offset;
                if (is_linearize_) {
                    raw.at<float>(i, j) = raw.at<float>(i, j) / (gr_sat - gr_offset) * ((1 << bit_depth_) - 1);
                }
            }
        }
    }

    // Convert back to 16-bit
    cv::Mat result;
    raw.convertTo(result, CV_16U);
    return result;
}

hdr_isp::EigenImage32 BlackLevelCorrection::apply_blc_parameters_eigen(const hdr_isp::EigenImage32& img) {
    int r_offset = static_cast<int>(parm_blc_["r_offset"].as<double>());
    int gb_offset = static_cast<int>(parm_blc_["gb_offset"].as<double>());
    int gr_offset = static_cast<int>(parm_blc_["gr_offset"].as<double>());
    int b_offset = static_cast<int>(parm_blc_["b_offset"].as<double>());
    int r_sat = static_cast<int>(parm_blc_["r_sat"].as<double>());
    int gr_sat = static_cast<int>(parm_blc_["gr_sat"].as<double>());
    int gb_sat = static_cast<int>(parm_blc_["gb_sat"].as<double>());
    int b_sat = static_cast<int>(parm_blc_["b_sat"].as<double>());
    
    hdr_isp::EigenImage32 result = img;
    apply_blc_bayer_eigen(result, r_offset, gr_offset, gb_offset, b_offset, r_sat, gr_sat, gb_sat, b_sat);
    return result;
}

void BlackLevelCorrection::apply_blc_bayer_eigen(hdr_isp::EigenImage32& img, int r_offset, int gr_offset, int gb_offset, int b_offset, int r_sat, int gr_sat, int gb_sat, int b_sat) {
    int rows = img.rows();
    int cols = img.cols();
    int max_val = (1 << bit_depth_) - 1;
    
    if (bayer_pattern_ == "rggb") {
        // R channel
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                img.data()(i, j) -= r_offset;
                if (is_linearize_) {
                    img.data()(i, j) = img.data()(i, j) / (r_sat - r_offset) * max_val;
                }
            }
        }
        // GR channel
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                img.data()(i, j) -= gr_offset;
                if (is_linearize_) {
                    img.data()(i, j) = img.data()(i, j) / (gr_sat - gr_offset) * max_val;
                }
            }
        }
        // GB channel
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                img.data()(i, j) -= gb_offset;
                if (is_linearize_) {
                    img.data()(i, j) = img.data()(i, j) / (gb_sat - gb_offset) * max_val;
                }
            }
        }
        // B channel
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                img.data()(i, j) -= b_offset;
                if (is_linearize_) {
                    img.data()(i, j) = img.data()(i, j) / (b_sat - b_offset) * max_val;
                }
            }
        }
    }
    // Add other Bayer patterns as needed
    
    // Clip values to valid range
    img = img.clip(0, max_val);
} 
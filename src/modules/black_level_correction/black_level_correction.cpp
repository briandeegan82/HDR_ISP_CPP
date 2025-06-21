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
        hdr_isp::EigenImage eigen_img = hdr_isp::EigenImage::fromOpenCV(raw_);
        hdr_isp::EigenImage corrected = apply_blc_parameters_eigen(eigen_img);
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

hdr_isp::EigenImage BlackLevelCorrection::apply_blc_parameters_eigen(const hdr_isp::EigenImage& img) {
    double r_offset = parm_blc_["r_offset"].as<double>();
    double gb_offset = parm_blc_["gb_offset"].as<double>();
    double gr_offset = parm_blc_["gr_offset"].as<double>();
    double b_offset = parm_blc_["b_offset"].as<double>();
    double r_sat = parm_blc_["r_sat"].as<double>();
    double gr_sat = parm_blc_["gr_sat"].as<double>();
    double gb_sat = parm_blc_["gb_sat"].as<double>();
    double b_sat = parm_blc_["b_sat"].as<double>();
    hdr_isp::EigenImage result = img;
    apply_blc_bayer_eigen(result, r_offset, gr_offset, gb_offset, b_offset, r_sat, gr_sat, gb_sat, b_sat);
    return result;
}

void BlackLevelCorrection::apply_blc_bayer_eigen(hdr_isp::EigenImage& img, double r_offset, double gr_offset, double gb_offset, double b_offset, double r_sat, double gr_sat, double gb_sat, double b_sat) {
    int rows = img.rows();
    int cols = img.cols();
    int max_val = (1 << bit_depth_) - 1;
    if (bayer_pattern_ == "rggb") {
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                img.data()(i, j) -= r_offset;
                if (is_linearize_) img.data()(i, j) = img.data()(i, j) / (r_sat - r_offset) * max_val;
            }
        }
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                img.data()(i, j) -= gr_offset;
                if (is_linearize_) img.data()(i, j) = img.data()(i, j) / (gr_sat - gr_offset) * max_val;
            }
        }
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                img.data()(i, j) -= gb_offset;
                if (is_linearize_) img.data()(i, j) = img.data()(i, j) / (gb_sat - gb_offset) * max_val;
            }
        }
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                img.data()(i, j) -= b_offset;
                if (is_linearize_) img.data()(i, j) = img.data()(i, j) / (b_sat - b_offset) * max_val;
            }
        }
    }
    // (other Bayer patterns can be added similarly)
} 
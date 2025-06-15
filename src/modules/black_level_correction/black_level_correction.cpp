#include "black_level_correction.hpp"
#include <chrono>
#include <iostream>

BlackLevelCorrection::BlackLevelCorrection(const cv::Mat& img, const YAML::Node& sensor_info, const YAML::Node& parm_blc)
    : raw_(img)
    , sensor_info_(sensor_info)
    , parm_blc_(parm_blc)
    , enable_(parm_blc["is_enable"].as<bool>())
    , is_linearize_(parm_blc["is_linear"].as<bool>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , bayer_pattern_(sensor_info["bayer_pattern"].as<std::string>())
    , is_save_(parm_blc["is_save"].as<bool>())
{
}

cv::Mat BlackLevelCorrection::execute() {
    if (!enable_) {
        return raw_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result = apply_blc_parameters();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Black Level Correction execution time: " << duration.count() << " seconds" << std::endl;

    return result;
}

cv::Mat BlackLevelCorrection::apply_blc_parameters() {
    // Get parameters from config
    double r_offset = parm_blc_["r_offset"].as<double>();
    double gb_offset = parm_blc_["gb_offset"].as<double>();
    double gr_offset = parm_blc_["gr_offset"].as<double>();
    double b_offset = parm_blc_["b_offset"].as<double>();

    double r_sat = parm_blc_["r_sat"].as<double>();
    double gr_sat = parm_blc_["gr_sat"].as<double>();
    double gb_sat = parm_blc_["gb_sat"].as<double>();
    double b_sat = parm_blc_["b_sat"].as<double>();

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

    // Clip values and convert back to original type
    cv::Mat result;
    cv::threshold(raw, raw, 0, (1 << bit_depth_) - 1, cv::THRESH_TRUNC);
    raw.convertTo(result, raw_.type());
    return result;
} 
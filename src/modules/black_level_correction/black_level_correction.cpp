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
    std::cerr << "Debug - Entering BlackLevelCorrection::execute(), is_save_ = " << (is_save_ ? "true" : "false") << std::endl;
    if (!enable_) {
        std::cerr << "Debug - Black level correction is disabled" << std::endl;
        return raw_;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result;
    if (use_eigen_) {
        if (is_save_) std::cerr << "Debug - Using Eigen implementation for BLC" << std::endl;
        hdr_isp::EigenImage eigen_img = hdr_isp::EigenImage::fromOpenCV(raw_);
        hdr_isp::EigenImage corrected = apply_blc_parameters_eigen(eigen_img);
        result = corrected.toOpenCV(raw_.type());
    } else {
        if (is_save_) std::cerr << "Debug - Using OpenCV implementation for BLC" << std::endl;
        result = apply_blc_parameters_opencv();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cerr << "Black Level Correction execution time: " << duration.count() << " seconds" << std::endl;

    if (is_save_) {
        std::cerr << "Debug - Attempting to save intermediate image" << std::endl;
        try {
            std::filesystem::create_directories("out_frames/intermediate");
            std::string output_path = "out_frames/intermediate/Out_black_level_correction_" + 
                                    std::to_string(result.cols) + "x" + std::to_string(result.rows) + ".png";
            
            std::cerr << "Debug - Output path: " << output_path << std::endl;
            
            // Convert to 8-bit for saving
            cv::Mat save_img;
            std::cerr << "Debug - Before conversion, result type: " << result.type() << ", depth: " << result.depth() << std::endl;
            result.convertTo(save_img, CV_8U, 255.0 / ((1 << bit_depth_) - 1));
            std::cerr << "Debug - After conversion, save_img type: " << save_img.type() << ", depth: " << save_img.depth() << std::endl;
            
            if (save_img.empty()) {
                std::cerr << "Error: save_img is empty after conversion!" << std::endl;
                return result;
            }
            
            // Debug prints for image statistics
            double min_val, max_val;
            cv::minMaxLoc(save_img, &min_val, &max_val);
            cv::Scalar mean_val = cv::mean(save_img);
            std::cerr << "Debug - save_img statistics:" << std::endl;
            std::cerr << "  Mean: " << mean_val[0] << std::endl;
            std::cerr << "  Min: " << min_val << std::endl;
            std::cerr << "  Max: " << max_val << std::endl;
            std::cerr << "  Image size: " << save_img.size() << std::endl;
            std::cerr << "  Number of channels: " << save_img.channels() << std::endl;
            
            bool write_success = cv::imwrite(output_path, save_img);
            if (!write_success) {
                std::cerr << "Error: Failed to write image to: " << output_path << std::endl;
            } else {
                std::cerr << "Successfully wrote image to: " << output_path << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error saving image: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Debug - Skipping save as is_save_ is false" << std::endl;
    }

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
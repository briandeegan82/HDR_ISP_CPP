#include "utils.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>

namespace utils {

cv::Mat introduce_defect(const cv::Mat& img, int total_defective_pixels, bool padding) {
    cv::Mat padded_img;
    if (padding) {
        cv::copyMakeBorder(img, padded_img, 2, 2, 2, 2, cv::BORDER_REFLECT);
    } else {
        padded_img = img.clone();
    }

    cv::Mat orig_val = cv::Mat::zeros(padded_img.size(), CV_32F);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> defect_low(1, 15);
    std::uniform_int_distribution<> defect_high(4081, 4095);
    std::uniform_int_distribution<> row_dist(2, img.rows - 3);
    std::uniform_int_distribution<> col_dist(2, img.cols - 3);

    while (total_defective_pixels > 0) {
        int defect_val = (gen() % 2 == 0) ? defect_low(gen) : defect_high(gen);
        int random_row = row_dist(gen);
        int random_col = col_dist(gen);

        float left = orig_val.at<float>(random_row, random_col - 2);
        float right = orig_val.at<float>(random_row, random_col + 2);
        float top = orig_val.at<float>(random_row - 2, random_col);
        float bottom = orig_val.at<float>(random_row + 2, random_col);

        if (left == 0 && right == 0 && top == 0 && bottom == 0 && 
            orig_val.at<float>(random_row, random_col) == 0) {
            orig_val.at<float>(random_row, random_col) = padded_img.at<float>(random_row, random_col);
            padded_img.at<float>(random_row, random_col) = defect_val;
            total_defective_pixels--;
        }
    }

    return padded_img;
}

cv::Mat gauss_kern_raw(int size, float std_dev, float stride) {
    if (size % 2 == 0) {
        std::cerr << "Warning: kernel size cannot be even, setting it as odd value" << std::endl;
        size++;
    }

    if (size <= 0) {
        std::cerr << "Warning: kernel size cannot be <= zero, setting it as 3" << std::endl;
        size = 3;
    }

    cv::Mat out_kern = cv::Mat::zeros(size, size, CV_32F);
    float center = (size - 1) / 2.0f;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float x = stride * (i - center);
            float y = stride * (j - center);
            out_kern.at<float>(i, j) = std::exp(-(x*x + y*y) / (2 * std_dev * std_dev));
        }
    }

    float sum = cv::sum(out_kern)[0];
    out_kern /= sum;

    return out_kern;
}

cv::Mat crop(const cv::Mat& img, int rows_to_crop, int cols_to_crop) {
    cv::Mat result = img.clone();

    if (rows_to_crop) {
        if (rows_to_crop % 2 == 0) {
            result = result(cv::Rect(0, rows_to_crop/2, result.cols, result.rows - rows_to_crop));
        } else {
            result = result(cv::Rect(0, 0, result.cols, result.rows - 1));
        }
    }

    if (cols_to_crop) {
        if (cols_to_crop % 2 == 0) {
            result = result(cv::Rect(cols_to_crop/2, 0, result.cols - cols_to_crop, result.rows));
        } else {
            result = result(cv::Rect(0, 0, result.cols - 1, result.rows));
        }
    }

    return result;
}

cv::Mat stride_convolve2d(const cv::Mat& matrix, const cv::Mat& kernel) {
    cv::Mat result;
    cv::filter2D(matrix, result, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    
    // Apply stride
    cv::Mat strided_result;
    cv::resize(result, strided_result, cv::Size(), 1.0/kernel.rows, 1.0/kernel.cols, cv::INTER_NEAREST);
    
    return strided_result;
}

cv::Mat reconstruct_yuv_from_422_custom(const cv::Mat& yuv_422_custom, int width, int height) {
    cv::Mat yuv_img = cv::Mat::zeros(height, width, CV_8UC3);

    // Rearrange the flattened 4:2:2 YUV data back to 3D YUV format
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width/2; j++) {
            yuv_img.at<cv::Vec3b>(i, j*2)[0] = yuv_422_custom.at<uchar>(0, i*width/2 + j);
            yuv_img.at<cv::Vec3b>(i, j*2)[1] = yuv_422_custom.at<uchar>(0, i*width/2 + j + height*width/2);
            yuv_img.at<cv::Vec3b>(i, j*2+1)[0] = yuv_422_custom.at<uchar>(0, i*width/2 + j + height*width);
            yuv_img.at<cv::Vec3b>(i, j*2)[2] = yuv_422_custom.at<uchar>(0, i*width/2 + j + height*width*3/2);
        }
    }

    // Replicate the U and V (chroma) channels to the odd columns
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width/2; j++) {
            yuv_img.at<cv::Vec3b>(i, j*2+1)[1] = yuv_img.at<cv::Vec3b>(i, j*2)[1];
            yuv_img.at<cv::Vec3b>(i, j*2+1)[2] = yuv_img.at<cv::Vec3b>(i, j*2)[2];
        }
    }

    return yuv_img;
}

cv::Mat reconstruct_yuv_from_444_custom(const cv::Mat& yuv_444_custom, int width, int height) {
    cv::Mat yuv_img = cv::Mat::zeros(height, width, CV_8UC3);

    // Rearrange the flattened 4:4:4 YUV data back to 3D YUV format
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            yuv_img.at<cv::Vec3b>(i, j)[0] = yuv_444_custom.at<uchar>(0, i*width + j);
            yuv_img.at<cv::Vec3b>(i, j)[1] = yuv_444_custom.at<uchar>(0, i*width + j + height*width);
            yuv_img.at<cv::Vec3b>(i, j)[2] = yuv_444_custom.at<uchar>(0, i*width + j + height*width*2);
        }
    }

    return yuv_img;
}

cv::Mat get_image_from_yuv_format_conversion(const cv::Mat& yuv_img, int height, int width, 
                                           const std::string& yuv_custom_format) {
    if (yuv_custom_format == "422") {
        return reconstruct_yuv_from_422_custom(yuv_img, width, height);
    } else {
        return reconstruct_yuv_from_444_custom(yuv_img, width, height);
    }
}

cv::Mat yuv_to_rgb(const cv::Mat& yuv_img, const std::string& conv_std) {
    cv::Mat rgb_img;
    if (conv_std == "BT601") {
        cv::cvtColor(yuv_img, rgb_img, cv::COLOR_YUV2BGR);
    } else if (conv_std == "BT709") {
        cv::cvtColor(yuv_img, rgb_img, cv::COLOR_YUV2BGR_I420);
    }
    return rgb_img;
}

std::vector<cv::Mat> masks_cfa_bayer(const cv::Mat& img, const std::string& bayer) {
    std::vector<cv::Mat> masks(4);
    for (int i = 0; i < 4; i++) {
        masks[i] = cv::Mat::zeros(img.size(), CV_8U);
    }

    if (bayer == "rggb") {
        for (int i = 0; i < img.rows; i += 2) {
            for (int j = 0; j < img.cols; j += 2) {
                masks[0].at<uchar>(i, j) = 1;      // R
                masks[1].at<uchar>(i, j+1) = 1;    // G1
                masks[2].at<uchar>(i+1, j) = 1;    // G2
                masks[3].at<uchar>(i+1, j+1) = 1;  // B
            }
        }
    }
    // Add other Bayer patterns as needed...

    return masks;
}

cv::Mat apply_cfa(const cv::Mat& img, int bit_depth, const std::string& bayer) {
    std::vector<cv::Mat> masks = masks_cfa_bayer(img, bayer);
    cv::Mat result = cv::Mat::zeros(img.size(), CV_32F);
    
    for (int i = 0; i < 4; i++) {
        cv::Mat masked;
        cv::multiply(img, masks[i], masked);
        cv::add(result, masked, result);
    }

    return result;
}

void save_pipeline_output(const std::string& img_name, const cv::Mat& output_img, const YAML::Node& config_file) {
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "_%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    // Create output directory
    std::filesystem::create_directories(OUTPUT_DIR);

    // Save image
    std::string img_path = OUTPUT_DIR + img_name + timestamp + ".png";
    cv::imwrite(img_path, output_img);

    // Save config
    std::string config_path = OUTPUT_DIR + img_name + timestamp + ".yaml";
    std::ofstream config_file_out(config_path);
    YAML::Emitter emitter;
    emitter << config_file;
    config_file_out << emitter.c_str();
}

void save_output_array(const std::string& img_name, const cv::Mat& output_array, const std::string& module_name,
                      const YAML::Node& platform, int bitdepth, const std::string& bayer_pattern) {
    std::filesystem::create_directories(OUTPUT_ARRAY_DIR);
    std::string filename = OUTPUT_ARRAY_DIR + module_name + img_name;
    cv::imwrite(filename + ".png", output_array);
}

void save_output_array_yuv(const std::string& img_name, const cv::Mat& output_array, const std::string& module_name,
                          const YAML::Node& platform, const std::string& conv_std) {
    std::filesystem::create_directories(OUTPUT_ARRAY_DIR);
    std::string filename = OUTPUT_ARRAY_DIR + module_name + img_name;
    cv::imwrite(filename + ".yuv", output_array);
}

void save_image(const cv::Mat& img, const std::string& output_path) {
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
    cv::imwrite(output_path, img);
}

std::vector<std::string> parse_file_name(const std::string& filename) {
    std::regex pattern(R"((.+)_(\d+)x(\d+)_(\d+)(?:bit|bits)_(RGGB|GRBG|GBRG|BGGR))");
    std::smatch matches;
    
    if (std::regex_match(filename, matches, pattern)) {
        return {matches[2].str(), matches[3].str(), matches[4].str(), 
                std::string(matches[5].str())};
    }
    return {};
}

void display_ae_statistics(float ae_feedback, const std::vector<float>& awb_gains) {
    if (awb_gains.empty()) {
        std::cout << "   - 3A Stats    - AWB is Disable" << std::endl;
    } else {
        std::cout << "   - 3A Stats    - AWB Rgain = " << awb_gains[0] << std::endl;
        std::cout << "   - 3A Stats    - AWB Bgain = " << awb_gains[1] << std::endl;
    }

    if (ae_feedback < 0) {
        std::cout << "   - 3A Stats    - AE Feedback = Underexposed" << std::endl;
    } else if (ae_feedback > 0) {
        std::cout << "   - 3A Stats    - AE Feedback = Overexposed" << std::endl;
    } else {
        std::cout << "   - 3A Stats    - AE Feedback = Correct Exposure" << std::endl;
    }
}

} // namespace utils 
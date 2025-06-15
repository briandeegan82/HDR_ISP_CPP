#include "scale_2d.hpp"
#include <chrono>
#include <iostream>
#include <algorithm>

Scale2D::Scale2D(const cv::Mat& single_channel, const YAML::Node& sensor_info, const YAML::Node& parm_sca)
    : single_channel_(single_channel)
    , sensor_info_(sensor_info)
    , parm_sca_(parm_sca)
{
    get_scaling_params();
}

void Scale2D::get_scaling_params() {
    old_size_ = std::make_pair(sensor_info_["height"].as<int>(), sensor_info_["width"].as<int>());
    new_size_ = std::make_pair(parm_sca_["new_height"].as<int>(), parm_sca_["new_width"].as<int>());
    is_debug_ = parm_sca_["is_debug"].as<bool>();
    is_hardware_ = parm_sca_["is_hardware"].as<bool>();
    algo_ = parm_sca_["algorithm"].as<std::string>();
    upscale_method_ = parm_sca_["upscale_method"].as<std::string>();
    downscale_method_ = parm_sca_["downscale_method"].as<std::string>();
}

cv::Mat Scale2D::resize_by_non_int_fact(
    const std::pair<std::pair<int, int>, std::pair<int, int>>& red_fact,
    const std::pair<std::string, std::string>& method) {
    
    cv::Mat result = single_channel_.clone();

    for (int i = 0; i < 2; ++i) {
        if (red_fact.first.first > 0 || red_fact.second.first > 0) {
            int upscale_fact = (i == 0) ? red_fact.first.first : red_fact.second.first;
            int downscale_fact = (i == 0) ? red_fact.first.second : red_fact.second.second;

            cv::Size upscale_size;
            if (i == 0) {
                upscale_size = cv::Size(result.cols, upscale_fact * result.rows);
            } else {
                upscale_size = cv::Size(upscale_fact * result.cols, result.rows);
            }

            // Upscale
            if (method.first == "Nearest_Neighbor") {
                cv::resize(result, result, upscale_size, 0, 0, cv::INTER_NEAREST);
            } else if (method.first == "Bilinear") {
                cv::resize(result, result, upscale_size, 0, 0, cv::INTER_LINEAR);
            } else {
                if (is_debug_ && i == 0) {
                    std::cout << "   - Invalid scale method.\n"
                             << "   - UpScaling with default Nearest Neighbour method..." << std::endl;
                }
                cv::resize(result, result, upscale_size, 0, 0, cv::INTER_NEAREST);
            }

            // Downscale
            cv::Size downscale_size;
            if (i == 0) {
                downscale_size = cv::Size(result.cols, std::round(result.rows / downscale_fact));
            } else {
                downscale_size = cv::Size(std::round(result.cols / downscale_fact), result.rows);
            }

            if (method.second == "Nearest_Neighbor") {
                cv::resize(result, result, downscale_size, 0, 0, cv::INTER_NEAREST);
            } else if (method.second == "Bilinear") {
                cv::resize(result, result, downscale_size, 0, 0, cv::INTER_LINEAR);
            } else {
                if (is_debug_ && i == 0) {
                    std::cout << "   - Invalid scale method.\n"
                             << "   - DownScaling with default Nearest Neighbour method..." << std::endl;
                }
                cv::resize(result, result, downscale_size, 0, 0, cv::INTER_NEAREST);
            }
        }
    }

    return result;
}

std::vector<std::vector<std::pair<int, int>>> Scale2D::validate_input_output() {
    std::vector<std::pair<int, int>> valid_sizes = {
        {1080, 1920}, {1536, 2592}, {1944, 2592}
    };

    // Check if input size is valid
    if (std::find(valid_sizes.begin(), valid_sizes.end(), old_size_) == valid_sizes.end()) {
        return {{std::make_pair(0, 0), std::make_pair(0, 0), std::make_pair(0, 0)},
                {std::make_pair(0, 0), std::make_pair(0, 0), std::make_pair(0, 0)}};
    }

    // Determine scale factors based on input and output sizes
    std::vector<std::vector<std::pair<int, int>>> scale_info(2, std::vector<std::pair<int, int>>(3));

    if (old_size_ == std::make_pair(1080, 1920)) {
        if (new_size_ == std::make_pair(720, 1280)) {
            scale_info[0][2] = std::make_pair(2, 3);
            scale_info[1][2] = std::make_pair(2, 3);
        } else if (new_size_ == std::make_pair(480, 640)) {
            scale_info[0][0] = std::make_pair(2, 0);
            scale_info[0][1] = std::make_pair(60, 0);
            scale_info[1][0] = std::make_pair(3, 0);
        } else if (new_size_ == std::make_pair(360, 640)) {
            scale_info[0][0] = std::make_pair(3, 0);
            scale_info[1][0] = std::make_pair(3, 0);
        }
    }
    // Add more size combinations as needed...

    return scale_info;
}

cv::Mat Scale2D::apply_algo(
    const std::vector<std::vector<std::pair<int, int>>>& scale_info,
    const std::pair<std::string, std::string>& method) {
    
    cv::Mat result = single_channel_.clone();

    // Check if input size is valid
    if (scale_info[0][0].first == 0 && scale_info[0][1].first == 0 && scale_info[0][2].first == 0) {
        std::cout << "   - Invalid input size. It must be one of the following:\n"
                  << "   - 1920x1080\n"
                  << "   - 2592x1536\n"
                  << "   - 2592x1944" << std::endl;
        return result;
    }

    // Check if output size is valid
    if (scale_info[0][0].first == 1 && scale_info[0][1].first == 0 && scale_info[0][2].first == 0 &&
        scale_info[1][0].first == 1 && scale_info[1][1].first == 0 && scale_info[1][2].first == 0) {
        std::cout << "   - Invalid output size." << std::endl;
        return result;
    }

    // Step 1: Downscale by int factor
    if (scale_info[0][0].first > 1 || scale_info[1][0].first > 1) {
        cv::Size downscale_size(
            old_size_.second / scale_info[1][0].first,
            old_size_.first / scale_info[0][0].first
        );
        cv::resize(result, result, downscale_size, 0, 0, cv::INTER_LINEAR);

        if (is_debug_) {
            std::cout << "   - Shape after downscaling by integer factor ("
                     << scale_info[0][0].first << ", " << scale_info[1][0].first << "): "
                     << result.size() << std::endl;
        }
    }

    // Step 2: Crop
    if (scale_info[0][1].first > 0 || scale_info[1][1].first > 0) {
        cv::Rect crop_rect(
            scale_info[1][1].first,
            scale_info[0][1].first,
            result.cols - 2 * scale_info[1][1].first,
            result.rows - 2 * scale_info[0][1].first
        );
        result = result(crop_rect);

        if (is_debug_) {
            std::cout << "   - Shape after cropping ("
                     << scale_info[0][1].first << ", " << scale_info[1][1].first << "): "
                     << result.size() << std::endl;
        }
    }

    // Step 3: Scale with non-int factor
    if (scale_info[0][2].first > 0 || scale_info[1][2].first > 0) {
        result = resize_by_non_int_fact(
            std::make_pair(scale_info[0][2], scale_info[1][2]),
            method
        );

        if (is_debug_) {
            std::cout << "   - Shape after scaling by non-integer factor ("
                     << scale_info[0][2].first << ", " << scale_info[1][2].first << "): "
                     << result.size() << std::endl;
        }
    }

    return result;
}

cv::Mat Scale2D::hardware_dep_scaling() {
    std::vector<std::vector<std::pair<int, int>>> scale_info = validate_input_output();
    return apply_algo(scale_info, std::make_pair(upscale_method_, downscale_method_));
}

cv::Mat Scale2D::hardware_indp_scaling() {
    if (algo_ == "Nearest_Neighbor") {
        if (is_debug_) {
            std::cout << "   - Scaling with Nearest Neighbor method..." << std::endl;
        }
        cv::Mat result;
        cv::resize(single_channel_, result, cv::Size(new_size_.second, new_size_.first), 0, 0, cv::INTER_NEAREST);
        return result;
    } else if (algo_ == "Bilinear") {
        if (is_debug_) {
            std::cout << "   - Scaling with Bilinear method..." << std::endl;
        }
        cv::Mat result;
        cv::resize(single_channel_, result, cv::Size(new_size_.second, new_size_.first), 0, 0, cv::INTER_LINEAR);
        return result;
    } else {
        if (is_debug_) {
            std::cout << "   - Invalid scale method.\n"
                     << "   - Scaling with default Nearest Neighbor method..." << std::endl;
        }
        cv::Mat result;
        cv::resize(single_channel_, result, cv::Size(new_size_.second, new_size_.first), 0, 0, cv::INTER_NEAREST);
        return result;
    }
}

cv::Mat Scale2D::execute() {
    if (is_hardware_) {
        single_channel_ = hardware_dep_scaling();
    } else {
        single_channel_ = hardware_indp_scaling();
    }

    if (is_debug_) {
        std::cout << "   - Shape of scaled image for a single channel = " << single_channel_.size() << std::endl;
    }

    return single_channel_;
} 
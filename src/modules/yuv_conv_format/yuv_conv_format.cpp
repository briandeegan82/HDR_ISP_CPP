#include "yuv_conv_format.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>

YUVConvFormat::YUVConvFormat(const cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                            const YAML::Node& parm_yuv)
    : img_(img)
    , shape_(img.size())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_yuv_(parm_yuv)
    , in_file_(platform["in_file"].as<std::string>())
    , is_enable_(parm_yuv["is_enable"].as<bool>())
    , is_save_(parm_yuv["is_save"].as<bool>())
{
}

cv::Mat YUVConvFormat::convert2yuv_format() {
    std::string conv_type = parm_yuv_["conv_type"].as<std::string>();
    cv::Mat yuv;

    if (conv_type == "422") {
        // Extract Y, U, V channels
        std::vector<cv::Mat> channels;
        cv::split(img_, channels);

        // Get Y0, U, Y1, V for 422 format
        cv::Mat y0 = channels[0](cv::Rect(0, 0, img_.cols/2, img_.rows));
        cv::Mat u = channels[1](cv::Rect(0, 0, img_.cols/2, img_.rows));
        cv::Mat y1 = channels[0](cv::Rect(img_.cols/2, 0, img_.cols/2, img_.rows));
        cv::Mat v = channels[2](cv::Rect(0, 0, img_.cols/2, img_.rows));

        // Reshape to column vectors
        y0 = y0.reshape(1, y0.total());
        u = u.reshape(1, u.total());
        y1 = y1.reshape(1, y1.total());
        v = v.reshape(1, v.total());

        // Concatenate Y0, U, Y1, V
        std::vector<cv::Mat> yuv_planes = {y0, u, y1, v};
        cv::hconcat(yuv_planes, yuv);
    }
    else if (conv_type == "444") {
        // Extract Y, U, V channels
        std::vector<cv::Mat> channels;
        cv::split(img_, channels);

        // Reshape to column vectors
        cv::Mat y = channels[0].reshape(1, channels[0].total());
        cv::Mat u = channels[1].reshape(1, channels[1].total());
        cv::Mat v = channels[2].reshape(1, channels[2].total());

        // Concatenate Y, U, V
        std::vector<cv::Mat> yuv_planes = {y, u, v};
        cv::hconcat(yuv_planes, yuv);
    }

    // Save YUV data to file
    std::filesystem::path out_path = "out_frames/out_" + in_file_ + ".yuv";
    std::filesystem::create_directories(out_path.parent_path());
    
    std::ofstream raw_wb(out_path, std::ios::binary);
    if (raw_wb.is_open()) {
        raw_wb.write(reinterpret_cast<const char*>(yuv.data), yuv.total() * yuv.elemSize());
        raw_wb.close();
    }

    return yuv.reshape(1, yuv.total());
}

void YUVConvFormat::save() {
    if (is_save_) {
        // Update size in filename
        std::regex size_pattern(R"(\d+x\d+)");
        std::string new_size = std::to_string(shape_.width) + "x" + std::to_string(shape_.height);
        in_file_ = std::regex_replace(in_file_, size_pattern, new_size);

        // Save with .npy format
        std::string original_format = platform_["save_format"].as<std::string>();
        platform_["save_format"] = "npy";

        std::filesystem::path output_path = "out_frames/intermediate";
        std::filesystem::create_directories(output_path);
        
        std::string filename = "Out_yuv_conversion_format_" + 
                             parm_yuv_["conv_type"].as<std::string>() + "_" + in_file_;
        cv::imwrite((output_path / filename).string(), img_);

        // Restore original format
        platform_["save_format"] = original_format;
    }
}

cv::Mat YUVConvFormat::execute() {
    if (is_enable_) {
        if (platform_["rgb_output"].as<bool>()) {
            if (parm_yuv_["is_debug"].as<bool>()) {
                std::cout << "Invalid input for YUV conversion: RGB image format." << std::endl;
            }
            parm_yuv_["is_enable"] = false;
        } else {
            auto start = std::chrono::high_resolution_clock::now();
            
            img_ = convert2yuv_format();
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            if (parm_yuv_["is_debug"].as<bool>()) {
                std::cout << "YUV conversion completed in " << duration.count() << " ms" << std::endl;
            }
        }
    }

    save();
    return img_;
} 
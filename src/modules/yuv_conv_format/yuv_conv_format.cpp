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
    , use_eigen_(true) // Use Eigen by default
{
}

cv::Mat YUVConvFormat::convert2yuv_format_opencv() {
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

hdr_isp::EigenImage YUVConvFormat::convert2yuv_format_eigen() {
    std::string conv_type = parm_yuv_["conv_type"].as<std::string>();
    
    // Convert input to Eigen
    hdr_isp::EigenImage eigen_img = hdr_isp::opencv_to_eigen(img_);
    int rows = eigen_img.rows();
    int cols = eigen_img.cols();
    
    hdr_isp::EigenImage yuv;
    
    if (conv_type == "422") {
        // Extract Y, U, V channels using Eigen
        Eigen::MatrixXf y_channel = eigen_img.data().block(0, 0, rows, cols/3);
        Eigen::MatrixXf u_channel = eigen_img.data().block(0, cols/3, rows, cols/3);
        Eigen::MatrixXf v_channel = eigen_img.data().block(0, 2*cols/3, rows, cols/3);
        
        // Get Y0, U, Y1, V for 422 format
        Eigen::MatrixXf y0 = y_channel.block(0, 0, rows, cols/6);
        Eigen::MatrixXf u_half = u_channel.block(0, 0, rows, cols/6);
        Eigen::MatrixXf y1 = y_channel.block(0, cols/6, rows, cols/6);
        Eigen::MatrixXf v_half = v_channel.block(0, 0, rows, cols/6);
        
        // Reshape to column vectors
        Eigen::VectorXf y0_vec = Eigen::Map<Eigen::VectorXf>(y0.data(), y0.size());
        Eigen::VectorXf u_vec = Eigen::Map<Eigen::VectorXf>(u_half.data(), u_half.size());
        Eigen::VectorXf y1_vec = Eigen::Map<Eigen::VectorXf>(y1.data(), y1.size());
        Eigen::VectorXf v_vec = Eigen::Map<Eigen::VectorXf>(v_half.data(), v_half.size());
        
        // Concatenate Y0, U, Y1, V
        int total_size = y0_vec.size() + u_vec.size() + y1_vec.size() + v_vec.size();
        Eigen::MatrixXf yuv_matrix(total_size, 1);
        yuv_matrix.block(0, 0, y0_vec.size(), 1) = y0_vec;
        yuv_matrix.block(y0_vec.size(), 0, u_vec.size(), 1) = u_vec;
        yuv_matrix.block(y0_vec.size() + u_vec.size(), 0, y1_vec.size(), 1) = y1_vec;
        yuv_matrix.block(y0_vec.size() + u_vec.size() + y1_vec.size(), 0, v_vec.size(), 1) = v_vec;
        yuv = hdr_isp::EigenImage(yuv_matrix);
    }
    else if (conv_type == "444") {
        // Extract Y, U, V channels using Eigen
        Eigen::MatrixXf y_channel = eigen_img.data().block(0, 0, rows, cols/3);
        Eigen::MatrixXf u_channel = eigen_img.data().block(0, cols/3, rows, cols/3);
        Eigen::MatrixXf v_channel = eigen_img.data().block(0, 2*cols/3, rows, cols/3);
        
        // Reshape to column vectors
        Eigen::VectorXf y_vec = Eigen::Map<Eigen::VectorXf>(y_channel.data(), y_channel.size());
        Eigen::VectorXf u_vec = Eigen::Map<Eigen::VectorXf>(u_channel.data(), u_channel.size());
        Eigen::VectorXf v_vec = Eigen::Map<Eigen::VectorXf>(v_channel.data(), v_channel.size());
        
        // Concatenate Y, U, V
        int total_size = y_vec.size() + u_vec.size() + v_vec.size();
        Eigen::MatrixXf yuv_matrix(total_size, 1);
        yuv_matrix.block(0, 0, y_vec.size(), 1) = y_vec;
        yuv_matrix.block(y_vec.size(), 0, u_vec.size(), 1) = u_vec;
        yuv_matrix.block(y_vec.size() + u_vec.size(), 0, v_vec.size(), 1) = v_vec;
        yuv = hdr_isp::EigenImage(yuv_matrix);
    }
    
    // Save YUV data to file (convert back to OpenCV for file I/O)
    cv::Mat yuv_cv = hdr_isp::eigen_to_opencv(yuv);
    std::filesystem::path out_path = "out_frames/out_" + in_file_ + ".yuv";
    std::filesystem::create_directories(out_path.parent_path());
    
    std::ofstream raw_wb(out_path, std::ios::binary);
    if (raw_wb.is_open()) {
        raw_wb.write(reinterpret_cast<const char*>(yuv_cv.data), yuv_cv.total() * yuv_cv.elemSize());
        raw_wb.close();
    }
    
    return yuv;
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
            
            if (use_eigen_) {
                hdr_isp::EigenImage eigen_result = convert2yuv_format_eigen();
                img_ = hdr_isp::eigen_to_opencv(eigen_result);
            } else {
                img_ = convert2yuv_format_opencv();
            }
            
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
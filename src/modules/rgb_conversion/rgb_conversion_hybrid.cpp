#include "rgb_conversion_hybrid.hpp"
#include "../../../include/isp_backend_wrapper.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

RGBConversionHybrid::RGBConversionHybrid(cv::Mat& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                                        const YAML::Node& parm_rgb, const YAML::Node& parm_csc)
    : img_(img.clone())
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_rgb_(parm_rgb)
    , parm_csc_(parm_csc)
    , enable_(parm_rgb["is_enable"].as<bool>())
    , is_save_(parm_rgb["is_save"].as<bool>())
    , is_debug_(parm_rgb["is_debug"].IsDefined() ? parm_rgb["is_debug"].as<bool>() : false)
    , bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , conv_std_(parm_csc["conv_standard"].as<int>())
    , yuv_img_(img)
    , use_eigen_(true)
    , has_eigen_input_(false)
    , preferred_backend_(hdr_isp::BackendType::AUTO)
{
    // Pre-compute conversion matrices
    if (conv_std_ == 1) {
        // BT.709
        yuv2rgb_mat_ = (cv::Mat_<int>(3, 3) << 74, 0, 114,
                                             74, -13, -34,
                                             74, 135, 0);
    } else {
        // BT.601/407
        yuv2rgb_mat_ = (cv::Mat_<int>(3, 3) << 64, 87, 0,
                                             64, -44, -20,
                                             61, 0, 105);
    }

    // Pre-compute offset array
    offset_ = cv::Vec3i(16, 128, 128);
    
    // Initialize backend
    initializeBackend();
}

RGBConversionHybrid::RGBConversionHybrid(const hdr_isp::EigenImage3C& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                                        const YAML::Node& parm_rgb, const YAML::Node& parm_csc)
    : eigen_img_(img)
    , platform_(platform)
    , sensor_info_(sensor_info)
    , parm_rgb_(parm_rgb)
    , parm_csc_(parm_csc)
    , enable_(parm_rgb["is_enable"].as<bool>())
    , is_save_(parm_rgb["is_save"].as<bool>())
    , is_debug_(parm_rgb["is_debug"].IsDefined() ? parm_rgb["is_debug"].as<bool>() : false)
    , bit_depth_(sensor_info["output_bit_depth"].as<int>())
    , conv_std_(parm_csc["conv_standard"].as<int>())
    , use_eigen_(true)
    , has_eigen_input_(true)
    , preferred_backend_(hdr_isp::BackendType::AUTO)
{
    // Pre-compute conversion matrices
    if (conv_std_ == 1) {
        // BT.709
        yuv2rgb_mat_ = (cv::Mat_<int>(3, 3) << 74, 0, 114,
                                             74, -13, -34,
                                             74, 135, 0);
    } else {
        // BT.601/407
        yuv2rgb_mat_ = (cv::Mat_<int>(3, 3) << 64, 87, 0,
                                             64, -44, -20,
                                             61, 0, 105);
    }
    offset_ = cv::Vec3i(16, 128, 128);
    
    // Initialize backend
    initializeBackend();
}

void RGBConversionHybrid::initializeBackend() {
#ifdef USE_HYBRID_BACKEND
    if (hdr_isp::g_backend) {
        hdr_isp::g_backend->initialize(preferred_backend_);
        std::cout << "RGB Conversion Hybrid: Using backend: " << hdr_isp::g_backend->getBackendName() << std::endl;
    }
#endif
}

cv::Mat RGBConversionHybrid::execute() {
    if (!enable_) {
        return has_eigen_input_ ? eigen_img_.toOpenCV() : img_;
    }

    std::cout << "RGB Conversion Hybrid: Starting execution..." << std::endl;
    
    // Start performance monitoring
#ifdef USE_HYBRID_BACKEND
    hdr_isp::ISPBackendWrapper::startTimer();
#endif

    cv::Mat result;
    
    if (has_eigen_input_) {
        // Convert Eigen to OpenCV for processing
        yuv_img_ = eigen_img_.toOpenCV();
    }

    // Try optimized backend first
#ifdef USE_HYBRID_BACKEND
    if (hdr_isp::ISPBackendWrapper::isOptimizedBackendAvailable()) {
        try {
            if (hdr_isp::g_backend->getCurrentBackend() == hdr_isp::BackendType::OPENCV_OPENCL) {
                result = yuv_to_rgb_opencv_opencl();
            } else if (hdr_isp::g_backend->getCurrentBackend() == hdr_isp::BackendType::HALIDE_CPU || 
                       hdr_isp::g_backend->getCurrentBackend() == hdr_isp::BackendType::HALIDE_OPENCL) {
                result = yuv_to_rgb_halide();
            } else {
                result = yuv_to_rgb_opencv();
            }
        } catch (const std::exception& e) {
            std::cerr << "Optimized RGB conversion failed, falling back to OpenCV: " << e.what() << std::endl;
            result = yuv_to_rgb_opencv();
        }
    } else {
        result = yuv_to_rgb_opencv();
    }
#else
    result = yuv_to_rgb_opencv();
#endif

    // End performance monitoring
#ifdef USE_HYBRID_BACKEND
    double time_ms = hdr_isp::ISPBackendWrapper::endTimer();
    std::cout << "RGB Conversion Hybrid: Total execution time: " << time_ms << "ms" << std::endl;
#endif

    if (is_save_) {
        save();
    }

    return result;
}

hdr_isp::EigenImage3C RGBConversionHybrid::execute_eigen() {
    if (!enable_) {
        return has_eigen_input_ ? eigen_img_ : hdr_isp::EigenImage3C::fromOpenCV(img_);
    }

    // Convert to OpenCV, process, then convert back to Eigen
    cv::Mat result = execute();
    return hdr_isp::EigenImage3C::fromOpenCV(result);
}

cv::Mat RGBConversionHybrid::yuv_to_rgb_opencv() {
    // Standard OpenCV implementation (same as original)
    auto start_total = std::chrono::high_resolution_clock::now();

    if (yuv_img_.channels() == 1) {
        // Reshape and subtract offset
        auto start_reshape = std::chrono::high_resolution_clock::now();
        cv::Mat mat_2d = yuv_img_.reshape(1, yuv_img_.total());
        cv::Mat mat_2d_float;
        mat_2d.convertTo(mat_2d_float, CV_32F);
        mat_2d_float -= cv::Scalar(offset_[0], offset_[1], offset_[2]);
        cv::Mat mat2d_t = mat_2d_float.t();
        auto end_reshape = std::chrono::high_resolution_clock::now();

        // Matrix multiplication
        auto start_mult = std::chrono::high_resolution_clock::now();
        cv::Mat yuv2rgb_float;
        yuv2rgb_mat_.convertTo(yuv2rgb_float, CV_32F);
        cv::Mat rgb_2d = yuv2rgb_float * mat2d_t;
        rgb_2d = rgb_2d / 64.0;
        auto end_mult = std::chrono::high_resolution_clock::now();

        // Final conversion
        auto start_final = std::chrono::high_resolution_clock::now();
        cv::Mat rgb_2d_t = rgb_2d.t();
        cv::Mat rgb_reshaped = rgb_2d_t.reshape(3, yuv_img_.rows);
        cv::Mat rgb_clipped;
        cv::threshold(rgb_reshaped, rgb_clipped, 255, 255, cv::THRESH_TRUNC);
        cv::threshold(rgb_clipped, rgb_clipped, 0, 0, cv::THRESH_TOZERO);
        rgb_clipped.convertTo(yuv_img_, CV_8UC3);
        auto end_final = std::chrono::high_resolution_clock::now();

        // Print timing information
        std::chrono::duration<double, std::milli> reshape_time = end_reshape - start_reshape;
        std::chrono::duration<double, std::milli> mult_time = end_mult - start_mult;
        std::chrono::duration<double, std::milli> final_time = end_final - start_final;
        std::chrono::duration<double, std::milli> total_time = end_final - start_total;

        std::cout << "  Matrix reshaping and offset time: " << reshape_time.count() << "ms" << std::endl;
        std::cout << "  Matrix multiplication time: " << mult_time.count() << "ms" << std::endl;
        std::cout << "  Final conversion time: " << final_time.count() << "ms" << std::endl;
        std::cout << "  Total RGB conversion time: " << total_time.count() << "ms" << std::endl;

        return yuv_img_;
    } else if (yuv_img_.channels() == 3) {
        auto end_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_time = end_total - start_total;
        
        std::cout << "  Multi-channel image detected - passing through without conversion" << std::endl;
        std::cout << "  Total RGB conversion time: " << total_time.count() << "ms" << std::endl;
        
        return yuv_img_.clone();
    } else {
        throw std::runtime_error("Unsupported number of channels. Use 1 or 3 channels.");
    }
}

cv::Mat RGBConversionHybrid::yuv_to_rgb_opencv_opencl() {
    // OpenCV OpenCL implementation
    auto start_total = std::chrono::high_resolution_clock::now();

    if (yuv_img_.channels() == 1) {
        // Use OpenCV OpenCL for matrix operations
        cv::UMat input_umat, output_umat;
        yuv_img_.copyTo(input_umat);

        // Reshape and subtract offset
        auto start_reshape = std::chrono::high_resolution_clock::now();
        cv::UMat mat_2d = input_umat.reshape(1, input_umat.total());
        cv::UMat mat_2d_float;
        mat_2d.convertTo(mat_2d_float, CV_32F);
        mat_2d_float -= cv::Scalar(offset_[0], offset_[1], offset_[2]);
        cv::UMat mat2d_t = mat_2d_float.t();
        auto end_reshape = std::chrono::high_resolution_clock::now();

        // Matrix multiplication using OpenCL
        auto start_mult = std::chrono::high_resolution_clock::now();
        cv::UMat yuv2rgb_float;
        yuv2rgb_mat_.copyTo(yuv2rgb_float);
        yuv2rgb_float.convertTo(yuv2rgb_float, CV_32F);
        cv::UMat rgb_2d;
        cv::gemm(yuv2rgb_float, mat2d_t, 1.0, cv::UMat(), 0.0, rgb_2d);
        rgb_2d = rgb_2d / 64.0;
        auto end_mult = std::chrono::high_resolution_clock::now();

        // Final conversion
        auto start_final = std::chrono::high_resolution_clock::now();
        cv::UMat rgb_2d_t = rgb_2d.t();
        cv::UMat rgb_reshaped = rgb_2d_t.reshape(3, yuv_img_.rows);
        cv::UMat rgb_clipped;
        cv::threshold(rgb_reshaped, rgb_clipped, 255, 255, cv::THRESH_TRUNC);
        cv::threshold(rgb_clipped, rgb_clipped, 0, 0, cv::THRESH_TOZERO);
        cv::Mat result;
        rgb_clipped.convertTo(result, CV_8UC3);
        auto end_final = std::chrono::high_resolution_clock::now();

        // Print timing information
        std::chrono::duration<double, std::milli> reshape_time = end_reshape - start_reshape;
        std::chrono::duration<double, std::milli> mult_time = end_mult - start_mult;
        std::chrono::duration<double, std::milli> final_time = end_final - start_final;
        std::chrono::duration<double, std::milli> total_time = end_final - start_total;

        std::cout << "  OpenCL Matrix reshaping and offset time: " << reshape_time.count() << "ms" << std::endl;
        std::cout << "  OpenCL Matrix multiplication time: " << mult_time.count() << "ms" << std::endl;
        std::cout << "  OpenCL Final conversion time: " << final_time.count() << "ms" << std::endl;
        std::cout << "  OpenCL Total RGB conversion time: " << total_time.count() << "ms" << std::endl;

        return result;
    } else if (yuv_img_.channels() == 3) {
        auto end_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_time = end_total - start_total;
        
        std::cout << "  Multi-channel image detected - passing through without conversion" << std::endl;
        std::cout << "  OpenCL Total RGB conversion time: " << total_time.count() << "ms" << std::endl;
        
        return yuv_img_.clone();
    } else {
        throw std::runtime_error("Unsupported number of channels. Use 1 or 3 channels.");
    }
}

cv::Mat RGBConversionHybrid::yuv_to_rgb_halide() {
    // Halide implementation (simplified - would use actual Halide kernels in full implementation)
    auto start_total = std::chrono::high_resolution_clock::now();

    if (yuv_img_.channels() == 1) {
        // For now, use the optimized matrix multiplication from ISPBackendWrapper
        auto start_reshape = std::chrono::high_resolution_clock::now();
        cv::Mat mat_2d = yuv_img_.reshape(1, yuv_img_.total());
        cv::Mat mat_2d_float;
        mat_2d.convertTo(mat_2d_float, CV_32F);
        mat_2d_float -= cv::Scalar(offset_[0], offset_[1], offset_[2]);
        cv::Mat mat2d_t = mat_2d_float.t();
        auto end_reshape = std::chrono::high_resolution_clock::now();

        // Use optimized matrix multiplication
        auto start_mult = std::chrono::high_resolution_clock::now();
        cv::Mat yuv2rgb_float;
        yuv2rgb_mat_.convertTo(yuv2rgb_float, CV_32F);
        cv::Mat rgb_2d = hdr_isp::ISPBackendWrapper::matrixMultiplication(yuv2rgb_float, mat2d_t);
        rgb_2d = rgb_2d / 64.0;
        auto end_mult = std::chrono::high_resolution_clock::now();

        // Final conversion
        auto start_final = std::chrono::high_resolution_clock::now();
        cv::Mat rgb_2d_t = rgb_2d.t();
        cv::Mat rgb_reshaped = rgb_2d_t.reshape(3, yuv_img_.rows);
        cv::Mat rgb_clipped;
        cv::threshold(rgb_reshaped, rgb_clipped, 255, 255, cv::THRESH_TRUNC);
        cv::threshold(rgb_clipped, rgb_clipped, 0, 0, cv::THRESH_TOZERO);
        rgb_clipped.convertTo(rgb_clipped, CV_8UC3);
        auto end_final = std::chrono::high_resolution_clock::now();

        // Print timing information
        std::chrono::duration<double, std::milli> reshape_time = end_reshape - start_reshape;
        std::chrono::duration<double, std::milli> mult_time = end_mult - start_mult;
        std::chrono::duration<double, std::milli> final_time = end_final - start_final;
        std::chrono::duration<double, std::milli> total_time = end_final - start_total;

        std::cout << "  Halide Matrix reshaping and offset time: " << reshape_time.count() << "ms" << std::endl;
        std::cout << "  Halide Matrix multiplication time: " << mult_time.count() << "ms" << std::endl;
        std::cout << "  Halide Final conversion time: " << final_time.count() << "ms" << std::endl;
        std::cout << "  Halide Total RGB conversion time: " << total_time.count() << "ms" << std::endl;

        return rgb_clipped;
    } else if (yuv_img_.channels() == 3) {
        auto end_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_time = end_total - start_total;
        
        std::cout << "  Multi-channel image detected - passing through without conversion" << std::endl;
        std::cout << "  Halide Total RGB conversion time: " << total_time.count() << "ms" << std::endl;
        
        return yuv_img_.clone();
    } else {
        throw std::runtime_error("Unsupported number of channels. Use 1 or 3 channels.");
    }
}

cv::Mat RGBConversionHybrid::yuv_to_rgb_halide_opencl() {
    // Halide OpenCL implementation (same as Halide CPU for now)
    return yuv_to_rgb_halide();
}

void RGBConversionHybrid::save() {
    if (!is_save_) return;

    fs::path output_dir = fs::path(PROJECT_ROOT_DIR) / "out_frames";
    fs::create_directories(output_dir);
    
    std::string filename = "rgb_conversion_hybrid_output.png";
    fs::path output_path = output_dir / filename;
    
    cv::Mat output_img = has_eigen_input_ ? eigen_img_.toOpenCV() : img_;
    cv::imwrite(output_path.string(), output_img);
    
    std::cout << "RGB Conversion Hybrid: Output saved to " << output_path << std::endl;
} 
#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include "../../common/eigen_utils.hpp"

// Forward declarations for Halide
#ifdef USE_HYBRID_BACKEND
#include <Halide.h>
#endif

enum class HDRBackend {
    OPENCV_CPU,      // OpenCV CPU implementation
    OPENCV_OPENCL,   // OpenCV OpenCL GPU acceleration
    HALIDE_CPU,      // Halide CPU optimization
    HALIDE_OPENCL,   // Halide OpenCL GPU acceleration
    SIMD_OPTIMIZED,  // SIMD vectorized CPU implementation
    AUTO            // Auto-select best available
};

class HDRDurandToneMappingHybrid {
public:
    HDRDurandToneMappingHybrid(const cv::Mat& img, const YAML::Node& platform,
                               const YAML::Node& sensor_info, const YAML::Node& params);
    HDRDurandToneMappingHybrid(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
                               const YAML::Node& sensor_info, const YAML::Node& params);

    cv::Mat execute();
    hdr_isp::EigenImageU32 execute_eigen();

    // Backend selection
    void setBackend(HDRBackend backend);
    HDRBackend getBackend() const { return current_backend_; }
    bool isBackendAvailable(HDRBackend backend) const;

private:
    // Backend-specific implementations
    cv::Mat execute_opencv_cpu();
    cv::Mat execute_opencv_opencl();
    cv::Mat execute_halide_cpu();
    cv::Mat execute_halide_opencl();
    cv::Mat execute_simd_optimized();
    
    hdr_isp::EigenImageU32 execute_eigen_opencv_cpu();
    hdr_isp::EigenImageU32 execute_eigen_opencv_opencl();
    hdr_isp::EigenImageU32 execute_eigen_halide_cpu();
    hdr_isp::EigenImageU32 execute_eigen_halide_opencl();
    hdr_isp::EigenImageU32 execute_eigen_simd_optimized();

    // OpenCV OpenCL specific functions
    cv::Mat tone_mapping_opencv_opencl(const cv::Mat& input);
    cv::Mat bilateral_filter_opencv_opencl(const cv::Mat& input, float sigma_color, float sigma_space);
    
    // Halide-specific functions (if available)
#ifdef USE_HYBRID_BACKEND
    Halide::Buffer<float> tone_mapping_halide_cpu(const Halide::Buffer<float>& input);
    Halide::Buffer<float> tone_mapping_halide_opencl(const Halide::Buffer<float>& input);
    Halide::Buffer<float> bilateral_filter_halide(const Halide::Buffer<float>& input, float sigma_color, float sigma_space);
    
    // Durand algorithm Halide functions
    Halide::Func apply_durand_algorithm_halide(const Halide::Buffer<float>& input);
    Halide::Func log_domain_conversion_halide(const Halide::Buffer<float>& input);
    Halide::Func bilateral_filter_halide_func(const Halide::Buffer<float>& input, float sigma_color, float sigma_space);
    Halide::Func contrast_compression_halide(const Halide::Buffer<float>& base_layer, float contrast_factor);
    Halide::Func detail_preservation_halide(const Halide::Buffer<float>& log_luminance, const Halide::Buffer<float>& base_layer);
    Halide::Func exponential_conversion_halide(const Halide::Buffer<float>& log_output);
    
    // Utility functions for data conversion
    Halide::Buffer<float> cv_mat_to_halide(const cv::Mat& cv_mat);
    Halide::Buffer<float> eigen_to_halide(const hdr_isp::EigenImage& eigen_img);
    cv::Mat halide_to_cv_mat(const Halide::Buffer<float>& buffer, int rows, int cols);
    hdr_isp::EigenImage halide_to_eigen(const Halide::Buffer<float>& buffer, int rows, int cols);
#endif

    // SIMD-optimized functions
    cv::Mat tone_mapping_simd_opencv(const cv::Mat& input);
    cv::Mat bilateral_filter_simd(const cv::Mat& input, float sigma_color, float sigma_space);
    
    // Vectorized mathematical operations using SIMD
    void log_domain_conversion_simd(const cv::Mat& input, cv::Mat& output);
    void exponential_conversion_simd(const cv::Mat& input, cv::Mat& output);
    void bilateral_filter_simd_impl(const cv::Mat& input, cv::Mat& output, float sigma_color, float sigma_space);
    
    // Utility functions
    void save();
    HDRBackend selectBestBackend() const;
    bool initializeBackends();
    
    // Member variables
    cv::Mat img_;
    hdr_isp::EigenImageU32 eigen_img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node params_;
    bool is_enable_;
    bool is_save_;
    bool is_debug_;
    float sigma_space_;
    float sigma_color_;
    float contrast_factor_;
    int downsample_factor_;
    int output_bit_depth_;
    bool use_eigen_; // Use Eigen by default
    bool has_eigen_input_; // Flag to indicate if input is Eigen
    
    HDRBackend current_backend_;
    bool opencv_opencl_available_;
    bool halide_available_;
    bool simd_available_;
    
    // Performance monitoring
    mutable double last_execution_time_ms_;
}; 
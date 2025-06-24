#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include "../../common/eigen_utils.hpp"
#include "../../common/fixed_point_utils.hpp"
#include "demosaic.hpp"

// Forward declarations for Halide
#ifdef USE_HYBRID_BACKEND
#include <Halide.h>
#endif

enum class DemosaicBackend {
    OPENCV_CPU,      // OpenCV CPU implementation
    OPENCV_OPENCL,   // OpenCV OpenCL GPU acceleration
    HALIDE_CPU,      // Halide CPU optimization
    HALIDE_OPENCL,   // Halide OpenCL GPU acceleration
    SIMD_OPTIMIZED,  // SIMD vectorized CPU implementation
    AUTO            // Auto-select best available
};

class DemosaicHybrid : public Demosaic {
public:
    DemosaicHybrid(const hdr_isp::EigenImageU32& img, const std::string& bayer_pattern, int bit_depth = 16, bool is_save = true);
    DemosaicHybrid(const hdr_isp::EigenImageU32& img, const std::string& bayer_pattern, const hdr_isp::FixedPointConfig& fp_config, int bit_depth = 16, bool is_save = true);

    hdr_isp::EigenImage3C execute() override;
    hdr_isp::EigenImage3CFixed execute_fixed() override;

    // Backend selection
    void setBackend(DemosaicBackend backend);
    DemosaicBackend getBackend() const { return current_backend_; }
    bool isBackendAvailable(DemosaicBackend backend) const;

private:
    // Backend-specific implementations
    hdr_isp::EigenImage3C execute_opencv_cpu();
    hdr_isp::EigenImage3C execute_opencv_opencl();
    hdr_isp::EigenImage3C execute_halide_cpu();
    hdr_isp::EigenImage3C execute_halide_opencl();
    hdr_isp::EigenImage3C execute_simd_optimized();
    
    hdr_isp::EigenImage3CFixed execute_fixed_opencv_cpu();
    hdr_isp::EigenImage3CFixed execute_fixed_opencv_opencl();
    hdr_isp::EigenImage3CFixed execute_fixed_halide_cpu();
    hdr_isp::EigenImage3CFixed execute_fixed_halide_opencl();
    hdr_isp::EigenImage3CFixed execute_fixed_simd_optimized();

    // OpenCV OpenCL specific functions
    cv::Mat demosaic_opencv_opencl(const cv::Mat& input);
    int getOpenCVBayerPattern(const std::string& bayer_pattern);
    
    // Halide-specific functions (if available)
#ifdef USE_HYBRID_BACKEND
    Halide::Buffer<float> demosaic_halide_cpu(const Halide::Buffer<uint32_t>& input);
    Halide::Buffer<int16_t> demosaic_halide_fixed(const Halide::Buffer<uint32_t>& input);
    Halide::Buffer<float> demosaic_halide_opencl(const Halide::Buffer<uint32_t>& input);
    
    // Bayer pattern-specific Halide functions
    Halide::Func demosaic_rggb_halide(const Halide::Buffer<uint32_t>& input);
    Halide::Func demosaic_bggr_halide(const Halide::Buffer<uint32_t>& input);
    Halide::Func demosaic_grbg_halide(const Halide::Buffer<uint32_t>& input);
    Halide::Func demosaic_gbrg_halide(const Halide::Buffer<uint32_t>& input);
    
    // Vectorized interpolation functions
    Halide::Func interpolate_green_vectorized(const Halide::Buffer<uint32_t>& input, const std::string& pattern);
    Halide::Func interpolate_red_vectorized(const Halide::Buffer<uint32_t>& input, const std::string& pattern);
    Halide::Func interpolate_blue_vectorized(const Halide::Buffer<uint32_t>& input, const std::string& pattern);
    
    // Utility functions for data conversion
    Halide::Buffer<uint32_t> eigen_to_halide_uint32(const hdr_isp::EigenImageU32& eigen_img);
    hdr_isp::EigenImage3C halide_to_eigen_3c(const Halide::Buffer<float>& buffer, int rows, int cols);
    hdr_isp::EigenImage3CFixed halide_to_eigen_3c_fixed(const Halide::Buffer<int16_t>& buffer, int rows, int cols);
#endif

    // SIMD-optimized functions
    hdr_isp::EigenImage3C demosaic_simd_rggb(const hdr_isp::EigenImageU32& input);
    hdr_isp::EigenImage3C demosaic_simd_bggr(const hdr_isp::EigenImageU32& input);
    hdr_isp::EigenImage3C demosaic_simd_grbg(const hdr_isp::EigenImageU32& input);
    hdr_isp::EigenImage3C demosaic_simd_gbrg(const hdr_isp::EigenImageU32& input);
    
    // Vectorized interpolation using SIMD
    void interpolate_green_simd(const hdr_isp::EigenImageU32& input, hdr_isp::EigenImage& output, const std::string& pattern);
    void interpolate_red_simd(const hdr_isp::EigenImageU32& input, hdr_isp::EigenImage& output, const std::string& pattern);
    void interpolate_blue_simd(const hdr_isp::EigenImageU32& input, hdr_isp::EigenImage& output, const std::string& pattern);
    
    // Utility functions
    void save();
    DemosaicBackend selectBestBackend() const;
    bool initializeBackends();
    
    // Member variables
    DemosaicBackend current_backend_;
    bool opencv_opencl_available_;
    bool halide_available_;
    bool simd_available_;
    bool is_debug_;
    bool is_save_;
    
    // Performance monitoring
    mutable double last_execution_time_ms_;
}; 
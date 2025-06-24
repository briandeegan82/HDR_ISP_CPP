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

enum class GammaBackend {
    OPENCV_CPU,      // OpenCV CPU implementation
    OPENCV_OPENCL,   // OpenCV OpenCL GPU acceleration
    HALIDE_CPU,      // Halide CPU optimization
    HALIDE_OPENCL,   // Halide OpenCL GPU acceleration
    SIMD_OPTIMIZED,  // SIMD vectorized CPU implementation
    AUTO            // Auto-select best available
};

class GammaCorrectionHybrid {
public:
    GammaCorrectionHybrid(const hdr_isp::EigenImage3C& img, const YAML::Node& platform,
                         const YAML::Node& sensor_info, const YAML::Node& parm_gmm);

    hdr_isp::EigenImage3C execute();

    // Backend selection
    void setBackend(GammaBackend backend);
    GammaBackend getBackend() const { return current_backend_; }
    bool isBackendAvailable(GammaBackend backend) const;

private:
    // Backend-specific implementations
    hdr_isp::EigenImage3C execute_opencv_cpu();
    hdr_isp::EigenImage3C execute_opencv_opencl();
    hdr_isp::EigenImage3C execute_halide_cpu();
    hdr_isp::EigenImage3C execute_halide_opencl();
    hdr_isp::EigenImage3C execute_simd_optimized();

    // Halide-specific functions (if available)
#ifdef USE_HYBRID_BACKEND
    Halide::Buffer<float> apply_gamma_halide_cpu(const Halide::Buffer<float>& input);
    Halide::Buffer<float> apply_gamma_halide_opencl(const Halide::Buffer<float>& input);
    
    // Gamma LUT generation and application
    Halide::Buffer<uint32_t> generate_gamma_lut_halide(int bit_depth);
    Halide::Func apply_gamma_lut_halide(const Halide::Buffer<float>& input, const Halide::Buffer<uint32_t>& lut);
    
    // Utility functions for data conversion
    Halide::Buffer<float> eigen_to_halide_3c(const hdr_isp::EigenImage3C& eigen_img);
    hdr_isp::EigenImage3C halide_to_eigen_3c(const Halide::Buffer<float>& buffer, int rows, int cols);
#endif

    // SIMD-optimized functions
    hdr_isp::EigenImage3C apply_gamma_simd_opencv(const hdr_isp::EigenImage3C& input);
    
    // Vectorized LUT application using SIMD
    void apply_gamma_lut_simd(const hdr_isp::EigenImage3C& input, hdr_isp::EigenImage3C& output, const std::vector<uint32_t>& lut);

    // Helper functions
    void initializeBackends();
    GammaBackend selectBestBackend() const;
    void save();

    // Data members
    hdr_isp::EigenImage3C img_;
    const YAML::Node& platform_;
    const YAML::Node& sensor_info_;
    const YAML::Node& parm_gmm_;
    bool enable_;
    int output_bit_depth_;
    bool is_save_;
    bool is_debug_;
    
    // Backend selection
    GammaBackend current_backend_;
    bool opencv_opencl_available_;
    bool halide_available_;
    bool simd_available_;
    double last_execution_time_ms_;
}; 
#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include <Halide.h>
#include "../../common/eigen_utils.hpp"
#include "../../common/fixed_point_utils.hpp"
#include "color_correction_matrix.hpp"

class ColorCorrectionMatrixHybrid : public ColorCorrectionMatrix {
public:
    ColorCorrectionMatrixHybrid(const hdr_isp::EigenImage3C& img, const YAML::Node& sensor_info, 
                                const YAML::Node& parm_ccm, const hdr_isp::FixedPointConfig& fp_config);
    ColorCorrectionMatrixHybrid(const hdr_isp::EigenImage3CFixed& img, const YAML::Node& sensor_info, 
                                const YAML::Node& parm_ccm, const hdr_isp::FixedPointConfig& fp_config);

    hdr_isp::EigenImage3C execute() override;
    hdr_isp::EigenImage3CFixed execute_fixed() override;

private:
    // Halide-optimized matrix multiplication functions
    Halide::Buffer<float> apply_ccm_halide(const Halide::Buffer<float>& input);
    Halide::Buffer<int16_t> apply_ccm_fixed_halide(const Halide::Buffer<int16_t>& input);
    Halide::Buffer<float> apply_ccm_vectorized_halide(const Halide::Buffer<float>& input);
    
    // Utility functions for data conversion
    Halide::Buffer<float> eigen_to_halide_float(const hdr_isp::EigenImage3C& eigen_img);
    hdr_isp::EigenImage3C halide_to_eigen_float(const Halide::Buffer<float>& buffer, int rows, int cols);
    Halide::Buffer<int16_t> eigen_to_halide_fixed(const hdr_isp::EigenImage3CFixed& eigen_img);
    hdr_isp::EigenImage3CFixed halide_to_eigen_fixed(const Halide::Buffer<int16_t>& buffer, int rows, int cols);
    
    // Helper functions
    void save();
    
    // Member variables
    bool is_debug_;
    bool is_save_;
}; 
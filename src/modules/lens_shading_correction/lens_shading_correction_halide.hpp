#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include <Halide.h>
#include "../../common/eigen_utils.hpp"

class LensShadingCorrectionHalide {
public:
    LensShadingCorrectionHalide(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
                                const YAML::Node& sensor_info, const YAML::Node& parm_lsc);

    hdr_isp::EigenImageU32 execute();

private:
    Halide::Buffer<uint32_t> apply_lsc_halide(const Halide::Buffer<uint32_t>& input);
    Halide::Func create_shading_correction(Halide::Buffer<uint32_t> input);
    Halide::Func apply_radial_correction(Halide::Buffer<uint32_t> input, Halide::Func shading_correction);
    void save(const std::string& filename_tag);

    hdr_isp::EigenImageU32 img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_lsc_;
    bool enable_;
    bool is_save_;
    bool is_debug_;
}; 
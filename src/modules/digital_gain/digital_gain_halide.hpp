#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Halide.h>
#include "../../common/eigen_utils.hpp"

class DigitalGainHalide {
public:
    DigitalGainHalide(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
                      const YAML::Node& sensor_info, const YAML::Node& parm_dga);

    std::pair<hdr_isp::EigenImageU32, float> execute();

private:
    Halide::Buffer<uint32_t> apply_digital_gain_halide(const Halide::Buffer<uint32_t>& input);
    Halide::Func vectorized_multiply(const Halide::Buffer<uint32_t>& input, float gain);
    void save(const std::string& filename_tag);

    hdr_isp::EigenImageU32 img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_dga_;
    bool is_save_;
    bool is_debug_;
    bool is_auto_;
    std::vector<float> gains_array_;
    int current_gain_;
    float ae_feedback_;
}; 
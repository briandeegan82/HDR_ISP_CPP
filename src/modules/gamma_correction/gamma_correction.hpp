#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include "../../common/eigen_utils.hpp"

class GammaCorrection {
public:
    GammaCorrection(const hdr_isp::EigenImage3C& img, const YAML::Node& platform, 
                   const YAML::Node& sensor_info, const YAML::Node& parm_gmm);
    hdr_isp::EigenImage3C execute();

private:
    std::vector<uint32_t> generate_gamma_lut(int bit_depth);
    hdr_isp::EigenImage3C apply_gamma();
    void save();

    hdr_isp::EigenImage3C img_;
    bool enable_;
    YAML::Node sensor_info_;
    int output_bit_depth_;
    YAML::Node parm_gmm_;
    bool is_save_;
    bool is_debug_;
    YAML::Node platform_;
}; 
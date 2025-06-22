#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class DigitalGain {
public:
    DigitalGain(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
                const YAML::Node& sensor_info, const YAML::Node& parm_dga);

    std::pair<hdr_isp::EigenImageU32, float> execute();

private:
    hdr_isp::EigenImageU32 apply_digital_gain_eigen();
    void save();

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
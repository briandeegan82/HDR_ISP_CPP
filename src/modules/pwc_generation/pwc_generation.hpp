#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class PiecewiseCurve {
public:
    PiecewiseCurve(const hdr_isp::EigenImageU32& img, const YAML::Node& platform, const YAML::Node& sensor_info, const YAML::Node& parm_cmpd);

    hdr_isp::EigenImageU32 execute();

private:
    static std::vector<double> generate_decompanding_lut(
        const std::vector<int>& companded_pin,
        const std::vector<int>& companded_pout,
        int max_input_value = 4095
    );
    void save(const std::string& filename_tag);

    hdr_isp::EigenImageU32 apply_decompanding_eigen(const hdr_isp::EigenImageU32& img);

    hdr_isp::EigenImageU32 img_;
    const YAML::Node& platform_;
    const YAML::Node& sensor_info_;
    const YAML::Node& parm_cmpd_;
    bool enable_;
    int bit_depth_;
    std::vector<int> companded_pin_;
    std::vector<int> companded_pout_;
    bool is_save_;
    bool is_debug_;
}; 
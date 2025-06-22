#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class BlackLevelCorrection {
public:
    BlackLevelCorrection(const hdr_isp::EigenImageU32& img, const YAML::Node& sensor_info, const YAML::Node& parm_blc);
    
    hdr_isp::EigenImageU32 execute();

private:
    hdr_isp::EigenImageU32 apply_blc_parameters_eigen(const hdr_isp::EigenImageU32& img);
    void apply_blc_bayer_eigen(hdr_isp::EigenImageU32& img, int r_offset, int gr_offset, int gb_offset, int b_offset, int r_sat, int gr_sat, int gb_sat, int b_sat);
    void save(const std::string& filename_tag);
    
    hdr_isp::EigenImageU32 raw_;
    YAML::Node sensor_info_;
    YAML::Node parm_blc_;
    bool enable_;
    bool is_linearize_;
    int bit_depth_;
    std::string bayer_pattern_;
    bool is_save_;
}; 
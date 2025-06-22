#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class WhiteBalance {
public:
    WhiteBalance(const hdr_isp::EigenImageU32& img, const YAML::Node& platform, const YAML::Node& sensor_info,
                 const YAML::Node& parm_wbc);

    hdr_isp::EigenImageU32 execute();

private:
    hdr_isp::EigenImageU32 apply_wb_parameters();
    void save();

    hdr_isp::EigenImageU32 img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_wbc_;

    bool is_enable_;
    bool is_save_;
    bool is_auto_;
    bool is_debug_;
    std::string bayer_;
    int bpp_;
    hdr_isp::EigenImageU32 raw_;
}; 
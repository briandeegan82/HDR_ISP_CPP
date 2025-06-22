#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class LensShadingCorrection {
public:
    LensShadingCorrection(const hdr_isp::EigenImageU32& img, const YAML::Node& platform,
                          const YAML::Node& sensor_info, const YAML::Node& parm_lsc);

    hdr_isp::EigenImageU32 execute();

private:
    hdr_isp::EigenImageU32 apply_lsc_eigen(const hdr_isp::EigenImageU32& img);
    void save(const std::string& filename_tag);

    hdr_isp::EigenImageU32 img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_lsc_;
    bool enable_;
    bool is_save_;
    bool is_debug_;
}; 
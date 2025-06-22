#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class OECF {
public:
    OECF(const hdr_isp::EigenImageU32& img, const YAML::Node& platform, const YAML::Node& sensor_info, const YAML::Node& parm_oecf);

    hdr_isp::EigenImageU32 execute();

private:
    hdr_isp::EigenImageU32 apply_oecf_eigen(const hdr_isp::EigenImageU32& img);
    void save(const std::string& filename_tag);

    hdr_isp::EigenImageU32 img_;
    const YAML::Node& platform_;
    const YAML::Node& sensor_info_;
    const YAML::Node& parm_oecf_;
    bool enable_;
    bool is_save_;
    bool is_debug_;
}; 
#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class Crop {
public:
    Crop(const hdr_isp::EigenImageU32& img, const YAML::Node& platform, 
         const YAML::Node& sensor_info, const YAML::Node& parm_cro);

    hdr_isp::EigenImageU32 execute();

private:
    void update_sensor_info(YAML::Node& dictionary);
    hdr_isp::EigenImageU32 crop_eigen(const hdr_isp::EigenImageU32& img, int rows_to_crop, int cols_to_crop);
    hdr_isp::EigenImageU32 apply_cropping();
    void save(const std::string& filename_tag);

    hdr_isp::EigenImageU32 img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_cro_;
    std::pair<int, int> old_size_;
    std::pair<int, int> new_size_;
    bool enable_;
    bool is_debug_;
    bool is_save_;
}; 
#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"

class BayerNoiseReduction {
public:
    BayerNoiseReduction(const hdr_isp::EigenImageU32& img, const YAML::Node& sensor_info, const YAML::Node& parm_bnr);
    
    hdr_isp::EigenImageU32 execute();

private:
    hdr_isp::EigenImageU32 apply_bnr_eigen(const hdr_isp::EigenImageU32& img);
    void extract_channels_eigen(const hdr_isp::EigenImageU32& img, hdr_isp::EigenImageU32& r_channel, hdr_isp::EigenImageU32& b_channel);
    void combine_channels_eigen(const hdr_isp::EigenImageU32& r_channel, const hdr_isp::EigenImageU32& g_channel, const hdr_isp::EigenImageU32& b_channel, hdr_isp::EigenImageU32& output);
    hdr_isp::EigenImageU32 interpolate_green_channel_eigen(const hdr_isp::EigenImageU32& img);
    hdr_isp::EigenImageU32 bilateral_filter_eigen(const hdr_isp::EigenImageU32& src, int d, double sigmaColor, double sigmaSpace);
    void save(const std::string& filename_tag);

    hdr_isp::EigenImageU32 img_;
    YAML::Node sensor_info_;
    YAML::Node parm_bnr_;
    int bit_depth_;
    std::string bayer_pattern_;
    int width_;
    int height_;
    bool is_save_;
    bool is_debug_;
}; 
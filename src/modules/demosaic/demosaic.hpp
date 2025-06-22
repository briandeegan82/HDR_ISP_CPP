#pragma once

#include <string>
#include <array>
#include "../../common/eigen_utils.hpp"

class BilinearDemosaic {
public:
    BilinearDemosaic(const hdr_isp::EigenImageU32& raw_in, const std::string& bayer_pattern);
    hdr_isp::EigenImage3C execute();

private:
    hdr_isp::EigenImageU32 raw_in_;
    std::string bayer_pattern_;
    int rows_;
    int cols_;

    // Bayer pattern masks
    std::array<hdr_isp::EigenImage, 3> create_bayer_masks();
    
    // Bilinear interpolation functions
    hdr_isp::EigenImage interpolate_green();
    hdr_isp::EigenImage interpolate_red();
    hdr_isp::EigenImage interpolate_blue();
    
    // Helper functions for different bayer patterns
    hdr_isp::EigenImage interpolate_green_rggb();
    hdr_isp::EigenImage interpolate_green_bggr();
    hdr_isp::EigenImage interpolate_green_grbg();
    hdr_isp::EigenImage interpolate_green_gbrg();
    
    hdr_isp::EigenImage interpolate_red_rggb();
    hdr_isp::EigenImage interpolate_red_bggr();
    hdr_isp::EigenImage interpolate_red_grbg();
    hdr_isp::EigenImage interpolate_red_gbrg();
    
    hdr_isp::EigenImage interpolate_blue_rggb();
    hdr_isp::EigenImage interpolate_blue_bggr();
    hdr_isp::EigenImage interpolate_blue_grbg();
    hdr_isp::EigenImage interpolate_blue_gbrg();
};

class Demosaic {
public:
    Demosaic(const hdr_isp::EigenImageU32& img, const std::string& bayer_pattern, int bit_depth = 16, bool is_save = true);
    hdr_isp::EigenImage3C execute();

private:
    hdr_isp::EigenImageU32 img_;
    std::string bayer_pattern_;
    int bit_depth_;
    bool is_save_;
    bool is_debug_;
    bool is_enable_;

    void save(const hdr_isp::EigenImage3C& result);
}; 
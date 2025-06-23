#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "../../common/eigen_utils.hpp"
#include "../../common/fixed_point_utils.hpp"

class LDCI {
public:
    LDCI(const hdr_isp::EigenImage3C& img, const YAML::Node& platform,
         const YAML::Node& sensor_info, const YAML::Node& params);
    LDCI(const hdr_isp::EigenImage3CFixed& img, const YAML::Node& platform,
         const YAML::Node& sensor_info, const YAML::Node& params,
         const hdr_isp::FixedPointConfig& fp_config);

    hdr_isp::EigenImage3C execute();
    hdr_isp::EigenImage3CFixed execute_fixed();
    
    // Alias for pipeline compatibility
    hdr_isp::EigenImage3C execute_eigen() { return execute(); }

private:
    // OpenCV CLAHE-based implementation
    hdr_isp::EigenImage3C apply_ldci_opencv();
    hdr_isp::EigenImage3CFixed apply_ldci_fixed();
    
    // Save functions
    void save();
    void save_fixed();

    // Member variables
    hdr_isp::EigenImage3C eigen_img_;
    hdr_isp::EigenImage3CFixed eigen_img_fixed_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node params_;
    bool is_enable_;
    bool is_save_;
    bool is_debug_;
    float strength_;           // CLAHE clip limit
    int window_size_;          // CLAHE tile size
    int output_bit_depth_;
    hdr_isp::FixedPointConfig fp_config_;
    bool use_fixed_point_;
    bool use_fixed_input_;
}; 
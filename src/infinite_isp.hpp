#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "common/eigen_utils.hpp"
#include "common/fixed_point_utils.hpp"

// Forward declarations of module classes
class Crop;
class DeadPixelCorrection;
class BlackLevelCorrection;
class PiecewiseCurve;
class OECF;
class DigitalGain;
class LensShadingCorrection;
class BayerNoiseReduction;
class AutoWhiteBalance;
class WhiteBalance;
class HDRDurandToneMapping;
class Demosaic;
class ColorCorrectionMatrix;
class GammaCorrection;
class AutoExposure;
class ColorSpaceConversion;
class LDCI;
class Sharpening;
class NoiseReduction2D;
class RGBConversion;
class Scale;
class YUVConvFormat;

class InfiniteISP {
public:
    InfiniteISP(const std::string& data_path, const std::string& config_path, cv::Mat* memory_mapped_data = nullptr);
    ~InfiniteISP() = default;

    void execute(const std::string& img_path = "", bool save_intermediate = false);
    void update_sensor_info(const std::vector<int>& sensor_info, bool update_blc_wb = false);

private:
    // Member variables
    std::string data_path_;
    std::string config_path_;
    cv::Mat* memory_mapped_data_;
    std::unordered_map<std::string, std::shared_ptr<void>> module_instances_;
    
    // Configuration parameters
    YAML::Node config_;
    hdr_isp::FixedPointConfig fp_config_;
    std::string platform_;
    std::string raw_file_;
    bool render_3a_;
    
    // Sensor information
    struct SensorInfo {
        int width;
        int height;
        int bit_depth;
        std::string bayer_pattern;
    } sensor_info_;

    // Raw data storage - changed from cv::Mat to EigenImageU32
    hdr_isp::EigenImageU32 raw_;

    // Module parameters
    YAML::Node parm_dpc_;
    YAML::Node parm_cmpd_;
    YAML::Node parm_dga_;
    YAML::Node parm_lsc_;
    YAML::Node parm_bnr_;
    YAML::Node parm_blc_;
    YAML::Node parm_oec_;
    YAML::Node parm_wbc_;
    YAML::Node parm_awb_;
    YAML::Node parm_dem_;
    YAML::Node parm_ae_;
    YAML::Node parm_ccm_;
    YAML::Node parm_gmc_;
    YAML::Node parm_durand_;
    YAML::Node parm_csc_;
    YAML::Node parm_cse_;
    YAML::Node parm_ldci_;
    YAML::Node parm_sha_;
    YAML::Node parm_2dn_;
    YAML::Node parm_rgb_;
    YAML::Node parm_sca_;
    YAML::Node parm_cro_;
    YAML::Node parm_yuv_;

    // 3A feedback variables
    float dga_current_gain_;
    int ae_feedback_;
    std::vector<float> awb_gains_;

    // File names
    std::string in_file_;
    std::string out_file_;

    // Private methods
    void load_config(const std::string& config_path);
    void load_raw();
    void handle_non_byte_aligned_bit_depth();
    hdr_isp::EigenImageU32 run_pipeline(bool visualize_output, bool save_intermediate);
    void load_3a_statistics(bool awb_on, bool ae_on);
    hdr_isp::EigenImageU32 execute_with_3a_statistics(bool save_intermediate);
}; 
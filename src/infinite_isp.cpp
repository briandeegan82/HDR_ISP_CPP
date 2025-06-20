#include "infinite_isp.hpp"
#include "modules/demosaic/demosaic.hpp"
#include "modules/gamma_correction/gamma_correction.hpp"
#include "modules/auto_exposure/auto_exposure.hpp"
#include "modules/auto_white_balance/auto_white_balance.hpp"
#include "modules/bayer_noise_reduction/bayer_noise_reduction.hpp"
#include "modules/black_level_correction/black_level_correction.hpp"
#include "modules/color_correction_matrix/color_correction_matrix.hpp"
#include "modules/color_space_conversion/color_space_conversion.hpp"
#include "modules/crop/crop.hpp"
#include "modules/dead_pixel_correction/dead_pixel_correction.hpp"
#include "modules/digital_gain/digital_gain.hpp"
#include "modules/hdr_durand/hdr_durand.hpp"
#include "modules/ldci/ldci.hpp"
#include "modules/lens_shading_correction/lens_shading_correction.hpp"
#include "modules/2d_noise_reduction/2d_noise_reduction.hpp"
#include "modules/oecf/oecf.hpp"
#include "modules/pwc_generation/pwc_generation.hpp"
#include "modules/rgb_conversion/rgb_conversion.hpp"
#include "modules/scale/scale.hpp"
#include "modules/sharpen/sharpen.hpp"
#include "modules/white_balance/white_balance.hpp"
#include "modules/yuv_conv_format/yuv_conv_format.hpp"
#include <filesystem>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

InfiniteISP::InfiniteISP(const std::string& data_path, const std::string& config_path, cv::Mat* memory_mapped_data)
    : data_path_(data_path)
    , config_path_(config_path)
    , memory_mapped_data_(memory_mapped_data)
    , dga_current_gain_(0.0f)
    , ae_feedback_(0) {
    load_config(config_path);
}

void InfiniteISP::load_config(const std::string& config_path) {
    try {
        config_ = YAML::LoadFile(config_path);
        // Extract workspace info
        raw_file_ = config_["platform"]["filename"].as<std::string>();
        render_3a_ = config_["platform"]["render_3a"].as<bool>();
        // Extract basic sensor info
        sensor_info_.width = config_["sensor_info"]["width"].as<int>();
        sensor_info_.height = config_["sensor_info"]["height"].as<int>();
        sensor_info_.bit_depth = config_["sensor_info"]["bit_depth"].as<int>();
        sensor_info_.bayer_pattern = config_["sensor_info"]["bayer_pattern"].as<std::string>();
        // Get ISP module parameters
        parm_dpc_ = config_["dead_pixel_correction"];
        parm_cmpd_ = config_["companding"];
        parm_dga_ = config_["digital_gain"];
        parm_lsc_ = config_["lens_shading_correction"];
        parm_bnr_ = config_["bayer_noise_reduction"];
        parm_blc_ = config_["black_level_correction"];
        parm_oec_ = config_["oecf"];
        parm_wbc_ = config_["white_balance"];
        parm_awb_ = config_["auto_white_balance"];
        parm_dem_ = config_["demosaic"];
        parm_ae_ = config_["auto_exposure"];
        parm_ccm_ = config_["color_correction_matrix"];
        parm_gmc_ = config_["gamma_correction"];
        parm_durand_ = config_["hdr_durand"];
        parm_csc_ = config_["color_space_conversion"];
        parm_cse_ = config_["color_saturation_enhancement"];
        parm_ldci_ = config_["ldci"];
        parm_sha_ = config_["sharpen"];
        parm_2dn_ = config_["noise_reduction_2d"];
        parm_rgb_ = config_["rgb_conversion"];
        parm_sca_ = config_["scale"];
        parm_cro_ = config_["crop"];
        parm_yuv_ = config_["yuv_conversion_format"];
        config_["platform"]["rgb_output"] = parm_rgb_["is_enable"].as<bool>();
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error loading config: " + std::string(e.what()));
    }
}

void InfiniteISP::load_raw() {
    std::cout << "load_raw() started..." << std::endl;
    
    fs::path path_object = fs::path(data_path_) / raw_file_;
    std::string raw_path = path_object.string();
    std::cout << "Raw file path: " << raw_path << std::endl;
    
    in_file_ = path_object.stem().string();
    out_file_ = "Out_" + in_file_;

    config_["platform"]["in_file"] = in_file_;
    config_["platform"]["out_file"] = out_file_;

    std::cout << "Checking if file exists..." << std::endl;
    if (!fs::exists(path_object)) {
        std::cerr << "Error: Input file not found: " << raw_path << std::endl;
        throw std::runtime_error("Input file not found: " + raw_path);
    }
    std::cout << "File exists!" << std::endl;

    // Calculate file size to determine if memory mapping should be used
    size_t file_size = fs::file_size(path_object);
    std::cout << "File size: " << file_size << " bytes" << std::endl;
    bool use_mmap = file_size > 100 * 1024 * 1024;  // Use mmap for files > 100MB

    if (memory_mapped_data_ != nullptr) {
        std::cout << "Using memory mapped data..." << std::endl;
        raw_ = *memory_mapped_data_;
    }
    else if (path_object.extension() == ".raw") {
        std::cout << "Processing .raw file..." << std::endl;
        if (use_mmap) {
            // TODO: Implement memory mapping for raw files
            throw std::runtime_error("Memory mapping for raw files not implemented yet");
        }
        else {
            // Direct loading for smaller files
            std::cout << "Opening file for reading..." << std::endl;
            std::ifstream file(raw_path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open raw file: " + raw_path);
            }
            std::cout << "File opened successfully!" << std::endl;

            // Calculate expected file size
            int bytes_per_pixel = (sensor_info_.bit_depth + 7) / 8;
            size_t expected_size = sensor_info_.width * sensor_info_.height * bytes_per_pixel;
            std::cout << "Expected file size: " << expected_size << " bytes" << std::endl;
            std::cout << "Sensor info - width: " << sensor_info_.width << ", height: " << sensor_info_.height << ", bit_depth: " << sensor_info_.bit_depth << std::endl;
            
            // Determine appropriate OpenCV type based on bit depth
            int cv_type;
            if (sensor_info_.bit_depth <= 8) {
                cv_type = CV_8UC1;
            } else if (sensor_info_.bit_depth <= 16) {
                cv_type = CV_16UC1;
            } else if (sensor_info_.bit_depth <= 24) {
                cv_type = CV_32SC1;  // Use 32-bit signed for HDR images (OpenCV doesn't have CV_32UC1)
            } else {
                throw std::runtime_error("Unsupported bit depth: " + std::to_string(sensor_info_.bit_depth));
            }
            std::cout << "OpenCV type: " << cv_type << std::endl;
            
            // Create matrix to hold raw data
            std::cout << "Creating OpenCV matrix..." << std::endl;
            raw_ = cv::Mat(sensor_info_.height, sensor_info_.width, cv_type);
            std::cout << "Matrix created successfully!" << std::endl;
            
            // Read raw data
            std::cout << "Reading raw data..." << std::endl;
            file.read(reinterpret_cast<char*>(raw_.data), expected_size);
            
            if (!file) {
                throw std::runtime_error("Error reading raw file: " + raw_path);
            }
            std::cout << "Raw data read successfully!" << std::endl;
            
            // For bit depths that don't align with byte boundaries, we need to handle bit shifting
            if (sensor_info_.bit_depth > 8 && sensor_info_.bit_depth % 8 != 0) {
                std::cout << "Handling non-byte-aligned bit depth..." << std::endl;
                // Handle non-byte-aligned bit depths (e.g., 10-bit, 12-bit, 14-bit)
                handle_non_byte_aligned_bit_depth();
            }
        }
    }
    else if (path_object.extension() == ".tiff") {
        std::cout << "Processing .tiff file..." << std::endl;
        if (use_mmap) {
            // TODO: Implement memory mapping for tiff files
            throw std::runtime_error("Memory mapping for tiff files not implemented yet");
        }
        else {
            raw_ = cv::imread(raw_path, cv::IMREAD_UNCHANGED);
            if (raw_.channels() == 3) {
                std::vector<cv::Mat> channels;
                cv::split(raw_, channels);
                raw_ = channels[0];
            }
        }
    }
    else {
        std::cout << "Processing other format file..." << std::endl;
        // For other formats, use OpenCV
        raw_ = cv::imread(raw_path, cv::IMREAD_UNCHANGED);
    }

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Loading RAW Image Done......" << std::endl;
    std::cout << "Filename: " << raw_file_ << std::endl;
    std::cout << "Image size: " << raw_.cols << "x" << raw_.rows << std::endl;
    std::cout << "Image type: " << raw_.type() << " (CV_8U=" << CV_8U << ", CV_16U=" << CV_16U << ", CV_32S=" << CV_32S << ", CV_32F=" << CV_32F << ")" << std::endl;
    std::cout << "Image empty: " << (raw_.empty() ? "true" : "false") << std::endl;
    std::cout << "Image channels: " << raw_.channels() << std::endl;
    std::cout << "Bit depth: " << sensor_info_.bit_depth << " bits" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
}

void InfiniteISP::handle_non_byte_aligned_bit_depth() {
    // Handle bit depths that don't align with byte boundaries
    // This is needed for sensors with 10-bit, 12-bit, 14-bit, etc.
    
    if (sensor_info_.bit_depth == 10) {
        // Convert from packed 10-bit to 16-bit
        cv::Mat temp = raw_.clone();
        raw_ = cv::Mat(sensor_info_.height, sensor_info_.width, CV_16UC1);
        
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.cols; j++) {
                int32_t packed_value = temp.at<int32_t>(i, j);
                uint16_t unpacked_value = (packed_value & 0x3FF); // Extract 10 bits
                raw_.at<uint16_t>(i, j) = unpacked_value;
            }
        }
    }
    else if (sensor_info_.bit_depth == 12) {
        // Convert from packed 12-bit to 16-bit
        cv::Mat temp = raw_.clone();
        raw_ = cv::Mat(sensor_info_.height, sensor_info_.width, CV_16UC1);
        
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.cols; j++) {
                uint16_t packed_value = temp.at<uint16_t>(i, j);
                uint16_t unpacked_value = packed_value & 0x0FFF; // Extract 12 bits
                raw_.at<uint16_t>(i, j) = unpacked_value;
            }
        }
    }
    else if (sensor_info_.bit_depth == 14) {
        // Convert from packed 14-bit to 16-bit
        cv::Mat temp = raw_.clone();
        raw_ = cv::Mat(sensor_info_.height, sensor_info_.width, CV_16UC1);
        
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.cols; j++) {
                int32_t packed_value = temp.at<int32_t>(i, j);
                uint16_t unpacked_value = (packed_value & 0x3FFF); // Extract 14 bits
                raw_.at<uint16_t>(i, j) = unpacked_value;
            }
        }
    }
    // Add more bit depth handling as needed
}

cv::Mat InfiniteISP::run_pipeline(bool visualize_output, bool save_intermediate) {
    cv::Mat img = raw_.clone();

    // Create output directory for intermediate images if needed
    fs::path intermediate_dir;
    if (save_intermediate) {
        intermediate_dir = fs::path(PROJECT_ROOT_DIR) / "out_frames" / "intermediate";
        fs::create_directories(intermediate_dir);
    }

    // =====================================================================
    // 1. Cropping
    std::cout << "Cropping" << std::endl;
    if (parm_cro_["is_enable"].as<bool>()) {
        Crop crop(img, config_["platform"], config_["sensor_info"], parm_cro_);
        img = crop.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "crop.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 2. Dead pixels correction
    std::cout << "Dead pixels correction" << std::endl;
    if (parm_dpc_["is_enable"].as<bool>()) {
        DeadPixelCorrection dpc(img, config_["platform"], config_["sensor_info"], parm_dpc_);
        img = dpc.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "dead_pixel_correction.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 3. Black level correction
    std::cout << "Black level correction" << std::endl;
    if (parm_blc_["is_enable"].as<bool>()) {
        BlackLevelCorrection blc(img, config_["sensor_info"], parm_blc_);
        img = blc.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "black_level_correction.png";
            // Convert to 8-bit before saving
            cv::Mat save_img;
            img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 4. Decompanding (Piecewise curve)
    std::cout << "Decompanding (Piecewise curve)" << std::endl;
    if (parm_cmpd_["is_enable"].as<bool>()) {
        PiecewiseCurve pwc(img, config_["platform"], config_["sensor_info"], parm_cmpd_);
        img = pwc.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "piecewise_curve.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 5. OECF
    std::cout << "OECF" << std::endl;
    if (parm_oec_["is_enable"].as<bool>()) {
        OECF oecf(img, config_["platform"], config_["sensor_info"], parm_oec_);
        img = oecf.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "oecf.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 6. Digital Gain
    std::cout << "Digital Gain" << std::endl;
      
    try {
        DigitalGain dga(img, config_["platform"], config_["sensor_info"], parm_dga_);
        auto [result_img, gain] = dga.execute();
        img = result_img;
        dga_current_gain_ = gain;
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "digital_gain.png";
            cv::imwrite(output_path.string(), img);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in DigitalGain: " << e.what() << std::endl;
        throw;
    }
    

    // =====================================================================
    // 7. Lens shading correction
    std::cout << "Lens shading correction" << std::endl;
    if (parm_lsc_["is_enable"].as<bool>()) {
        LensShadingCorrection lsc(img, config_["platform"], config_["sensor_info"], parm_lsc_);
        img = lsc.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "lens_shading_correction.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 8. Bayer noise reduction
    std::cout << "Bayer noise reduction" << std::endl;
    if (parm_bnr_["is_enable"].as<bool>()) {
        BayerNoiseReduction bnr(img, config_["sensor_info"], parm_bnr_);
        img = bnr.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "bayer_noise_reduction.png";
            // Convert to 8-bit before saving
            cv::Mat save_img;
            img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 9. Auto White Balance
    std::cout << "Auto White Balance" << std::endl;
    if (parm_awb_["is_enable"].as<bool>()) {
        std::cout << "Applying auto white balance..." << std::endl;
        AutoWhiteBalance awb(img, config_["sensor_info"], parm_awb_);
        auto [rgain, bgain] = awb.execute();
        awb_gains_ = {static_cast<float>(rgain), static_cast<float>(bgain)};
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "auto_white_balance.png";
            cv::Mat save_img;
            img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 10. White balancing
    std::cout << "White balancing" << std::endl;
    if (parm_wbc_["is_enable"].as<bool>()) {
        std::cout << "Applying white balance..." << std::endl;
        WhiteBalance wb(img, config_["platform"], config_["sensor_info"], parm_wbc_);
        img = wb.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "white_balance.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 11. HDR tone mapping
    std::cout << "HDR tone mapping" << std::endl;
    if (parm_durand_["is_enable"].as<bool>()) {
        HDRDurandToneMapping hdr(img, config_["platform"], config_["sensor_info"], parm_durand_);
        img = hdr.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "hdr_tone_mapping.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 12. CFA demosaicing
    std::cout << "CFA demosaicing" << std::endl;
    if (parm_dem_["is_enable"].as<bool>()) {
        std::cout << "Applying demosaic..." << std::endl;
        DemosaicAlgorithm algorithm = parm_dem_["algorithm"].as<std::string>() == "opencv" ? 
            DemosaicAlgorithm::OPENCV : DemosaicAlgorithm::MALVAR;
        Demosaic demosaic(img, sensor_info_.bayer_pattern, sensor_info_.bit_depth, save_intermediate, algorithm);
        img = demosaic.execute();
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "demosaic.png";
            cv::Mat save_img;
            img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 13. Color correction matrix
    std::cout << "Color correction matrix" << std::endl;
    if (parm_ccm_["is_enable"].as<bool>()) {
        ColorCorrectionMatrix ccm(img, config_["sensor_info"], parm_ccm_);
        img = ccm.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "color_correction_matrix.png";
            cv::Mat save_img;
            img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 14. Gamma
    std::cout << "Gamma" << std::endl;
    if (parm_gmc_["is_enable"].as<bool>()) {
        std::cout << "Applying gamma correction..." << std::endl;
        GammaCorrection gamma(img, config_["platform"], config_["sensor_info"], parm_gmc_);
        img = gamma.execute();
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "gamma_correction.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 15. Auto-Exposure
    std::cout << "Auto-Exposure" << std::endl;
    if (parm_ae_["is_enable"].as<bool>()) {
        std::cout << "Applying auto exposure..." << std::endl;
        AutoExposure ae(img, config_["sensor_info"], parm_ae_);
        ae_feedback_ = ae.execute();
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "auto_exposure.png";
            cv::Mat save_img;
            img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 16. Color space conversion
    std::cout << "Color space conversion" << std::endl;
    if (parm_csc_["is_enable"].as<bool>()) {
        ColorSpaceConversion csc(img, config_["sensor_info"], parm_csc_, parm_cse_);
        img = csc.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "color_space_conversion.png";
            cv::Mat save_img;
            img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 17. Local Dynamic Contrast Improvement
    std::cout << "Local Dynamic Contrast Improvement" << std::endl;
    if (parm_ldci_["is_enable"].as<bool>()) {
        LDCI ldci(img, config_["platform"], config_["sensor_info"], parm_ldci_);
        img = ldci.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "ldci.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 18. Sharpening
    std::cout << "Sharpening" << std::endl;
    if (parm_sha_["is_enable"].as<bool>()) {
        std::cout << "Applying sharpening..." << std::endl;
        
        try {
            // Get conv_std with default value if not defined
            std::string conv_std_value = "709"; // Default value
            if (parm_csc_.IsDefined() && parm_csc_["conv_std"].IsDefined()) {
                conv_std_value = parm_csc_["conv_std"].as<std::string>();
            }
            
            Sharpen sharpen(img, config_["platform"], config_["sensor_info"], parm_sha_, conv_std_value);
            
            img = sharpen.execute();
            
        } catch (const std::exception& e) {
            std::cerr << "Exception during Sharpen creation/execution: " << e.what() << std::endl;
            throw;
        }
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "sharpen.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 19. 2d noise reduction
    std::cout << "2d noise reduction" << std::endl;
    
    if (parm_2dn_["is_enable"].as<bool>()) {
        NoiseReduction2D nr2d(img, config_["platform"], config_["sensor_info"], parm_2dn_);
        img = nr2d.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "2d_noise_reduction.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 20. RGB conversion
    std::cout << "RGB conversion" << std::endl;
    if (parm_rgb_["is_enable"].as<bool>()) {
        RGBConversion rgb_conv(img, config_["platform"], config_["sensor_info"], parm_rgb_, parm_csc_);
        img = rgb_conv.execute();
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "rgb_conversion.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 21. Scaling
    std::cout << "Scaling" << std::endl;
    if (parm_sca_["is_enable"].as<bool>()) {
        std::cout << "Applying scaling..." << std::endl;
        
        try {
            // Get conv_std with default value if not defined
            int conv_std_value = 709; // Default value
            if (parm_csc_.IsDefined() && parm_csc_["conv_std"].IsDefined()) {
                conv_std_value = parm_csc_["conv_std"].as<int>();
            }
            
            Scale scale(img, config_["platform"], config_["sensor_info"], parm_sca_, conv_std_value);
            img = scale.execute();
            
        } catch (const std::exception& e) {
            std::cerr << "Exception during Scale creation/execution: " << e.what() << std::endl;
            throw;
        }
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "scale.png";
            cv::imwrite(output_path.string(), img);
        }
    }

    // =====================================================================
    // 22. YUV saving format 444, 422 etc
    std::cout << "YUV saving format 444, 422 etc" << std::endl;
    if (parm_yuv_["is_enable"].as<bool>()) {
        std::cout << "Applying YUV conversion format..." << std::endl;
        YUVConvFormat yuv_conv(img, config_["platform"], config_["sensor_info"], parm_yuv_);
        img = yuv_conv.execute();
        if (save_intermediate) {
            std::string output_path = "yuv_conversion_format";
            cv::imwrite(output_path + ".png", img);
        }
    }

    return img;
}

void InfiniteISP::load_3a_statistics(bool awb_on, bool ae_on) {
    if (awb_on && parm_dga_["is_auto"].as<bool>() && parm_awb_["is_enable"].as<bool>()) {
        parm_wbc_["r_gain"] = config_["white_balance"]["r_gain"] = awb_gains_[0];
        parm_wbc_["b_gain"] = config_["white_balance"]["b_gain"] = awb_gains_[1];
    }

    if (ae_on && parm_dga_["is_auto"].as<bool>() && parm_ae_["is_enable"].as<bool>()) {
        parm_dga_["ae_feedback"] = config_["digital_gain"]["ae_feedback"] = ae_feedback_;
        parm_dga_["current_gain"] = config_["digital_gain"]["current_gain"] = dga_current_gain_;
    }
}

cv::Mat InfiniteISP::execute_with_3a_statistics(bool save_intermediate) {
    int max_dg = parm_dga_["gain_array"].size();

    run_pipeline(false, save_intermediate);
    load_3a_statistics();

    while (!(ae_feedback_ == 0 ||
            (ae_feedback_ == -1 && dga_current_gain_ == max_dg) ||
            (ae_feedback_ == 1 && dga_current_gain_ == 0) ||
            ae_feedback_ == -1)) {
        run_pipeline(false, save_intermediate);
        load_3a_statistics();
    }

    return run_pipeline(true, save_intermediate);
}

void InfiniteISP::execute(const std::string& img_path, bool save_intermediate) {
    std::cout << "Starting execute() function..." << std::endl;
    
    if (!img_path.empty()) {
        raw_file_ = img_path;
        config_["platform"]["filename"] = raw_file_;
        std::cout << "Set raw_file_ to: " << raw_file_ << std::endl;
    }

    std::cout << "About to call load_raw()..." << std::endl;
    load_raw();
    std::cout << "load_raw() completed successfully" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Generate timestamp for output filename
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "_%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    std::cout << "About to run pipeline..." << std::endl;
    cv::Mat final_img;
    if (!render_3a_) {
        final_img = run_pipeline(true, save_intermediate);
    }
    else {
        final_img = execute_with_3a_statistics(save_intermediate);
    }

    // Save final output
    fs::path output_dir = fs::path(PROJECT_ROOT_DIR) / "out_frames";
    fs::create_directories(output_dir);
    fs::path output_path = output_dir / (out_file_ + timestamp + ".png");
    cv::imwrite(output_path.string(), final_img);
    std::cout << "Final output saved to: " << output_path << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nPipeline Elapsed Time: " << elapsed.count() << "s" << std::endl;
}

void InfiniteISP::update_sensor_info(const std::vector<int>& sensor_info, bool update_blc_wb) {
    config_["sensor_info"]["width"] = sensor_info[0];
    config_["sensor_info"]["height"] = sensor_info[1];
    config_["sensor_info"]["bit_depth"] = sensor_info[2];
    config_["sensor_info"]["bayer_pattern"] = std::to_string(sensor_info[3]);
    
    sensor_info_.width = config_["sensor_info"]["width"].as<int>();
    sensor_info_.height = config_["sensor_info"]["height"].as<int>();
    sensor_info_.bit_depth = config_["sensor_info"]["bit_depth"].as<int>();
    sensor_info_.bayer_pattern = config_["sensor_info"]["bayer_pattern"].as<std::string>();

    if (update_blc_wb) {
        // Update black level correction parameters
        config_["black_level_correction"]["r_offset"] = sensor_info[4];
        config_["black_level_correction"]["gr_offset"] = sensor_info[5];
        config_["black_level_correction"]["gb_offset"] = sensor_info[6];
        config_["black_level_correction"]["b_offset"] = sensor_info[7];

        parm_blc_["r_offset"] = config_["black_level_correction"]["r_offset"].as<int>();
        parm_blc_["gr_offset"] = config_["black_level_correction"]["gr_offset"].as<int>();
        parm_blc_["gb_offset"] = config_["black_level_correction"]["gb_offset"].as<int>();
        parm_blc_["b_offset"] = config_["black_level_correction"]["b_offset"].as<int>();

        // Update saturation levels
        config_["black_level_correction"]["r_sat"] = sensor_info[8];
        config_["black_level_correction"]["gr_sat"] = sensor_info[8];
        config_["black_level_correction"]["gb_sat"] = sensor_info[8];
        config_["black_level_correction"]["b_sat"] = sensor_info[8];

        parm_blc_["r_sat"] = config_["black_level_correction"]["r_sat"].as<int>();
        parm_blc_["gr_sat"] = config_["black_level_correction"]["gr_sat"].as<int>();
        parm_blc_["gb_sat"] = config_["black_level_correction"]["gb_sat"].as<int>();
        parm_blc_["b_sat"] = config_["black_level_correction"]["b_sat"].as<int>();

        // Update white balance gains
        config_["white_balance"]["r_gain"] = sensor_info[9];
        config_["white_balance"]["b_gain"] = sensor_info[10];

        parm_wbc_["r_gain"] = config_["white_balance"]["r_gain"].as<int>();
        parm_wbc_["b_gain"] = config_["white_balance"]["b_gain"].as<int>();
    }
} 
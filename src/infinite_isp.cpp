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
#include "common/fixed_point_utils.hpp"
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
    , fp_config_(YAML::Node())  // Initialize with empty YAML node
    , dga_current_gain_(0.0f)
    , ae_feedback_(0) {
    load_config(config_path);
}

void InfiniteISP::load_config(const std::string& config_path) {
    try {
        config_ = YAML::LoadFile(config_path);
        
        // Initialize fixed-point configuration
        fp_config_ = hdr_isp::FixedPointConfig(config_);
        
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
        // Convert memory mapped OpenCV data to EigenImageU32
        raw_ = hdr_isp::EigenImageU32::fromOpenCV(*memory_mapped_data_);
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
            
            // Create Eigen matrix directly
            std::cout << "Creating Eigen matrix..." << std::endl;
            raw_ = hdr_isp::EigenImageU32(sensor_info_.height, sensor_info_.width);
            std::cout << "Matrix created successfully!" << std::endl;
            
            // Read raw data directly into Eigen matrix
            std::cout << "Reading raw data..." << std::endl;
            
            // Determine appropriate data type based on bit depth
            if (sensor_info_.bit_depth <= 8) {
                // Read as 8-bit and convert to uint32_t
                std::vector<uint8_t> buffer(sensor_info_.width * sensor_info_.height);
                file.read(reinterpret_cast<char*>(buffer.data()), expected_size);
                
                for (int i = 0; i < sensor_info_.height; ++i) {
                    for (int j = 0; j < sensor_info_.width; ++j) {
                        raw_.data()(i, j) = static_cast<uint32_t>(buffer[i * sensor_info_.width + j]);
                    }
                }
            } else if (sensor_info_.bit_depth <= 16) {
                // Read as 16-bit and convert to uint32_t
                std::vector<uint16_t> buffer(sensor_info_.width * sensor_info_.height);
                file.read(reinterpret_cast<char*>(buffer.data()), expected_size);
                
                for (int i = 0; i < sensor_info_.height; ++i) {
                    for (int j = 0; j < sensor_info_.width; ++j) {
                        raw_.data()(i, j) = static_cast<uint32_t>(buffer[i * sensor_info_.width + j]);
                    }
                }
            } else if (sensor_info_.bit_depth <= 24) {
                // Read as 32-bit unsigned for HDR images
                std::vector<uint32_t> buffer(sensor_info_.width * sensor_info_.height);
                file.read(reinterpret_cast<char*>(buffer.data()), expected_size);
                
                for (int i = 0; i < sensor_info_.height; ++i) {
                    for (int j = 0; j < sensor_info_.width; ++j) {
                        raw_.data()(i, j) = buffer[i * sensor_info_.width + j];
                    }
                }
            } else {
                throw std::runtime_error("Unsupported bit depth: " + std::to_string(sensor_info_.bit_depth));
            }
            
            if (!file) {
                throw std::runtime_error("Error reading raw file: " + raw_path);
            }
            std::cout << "Raw data read successfully!" << std::endl;
            
            // Check if data is packed or unpacked by comparing file sizes
            // For bit depths that don't align with byte boundaries, we need to determine if data is packed
            if (sensor_info_.bit_depth > 8 && sensor_info_.bit_depth % 8 != 0) {
                // Calculate expected sizes for both packed and unpacked formats
                size_t unpacked_size = sensor_info_.width * sensor_info_.height * ((sensor_info_.bit_depth + 7) / 8);
                size_t packed_size = sensor_info_.width * sensor_info_.height * sensor_info_.bit_depth / 8;
                
                std::cout << "Bit depth " << sensor_info_.bit_depth << " doesn't align with byte boundaries" << std::endl;
                std::cout << "Expected unpacked size: " << unpacked_size << " bytes" << std::endl;
                std::cout << "Expected packed size: " << packed_size << " bytes" << std::endl;
                std::cout << "Actual file size: " << file_size << " bytes" << std::endl;
                
                // Determine if data is packed or unpacked based on file size
                if (file_size == unpacked_size) {
                    std::cout << "Data is unpacked (stored as " << ((sensor_info_.bit_depth + 7) / 8) * 8 << "-bit per pixel)" << std::endl;
                    // No need to handle unpacking - data is already in correct format
                } else if (file_size == packed_size) {
                    std::cout << "Data is packed (stored as " << sensor_info_.bit_depth << "-bit per pixel)" << std::endl;
                    // Handle unpacking for packed data
                    handle_non_byte_aligned_bit_depth();
                } else {
                    std::cout << "Warning: File size doesn't match expected packed or unpacked size" << std::endl;
                    std::cout << "Assuming unpacked format and proceeding..." << std::endl;
                }
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
            // Load TIFF using OpenCV first, then convert to Eigen
            cv::Mat temp_img = cv::imread(raw_path, cv::IMREAD_UNCHANGED);
            if (temp_img.channels() == 3) {
                std::vector<cv::Mat> channels;
                cv::split(temp_img, channels);
                temp_img = channels[0];
            }
            raw_ = hdr_isp::EigenImageU32::fromOpenCV(temp_img);
        }
    }
    else {
        std::cout << "Processing other format file..." << std::endl;
        // For other formats, use OpenCV first, then convert to Eigen
        cv::Mat temp_img = cv::imread(raw_path, cv::IMREAD_UNCHANGED);
        raw_ = hdr_isp::EigenImageU32::fromOpenCV(temp_img);
    }

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Loading RAW Image Done......" << std::endl;
    std::cout << "Filename: " << raw_file_ << std::endl;
    std::cout << "Image size: " << raw_.cols() << "x" << raw_.rows() << std::endl;
    std::cout << "Image type: EigenImageU32 (uint32_t)" << std::endl;
    std::cout << "Image empty: " << (raw_.size() == 0 ? "true" : "false") << std::endl;
    std::cout << "Bit depth: " << sensor_info_.bit_depth << " bits" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
}

void InfiniteISP::handle_non_byte_aligned_bit_depth() {
    // Handle bit depths that don't align with byte boundaries
    // This is needed for sensors with 10-bit, 12-bit, 14-bit, etc.
    
    if (sensor_info_.bit_depth == 10) {
        // Convert from packed 10-bit to 16-bit
        hdr_isp::EigenImageU32 temp = raw_.clone();
        raw_ = hdr_isp::EigenImageU32(sensor_info_.height, sensor_info_.width);
        
        for (int i = 0; i < temp.rows(); i++) {
            for (int j = 0; j < temp.cols(); j++) {
                uint32_t packed_value = temp.data()(i, j);
                uint32_t unpacked_value = (packed_value & 0x3FF); // Extract 10 bits
                raw_.data()(i, j) = unpacked_value;
            }
        }
    }
    else if (sensor_info_.bit_depth == 12) {
        // Convert from packed 12-bit to 16-bit
        hdr_isp::EigenImageU32 temp = raw_.clone();
        raw_ = hdr_isp::EigenImageU32(sensor_info_.height, sensor_info_.width);
        
        for (int i = 0; i < temp.rows(); i++) {
            for (int j = 0; j < temp.cols(); j++) {
                uint32_t packed_value = temp.data()(i, j);
                uint32_t unpacked_value = packed_value & 0x0FFF; // Extract 12 bits
                raw_.data()(i, j) = unpacked_value;
            }
        }
    }
    else if (sensor_info_.bit_depth == 14) {
        // Convert from packed 14-bit to 16-bit
        hdr_isp::EigenImageU32 temp = raw_.clone();
        raw_ = hdr_isp::EigenImageU32(sensor_info_.height, sensor_info_.width);
        
        for (int i = 0; i < temp.rows(); i++) {
            for (int j = 0; j < temp.cols(); j++) {
                uint32_t packed_value = temp.data()(i, j);
                uint32_t unpacked_value = (packed_value & 0x3FFF); // Extract 14 bits
                raw_.data()(i, j) = unpacked_value;
            }
        }
    }
    // Add more bit depth handling as needed
}

hdr_isp::EigenImageU32 InfiniteISP::run_pipeline(bool visualize_output, bool save_intermediate) {
    // Start with Eigen data for all modules before demosaic
    hdr_isp::EigenImageU32 eigen_img = raw_.clone();

    // Create output directory for intermediate images if needed
    fs::path intermediate_dir;
    if (save_intermediate) {
        intermediate_dir = fs::path(PROJECT_ROOT_DIR) / "out_frames" / "intermediate";
        fs::create_directories(intermediate_dir);
    }

    // Debug: Print initial image statistics using Eigen
    uint32_t min_val = eigen_img.min();
    uint32_t max_val = eigen_img.max();
    float mean_val = eigen_img.mean();
    std::cout << "=== INITIAL IMAGE STATS ===" << std::endl;
    std::cout << "Type: EigenImageU32 (uint32_t)" << std::endl;
    std::cout << "Min: " << min_val << ", Mean: " << mean_val << ", Max: " << max_val << std::endl;
    std::cout << "Size: " << eigen_img.cols() << "x" << eigen_img.rows() << ", Channels: 1" << std::endl;
    std::cout << "==========================" << std::endl;

    // Variable to hold the final result after demosaic (will be Eigen for modules that need it)
    hdr_isp::EigenImage3C eigen_img_3c;
    hdr_isp::EigenImage3CFixed eigen_img_3c_fixed;
    
    // Variables for OpenCV conversions and debug output
    cv::Mat opencv_img;
    double min_val_cv, max_val_cv;
    cv::Scalar mean_val_cv;

    // =====================================================================
    // 1. Cropping
    std::cout << "Cropping" << std::endl;
    if (parm_cro_["is_enable"].as<bool>()) {
        // Use Eigen-based crop module directly
        Crop crop(eigen_img, config_["platform"], config_["sensor_info"], parm_cro_);
        eigen_img = crop.execute();
        
        // Debug: Print image statistics after cropping using Eigen
        uint32_t min_val = eigen_img.min();
        uint32_t max_val = eigen_img.max();
        float mean_val = eigen_img.mean();
        std::cout << "=== AFTER CROPPING ===" << std::endl;
        std::cout << "Type: EigenImageU32 (uint32_t)" << std::endl;
        std::cout << "Min: " << min_val << ", Mean: " << mean_val << ", Max: " << max_val << std::endl;
        std::cout << "Size: " << eigen_img.cols() << "x" << eigen_img.rows() << ", Channels: 1" << std::endl;
        std::cout << "=====================" << std::endl;
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "crop.png";
            // Convert to OpenCV for saving
            opencv_img = eigen_img.toOpenCV(CV_32S);
            // Debug: Print image statistics before saving
            cv::minMaxLoc(opencv_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(opencv_img);
            std::cout << "Crop - Mean: " << mean_val_cv[0] << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << opencv_img.type() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (opencv_img.type() == CV_32F) {
                opencv_img.convertTo(save_img, CV_8U, 255.0);
            } else if (opencv_img.type() == CV_16U) {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / 65535.0);
            } else {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 2. Dead pixels correction
    std::cout << "Dead pixels correction" << std::endl;
    if (parm_dpc_["is_enable"].as<bool>()) {
        // Use Eigen-based dead pixel correction module directly
        DeadPixelCorrection dpc(eigen_img, config_["platform"], config_["sensor_info"], parm_dpc_);
        eigen_img = dpc.execute_eigen();
        
        // Debug: Print image statistics after dead pixel correction using Eigen
        uint32_t min_val = eigen_img.min();
        uint32_t max_val = eigen_img.max();
        float mean_val = eigen_img.mean();
        std::cout << "=== AFTER DEAD PIXEL CORRECTION ===" << std::endl;
        std::cout << "Type: EigenImageU32 (uint32_t)" << std::endl;
        std::cout << "Min: " << min_val << ", Mean: " << mean_val << ", Max: " << max_val << std::endl;
        std::cout << "Size: " << eigen_img.cols() << "x" << eigen_img.rows() << ", Channels: 1" << std::endl;
        std::cout << "==================================" << std::endl;
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "dead_pixel_correction.png";
            // Convert to OpenCV for saving
            opencv_img = eigen_img.toOpenCV(CV_32S);
            // Debug: Print image statistics before saving
            cv::minMaxLoc(opencv_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(opencv_img);
            std::cout << "DeadPixel - Mean: " << mean_val_cv[0] << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << opencv_img.type() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (opencv_img.type() == CV_32F) {
                opencv_img.convertTo(save_img, CV_8U, 255.0);
            } else if (opencv_img.type() == CV_16U) {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / 65535.0);
            } else {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 3. Black level correction
    std::cout << "Black level correction" << std::endl;
    if (parm_blc_["is_enable"].as<bool>()) {
        // Use Eigen-based black level correction module directly
        BlackLevelCorrection blc(eigen_img, config_["sensor_info"], parm_blc_);
        eigen_img = blc.execute();
        
        // Debug: Print image statistics after black level correction using Eigen
        uint32_t min_val = eigen_img.min();
        uint32_t max_val = eigen_img.max();
        float mean_val = eigen_img.mean();
        std::cout << "=== AFTER BLACK LEVEL CORRECTION ===" << std::endl;
        std::cout << "Type: EigenImageU32 (uint32_t)" << std::endl;
        std::cout << "Min: " << min_val << ", Mean: " << mean_val << ", Max: " << max_val << std::endl;
        std::cout << "Size: " << eigen_img.cols() << "x" << eigen_img.rows() << ", Channels: 1" << std::endl;
        std::cout << "===================================" << std::endl;
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "black_level_correction.png";
            // Convert to OpenCV for saving
            opencv_img = eigen_img.toOpenCV(CV_32S);
            // Debug: Print image statistics before saving
            cv::minMaxLoc(opencv_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(opencv_img);
            std::cout << "BLC - Mean: " << mean_val_cv[0] << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << opencv_img.type() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (opencv_img.type() == CV_32F) {
                opencv_img.convertTo(save_img, CV_8U, 255.0);
            } else if (opencv_img.type() == CV_16U) {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / 65535.0);
            } else {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 4. Decompanding (Piecewise curve)
    std::cout << "Decompanding (Piecewise curve)" << std::endl;
    if (parm_cmpd_["is_enable"].as<bool>()) {
        // Use Eigen-based piecewise curve module directly
        PiecewiseCurve pwc(eigen_img, config_["platform"], config_["sensor_info"], parm_cmpd_);
        eigen_img = pwc.execute();
        
        // Debug: Print image statistics after decompanding using Eigen
        uint32_t min_val = eigen_img.min();
        uint32_t max_val = eigen_img.max();
        float mean_val = eigen_img.mean();
        std::cout << "=== AFTER DECOMPANDING ===" << std::endl;
        std::cout << "Type: EigenImageU32 (uint32_t)" << std::endl;
        std::cout << "Min: " << min_val << ", Mean: " << mean_val << ", Max: " << max_val << std::endl;
        std::cout << "Size: " << eigen_img.cols() << "x" << eigen_img.rows() << ", Channels: 1" << std::endl;
        std::cout << "=========================" << std::endl;
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "piecewise_curve.png";
            // Convert to OpenCV for saving
            opencv_img = eigen_img.toOpenCV(CV_32S);
            // Debug: Print image statistics before saving
            cv::minMaxLoc(opencv_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(opencv_img);
            std::cout << "PWC - Mean: " << mean_val_cv[0] << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << opencv_img.type() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (opencv_img.type() == CV_32F) {
                opencv_img.convertTo(save_img, CV_8U, 255.0);
            } else if (opencv_img.type() == CV_16U) {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / 65535.0);
            } else {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 5. OECF
    std::cout << "OECF" << std::endl;
    if (parm_oec_["is_enable"].as<bool>()) {
        // Use Eigen-based OECF module directly
        OECF oecf(eigen_img, config_["platform"], config_["sensor_info"], parm_oec_);
        eigen_img = oecf.execute();
        
        // Debug: Print image statistics after OECF using Eigen
        uint32_t min_val = eigen_img.min();
        uint32_t max_val = eigen_img.max();
        float mean_val = eigen_img.mean();
        std::cout << "=== AFTER OECF ===" << std::endl;
        std::cout << "Type: EigenImageU32 (uint32_t)" << std::endl;
        std::cout << "Min: " << min_val << ", Mean: " << mean_val << ", Max: " << max_val << std::endl;
        std::cout << "Size: " << eigen_img.cols() << "x" << eigen_img.rows() << ", Channels: 1" << std::endl;
        std::cout << "=================" << std::endl;
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "oecf.png";
            // Convert to OpenCV for saving
            opencv_img = eigen_img.toOpenCV(CV_32S);
            // Debug: Print image statistics before saving
            cv::minMaxLoc(opencv_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(opencv_img);
            std::cout << "OECF - Mean: " << mean_val_cv[0] << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << opencv_img.type() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (opencv_img.type() == CV_32F) {
                opencv_img.convertTo(save_img, CV_8U, 255.0);
            } else if (opencv_img.type() == CV_16U) {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / 65535.0);
            } else {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 6. Digital Gain
    std::cout << "Digital Gain" << std::endl;
      
    try {
        // Digital gain already uses Eigen
        DigitalGain dga(eigen_img, config_["platform"], config_["sensor_info"], parm_dga_);
        auto [result_eigen, gain] = dga.execute();
        eigen_img = result_eigen;
        dga_current_gain_ = gain;
        
        // Debug: Print image statistics after digital gain using Eigen
        uint32_t min_val = eigen_img.min();
        uint32_t max_val = eigen_img.max();
        float mean_val = eigen_img.mean();
        std::cout << "=== AFTER DIGITAL GAIN ===" << std::endl;
        std::cout << "Type: EigenImageU32 (uint32_t)" << std::endl;
        std::cout << "Min: " << min_val << ", Mean: " << mean_val << ", Max: " << max_val << std::endl;
        std::cout << "Size: " << eigen_img.cols() << "x" << eigen_img.rows() << ", Channels: 1" << std::endl;
        std::cout << "Applied gain: " << gain << std::endl;
        std::cout << "=========================" << std::endl;
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "digital_gain.png";
            // Convert to OpenCV for saving
            opencv_img = eigen_img.toOpenCV(CV_32S);
            // Debug: Print image statistics before saving
            cv::minMaxLoc(opencv_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(opencv_img);
            std::cout << "DigitalGain - Mean: " << mean_val_cv[0] << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << opencv_img.type() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (opencv_img.type() == CV_32F) {
                opencv_img.convertTo(save_img, CV_8U, 255.0);
            } else if (opencv_img.type() == CV_16U) {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / 65535.0);
            } else {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
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
        // Use Eigen-based lens shading correction module directly
        LensShadingCorrection lsc(eigen_img, config_["platform"], config_["sensor_info"], parm_lsc_);
        eigen_img = lsc.execute();
        
        // Debug: Print image statistics after lens shading correction using Eigen
        uint32_t min_val = eigen_img.min();
        uint32_t max_val = eigen_img.max();
        float mean_val = eigen_img.mean();
        std::cout << "=== AFTER LENS SHADING CORRECTION ===" << std::endl;
        std::cout << "Type: EigenImageU32 (uint32_t)" << std::endl;
        std::cout << "Min: " << min_val << ", Mean: " << mean_val << ", Max: " << max_val << std::endl;
        std::cout << "Size: " << eigen_img.cols() << "x" << eigen_img.rows() << ", Channels: 1" << std::endl;
        std::cout << "====================================" << std::endl;
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "lens_shading_correction.png";
            // Convert to OpenCV for saving
            opencv_img = eigen_img.toOpenCV(CV_32S);
            // Debug: Print image statistics before saving
            cv::minMaxLoc(opencv_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(opencv_img);
            std::cout << "LSC - Mean: " << mean_val_cv[0] << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << opencv_img.type() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (opencv_img.type() == CV_32F) {
                opencv_img.convertTo(save_img, CV_8U, 255.0);
            } else if (opencv_img.type() == CV_16U) {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / 65535.0);
            } else {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 8. Bayer noise reduction
    std::cout << "Bayer noise reduction" << std::endl;
    if (parm_bnr_["is_enable"].as<bool>()) {
        try {
            std::cout << "BNR - Starting bayer noise reduction..." << std::endl;
            
            // Use Eigen-based bayer noise reduction module directly
            BayerNoiseReduction bnr(eigen_img, config_["sensor_info"], parm_bnr_);
            eigen_img = bnr.execute();
            
            std::cout << "BNR - Bayer noise reduction completed successfully" << std::endl;
            
            // Debug: Print image statistics after bayer noise reduction using Eigen
            uint32_t min_val = eigen_img.min();
            uint32_t max_val = eigen_img.max();
            float mean_val = eigen_img.mean();
            std::cout << "=== AFTER BAYER NOISE REDUCTION ===" << std::endl;
            std::cout << "Type: EigenImageU32 (uint32_t)" << std::endl;
            std::cout << "Min: " << min_val << ", Mean: " << mean_val << ", Max: " << max_val << std::endl;
            std::cout << "Size: " << eigen_img.cols() << "x" << eigen_img.rows() << ", Channels: 1" << std::endl;
            std::cout << "==================================" << std::endl;
            
            if (save_intermediate) {
                fs::path output_path = intermediate_dir / "bayer_noise_reduction.png";
                // Convert to OpenCV for saving
                opencv_img = eigen_img.toOpenCV(CV_32S);
                // Debug: Print image statistics before saving
                cv::minMaxLoc(opencv_img, &min_val_cv, &max_val_cv);
                mean_val_cv = cv::mean(opencv_img);
                std::cout << "BNR - Mean: " << mean_val_cv[0] << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << opencv_img.type() << std::endl;
                
                // Convert to 8-bit for display
                cv::Mat save_img;
                if (opencv_img.type() == CV_32F) {
                    opencv_img.convertTo(save_img, CV_8U, 255.0);
                } else if (opencv_img.type() == CV_16U) {
                    opencv_img.convertTo(save_img, CV_8U, 255.0 / 65535.0);
                } else {
                    opencv_img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
                }
                cv::imwrite(output_path.string(), save_img);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "BNR - Exception in bayer noise reduction: " << e.what() << std::endl;
            std::cout << "BNR - Continuing with original image..." << std::endl;
        }
        catch (...) {
            std::cerr << "BNR - Unknown exception in bayer noise reduction" << std::endl;
            std::cout << "BNR - Continuing with original image..." << std::endl;
        }
    }

    // =====================================================================
    // 9. Auto White Balance
    std::cout << "Auto White Balance" << std::endl;
    if (parm_awb_["is_enable"].as<bool>()) {
        std::cout << "Applying auto white balance..." << std::endl;
        // Use Eigen-based auto white balance module directly
        AutoWhiteBalance awb(eigen_img, config_["sensor_info"], parm_awb_);
        auto [rgain, bgain] = awb.execute();
        awb_gains_ = {static_cast<float>(rgain), static_cast<float>(bgain)};
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "auto_white_balance.png";
            // Convert to OpenCV for saving
            opencv_img = eigen_img.toOpenCV(CV_32S);
            // Debug: Print image statistics before saving
            cv::minMaxLoc(opencv_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(opencv_img);
            std::cout << "AWB - Mean: " << mean_val_cv[0] << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << opencv_img.type() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (opencv_img.type() == CV_32F) {
                opencv_img.convertTo(save_img, CV_8U, 255.0);
            } else if (opencv_img.type() == CV_16U) {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / 65535.0);
            } else {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 10. White balancing
    std::cout << "White balancing" << std::endl;
    if (parm_wbc_["is_enable"].as<bool>()) {
        std::cout << "Applying white balance..." << std::endl;
        // Use Eigen-based white balance module directly
        WhiteBalance wb(eigen_img, config_["platform"], config_["sensor_info"], parm_wbc_);
        eigen_img = wb.execute();
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "white_balance.png";
            // Convert to OpenCV for saving
            opencv_img = eigen_img.toOpenCV(CV_32S);
            // Debug: Print image statistics before saving
            cv::minMaxLoc(opencv_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(opencv_img);
            std::cout << "WB - Mean: " << mean_val_cv[0] << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << opencv_img.type() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (opencv_img.type() == CV_32F) {
                opencv_img.convertTo(save_img, CV_8U, 255.0);
            } else if (opencv_img.type() == CV_16U) {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / 65535.0);
            } else {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 11. HDR tone mapping
    std::cout << "HDR tone mapping" << std::endl;
    if (parm_durand_["is_enable"].as<bool>()) {
        // Use Eigen-based HDR tone mapping module directly
        HDRDurandToneMapping hdr(eigen_img, config_["platform"], config_["sensor_info"], parm_durand_);
        eigen_img = hdr.execute_eigen();
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "hdr_tone_mapping.png";
            // Convert to OpenCV for saving
            opencv_img = eigen_img.toOpenCV(CV_32S);
            // Debug: Print image statistics before saving
            cv::minMaxLoc(opencv_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(opencv_img);
            std::cout << "HDR - Mean: " << mean_val_cv[0] << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << opencv_img.type() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (opencv_img.type() == CV_32F) {
                opencv_img.convertTo(save_img, CV_8U, 255.0);
            } else if (opencv_img.type() == CV_16U) {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / 65535.0);
            } else {
                opencv_img.convertTo(save_img, CV_8U, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 12. CFA demosaicing - Convert back to OpenCV for demosaic and rest of pipeline
    std::cout << "CFA demosaicing" << std::endl;
    if (parm_dem_["is_enable"].as<bool>()) {
        std::cout << "Applying demosaic..." << std::endl;
        
        // Check if fixed-point is enabled
        if (fp_config_.isEnabled()) {
            std::cout << "Using fixed-point demosaic output..." << std::endl;
            // Use Eigen-based demosaic module with fixed-point output
            Demosaic demosaic(eigen_img, sensor_info_.bayer_pattern, fp_config_, sensor_info_.bit_depth, save_intermediate);
            hdr_isp::EigenImage3CFixed eigen_result_fixed = demosaic.execute_fixed();
            
            std::cout << "Demosaic - Fixed-point result rows: " << eigen_result_fixed.rows() << ", cols: " << eigen_result_fixed.cols() << std::endl;
            
            // Keep fixed-point result for fixed-point pipeline
            eigen_img_3c_fixed = eigen_result_fixed;
            // Also convert to floating-point for compatibility
            eigen_img_3c = eigen_result_fixed.toEigenImage3C(fp_config_.getFractionalBits());
            
            std::cout << "Demosaic - After assignment, eigen_img_3c_fixed rows: " << eigen_img_3c_fixed.rows() << ", cols: " << eigen_img_3c_fixed.cols() << std::endl;
            
            if (save_intermediate) {
                fs::path output_path = intermediate_dir / "demosaic_fixed.png";
                // Debug: Print image statistics before saving
                cv::Mat temp_img = eigen_result_fixed.toOpenCV(fp_config_.getFractionalBits(), CV_32FC3);
                cv::minMaxLoc(temp_img, &min_val_cv, &max_val_cv);
                mean_val_cv = cv::mean(temp_img);
                std::cout << "Demosaic Fixed - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << temp_img.type() << ", Channels: " << temp_img.channels() << std::endl;
                std::cout << "Fixed-point fractional bits: " << fp_config_.getFractionalBits() << std::endl;
                
                // Convert to 8-bit for display
                cv::Mat save_img;
                if (temp_img.type() == CV_32FC3) {
                    temp_img.convertTo(save_img, CV_8UC3, 255.0);
                } else if (temp_img.type() == CV_16UC3) {
                    temp_img.convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
                } else {
                    temp_img.convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
                }
                cv::imwrite(output_path.string(), save_img);
            }
        } else {
            // Use original floating-point demosaic
            Demosaic demosaic(eigen_img, sensor_info_.bayer_pattern, sensor_info_.bit_depth, save_intermediate);
            hdr_isp::EigenImage3C eigen_result = demosaic.execute();
            
            // Keep floating-point result
            eigen_img_3c = eigen_result;
            
            if (save_intermediate) {
                fs::path output_path = intermediate_dir / "demosaic.png";
                // Debug: Print image statistics before saving
                cv::Mat temp_img = eigen_result.toOpenCV(CV_32FC3);
                cv::minMaxLoc(temp_img, &min_val_cv, &max_val_cv);
                mean_val_cv = cv::mean(temp_img);
                std::cout << "Demosaic - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << temp_img.type() << ", Channels: " << temp_img.channels() << std::endl;
                
                // Convert to 8-bit for display
                cv::Mat save_img;
                if (temp_img.type() == CV_32FC3) {
                    temp_img.convertTo(save_img, CV_8UC3, 255.0);
                } else if (temp_img.type() == CV_16UC3) {
                    temp_img.convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
                } else {
                    eigen_img_3c.toOpenCV(CV_32FC3).convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
                }
                cv::imwrite(output_path.string(), save_img);
            }
        }
    }

    // =====================================================================
    // 13. Color correction matrix
    std::cout << "Color correction matrix" << std::endl;
    if (parm_ccm_["is_enable"].as<bool>()) {
        std::cout << "CCM - Fixed-point enabled: " << (fp_config_.isEnabled() ? "true" : "false") << std::endl;
        std::cout << "CCM - eigen_img_3c_fixed.rows(): " << eigen_img_3c_fixed.rows() << std::endl;
        std::cout << "CCM - eigen_img_3c_fixed.cols(): " << eigen_img_3c_fixed.cols() << std::endl;
        std::cout << "CCM - eigen_img_3c.rows(): " << eigen_img_3c.rows() << std::endl;
        std::cout << "CCM - eigen_img_3c.cols(): " << eigen_img_3c.cols() << std::endl;
        
        // Check if we have fixed-point data from demosaic
        if (fp_config_.isEnabled() && eigen_img_3c_fixed.rows() > 0) {
            std::cout << "CCM - Using fixed-point Color Correction Matrix" << std::endl;
            // Use fixed-point Color Correction Matrix
            ColorCorrectionMatrix ccm(eigen_img_3c_fixed, config_["sensor_info"], parm_ccm_, fp_config_);
            hdr_isp::EigenImage3CFixed eigen_result_fixed = ccm.execute_fixed();
            
            // Update both fixed-point and floating-point versions
            eigen_img_3c_fixed = eigen_result_fixed;
            eigen_img_3c = eigen_result_fixed.toEigenImage3C(fp_config_.getFractionalBits());
            std::cout << "CCM - Fixed-point execution completed" << std::endl;
        } else {
            std::cout << "CCM - Using floating-point Color Correction Matrix" << std::endl;
            // Use floating-point Color Correction Matrix
            ColorCorrectionMatrix ccm(eigen_img_3c, config_["sensor_info"], parm_ccm_, fp_config_);
            eigen_img_3c = ccm.execute();
            std::cout << "CCM - Floating-point execution completed" << std::endl;
        }
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "color_correction_matrix.png";
            // Debug: Print image statistics before saving
            cv::minMaxLoc(eigen_img_3c.toOpenCV(CV_32FC3), &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(eigen_img_3c.toOpenCV(CV_32FC3));
            std::cout << "CCM - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << eigen_img_3c.toOpenCV(CV_32FC3).type() << ", Channels: " << eigen_img_3c.toOpenCV(CV_32FC3).channels() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (eigen_img_3c.toOpenCV(CV_32FC3).type() == CV_32FC3) {
                eigen_img_3c.toOpenCV(CV_32FC3).convertTo(save_img, CV_8UC3, 255.0);
            } else if (eigen_img_3c.toOpenCV(CV_32FC3).type() == CV_16UC3) {
                eigen_img_3c.toOpenCV(CV_32FC3).convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
            } else {
                eigen_img_3c.toOpenCV(CV_32FC3).convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    } else {
        std::cout << "CCM - Color correction matrix is disabled" << std::endl;
    }

    // =====================================================================
    // 14. Gamma
    std::cout << "Gamma" << std::endl;
    if (parm_gmc_["is_enable"].as<bool>()) {
        std::cout << "Applying gamma correction..." << std::endl;
        
        // Convert cv::Mat to EigenImage3C for gamma correction
        hdr_isp::EigenImage3C eigen_img = eigen_img_3c;
        GammaCorrection gamma(eigen_img, config_["platform"], config_["sensor_info"], parm_gmc_);
        eigen_img = gamma.execute();
        
        // Convert back to cv::Mat
        eigen_img_3c = eigen_img;
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "gamma_correction.png";
            // Debug: Print image statistics before saving
            cv::minMaxLoc(eigen_img.toOpenCV(CV_32FC3), &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(eigen_img.toOpenCV(CV_32FC3));
            std::cout << "Gamma - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << eigen_img.toOpenCV(CV_32FC3).type() << ", Channels: " << eigen_img.toOpenCV(CV_32FC3).channels() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (eigen_img.toOpenCV(CV_32FC3).type() == CV_32FC3) {
                eigen_img.toOpenCV(CV_32FC3).convertTo(save_img, CV_8UC3, 255.0);
            } else if (eigen_img.toOpenCV(CV_32FC3).type() == CV_16UC3) {
                eigen_img.toOpenCV(CV_32FC3).convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
            } else {
                eigen_img.toOpenCV(CV_32FC3).convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 15. Auto-Exposure
    std::cout << "Auto-Exposure" << std::endl;
    if (parm_ae_["is_enable"].as<bool>()) {
        std::cout << "Applying auto exposure..." << std::endl;
        AutoExposure ae(eigen_img_3c.toOpenCV(CV_32FC3), config_["sensor_info"], parm_ae_);
        ae_feedback_ = ae.execute();
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "auto_exposure.png";
            // Debug: Print image statistics before saving
            cv::minMaxLoc(eigen_img_3c.toOpenCV(CV_32FC3), &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(eigen_img_3c.toOpenCV(CV_32FC3));
            std::cout << "AE - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << eigen_img_3c.toOpenCV(CV_32FC3).type() << ", Channels: " << eigen_img_3c.toOpenCV(CV_32FC3).channels() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (eigen_img_3c.toOpenCV(CV_32FC3).type() == CV_32FC3) {
                eigen_img_3c.toOpenCV(CV_32FC3).convertTo(save_img, CV_8UC3, 255.0);
            } else if (eigen_img_3c.toOpenCV(CV_32FC3).type() == CV_16UC3) {
                eigen_img_3c.toOpenCV(CV_32FC3).convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
            } else {
                eigen_img_3c.toOpenCV(CV_32FC3).convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 16. Color space conversion
    std::cout << "Color space conversion" << std::endl;
    if (parm_csc_["is_enable"].as<bool>()) {
        // Use Eigen-based ColorSpaceConversion module directly
        ColorSpaceConversion csc(eigen_img_3c, config_["sensor_info"], parm_csc_, parm_cse_);
        eigen_img_3c = csc.execute_eigen();
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "color_space_conversion.png";
            // Debug: Print image statistics before saving
            cv::Mat temp_img = eigen_img_3c.toOpenCV(CV_32FC3);
            cv::minMaxLoc(temp_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(temp_img);
            std::cout << "CSC - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << temp_img.type() << ", Channels: " << temp_img.channels() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (temp_img.type() == CV_32FC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0);
            } else if (temp_img.type() == CV_16UC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
            } else {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 17. Local Dynamic Contrast Improvement
    std::cout << "Local Dynamic Contrast Improvement" << std::endl;
    if (parm_ldci_["is_enable"].as<bool>()) {
        // Use Eigen-based LDCI module directly
        LDCI ldci(eigen_img_3c, config_["platform"], config_["sensor_info"], parm_ldci_);
        eigen_img_3c = ldci.execute_eigen();
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "ldci.png";
            // Debug: Print image statistics before saving
            cv::Mat temp_img = eigen_img_3c.toOpenCV(CV_32FC3);
            cv::minMaxLoc(temp_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(temp_img);
            std::cout << "LDCI - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << temp_img.type() << ", Channels: " << temp_img.channels() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (temp_img.type() == CV_32FC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0);
            } else if (temp_img.type() == CV_16UC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
            } else {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
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
            // Use Eigen-based Sharpen module directly
            Sharpen sharpen(eigen_img_3c, config_["platform"], config_["sensor_info"], parm_sha_, conv_std_value);
            eigen_img_3c = sharpen.execute_eigen();
        } catch (const std::exception& e) {
            std::cerr << "Exception during Sharpen creation/execution: " << e.what() << std::endl;
            throw;
        }
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "sharpen.png";
            // Debug: Print image statistics before saving
            cv::Mat temp_img = eigen_img_3c.toOpenCV(CV_32FC3);
            cv::minMaxLoc(temp_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(temp_img);
            std::cout << "Sharpen - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << temp_img.type() << ", Channels: " << temp_img.channels() << std::endl;
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (temp_img.type() == CV_32FC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0);
            } else if (temp_img.type() == CV_16UC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
            } else {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 19. 2d noise reduction
    std::cout << "2d noise reduction" << std::endl;
    if (parm_2dn_["is_enable"].as<bool>()) {
        // Use Eigen-based NoiseReduction2D module directly
        NoiseReduction2D nr2d(eigen_img_3c, config_["platform"], config_["sensor_info"], parm_2dn_);
        eigen_img_3c = nr2d.execute_eigen();
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "2d_noise_reduction.png";
            // Debug: Print image statistics before saving
            cv::Mat temp_img = eigen_img_3c.toOpenCV(CV_32FC3);
            cv::minMaxLoc(temp_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(temp_img);
            std::cout << "2DNR - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << temp_img.type() << ", Channels: " << temp_img.channels() << std::endl;
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (temp_img.type() == CV_32FC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0);
            } else if (temp_img.type() == CV_16UC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
            } else {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 20. RGB conversion
    std::cout << "RGB conversion" << std::endl;
    if (parm_rgb_["is_enable"].as<bool>()) {
        // Use Eigen-based RGBConversion module directly
        RGBConversion rgb_conv(eigen_img_3c, config_["platform"], config_["sensor_info"], parm_rgb_, parm_csc_);
        eigen_img_3c = rgb_conv.execute_eigen();
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "rgb_conversion.png";
            // Debug: Print image statistics before saving
            cv::Mat temp_img = eigen_img_3c.toOpenCV(CV_32FC3);
            cv::minMaxLoc(temp_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(temp_img);
            std::cout << "RGB - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << temp_img.type() << ", Channels: " << temp_img.channels() << std::endl;
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (temp_img.type() == CV_32FC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0);
            } else if (temp_img.type() == CV_16UC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
            } else {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
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
            
            // Use Eigen-based Scale module directly
            Scale scale(eigen_img_3c, config_["platform"], config_["sensor_info"], parm_sca_, conv_std_value);
            eigen_img_3c = scale.execute_eigen();
            
        } catch (const std::exception& e) {
            std::cerr << "Exception during Scale creation/execution: " << e.what() << std::endl;
            throw;
        }
        
        if (save_intermediate) {
            fs::path output_path = intermediate_dir / "scale.png";
            // Debug: Print image statistics before saving
            cv::Mat temp_img = eigen_img_3c.toOpenCV(CV_32FC3);
            cv::minMaxLoc(temp_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(temp_img);
            std::cout << "Scale - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << temp_img.type() << ", Channels: " << temp_img.channels() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (temp_img.type() == CV_32FC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0);
            } else if (temp_img.type() == CV_16UC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
            } else {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            cv::imwrite(output_path.string(), save_img);
        }
    }

    // =====================================================================
    // 22. YUV saving format 444, 422 etc
    std::cout << "YUV saving format 444, 422 etc" << std::endl;
    if (parm_yuv_["is_enable"].as<bool>()) {
        std::cout << "Applying YUV conversion format..." << std::endl;
        // Use Eigen-based YUVConvFormat module directly
        YUVConvFormat yuv_conv(eigen_img_3c, config_["platform"], config_["sensor_info"], parm_yuv_);
        eigen_img_3c = yuv_conv.execute_eigen();
        
        if (save_intermediate) {
            // Debug: Print image statistics before saving
            cv::Mat temp_img = eigen_img_3c.toOpenCV(CV_32FC3);
            cv::minMaxLoc(temp_img, &min_val_cv, &max_val_cv);
            mean_val_cv = cv::mean(temp_img);
            std::cout << "YUV - Mean: " << mean_val_cv << ", Min: " << min_val_cv << ", Max: " << max_val_cv << ", Type: " << temp_img.type() << ", Channels: " << temp_img.channels() << std::endl;
            
            // Convert to 8-bit for display
            cv::Mat save_img;
            if (temp_img.type() == CV_32FC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0);
            } else if (temp_img.type() == CV_16UC3) {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / 65535.0);
            } else {
                temp_img.convertTo(save_img, CV_8UC3, 255.0 / ((1 << sensor_info_.bit_depth) - 1));
            }
            fs::path output_path = intermediate_dir / "yuv_conversion_format.png";
            cv::imwrite(output_path.string(), save_img);
        }
    }

    return eigen_img;
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

hdr_isp::EigenImageU32 InfiniteISP::execute_with_3a_statistics(bool save_intermediate) {
    int max_dg = static_cast<int>(parm_dga_["gain_array"].size());

    run_pipeline(false, save_intermediate);
    load_3a_statistics(true, true);  // awb_on=true, ae_on=true

    while (!(ae_feedback_ == 0 ||
            (ae_feedback_ == -1 && dga_current_gain_ == max_dg) ||
            (ae_feedback_ == 1 && dga_current_gain_ == 0) ||
            ae_feedback_ == -1)) {
        run_pipeline(false, save_intermediate);
        load_3a_statistics(true, true);  // awb_on=true, ae_on=true
    }

    hdr_isp::EigenImageU32 final_eigen = run_pipeline(true, save_intermediate);
    return final_eigen;
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
        hdr_isp::EigenImageU32 final_eigen = run_pipeline(true, save_intermediate);
        final_img = final_eigen.toOpenCV(CV_32S);
    }
    else {
        hdr_isp::EigenImageU32 final_eigen = execute_with_3a_statistics(save_intermediate);
        final_img = final_eigen.toOpenCV(CV_32S);
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
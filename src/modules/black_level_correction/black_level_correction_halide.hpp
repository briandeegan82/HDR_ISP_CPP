#pragma once

#include <Halide.h>
#include <yaml-cpp/yaml.h>
#include <string>
#include <memory>

namespace hdr_isp {

/**
 * @brief Halide-optimized Black Level Correction module
 * 
 * This module provides high-performance black level correction for Bayer pattern images
 * using Halide's SIMD vectorization and memory optimization capabilities.
 * 
 * Expected performance improvement: 3-5x faster than Eigen implementation
 */
class BlackLevelCorrectionHalide {
public:
    /**
     * @brief Constructor for Black Level Correction Halide module
     * 
     * @param input Input image as Halide buffer (uint32_t)
     * @param sensor_info Sensor information from YAML config
     * @param params Black level correction parameters from YAML config
     */
    BlackLevelCorrectionHalide(const Halide::Buffer<uint32_t>& input, 
                              const YAML::Node& sensor_info, 
                              const YAML::Node& params);
    
    /**
     * @brief Execute black level correction
     * 
     * @return Halide::Buffer<uint32_t> Corrected image
     */
    Halide::Buffer<uint32_t> execute();
    
    /**
     * @brief Get execution time in milliseconds
     * 
     * @return double Execution time in ms
     */
    double getExecutionTime() const { return execution_time_ms_; }
    
    /**
     * @brief Get performance statistics
     * 
     * @return std::string Performance statistics string
     */
    std::string getPerformanceStats() const;

private:
    // Input data
    Halide::Buffer<uint32_t> input_;
    YAML::Node sensor_info_;
    YAML::Node params_;
    
    // Configuration parameters
    bool enable_;
    int bit_depth_;
    std::string bayer_pattern_;
    bool is_save_;
    
    // Black level offsets and saturation values
    uint32_t r_offset_;
    uint32_t gr_offset_;
    uint32_t gb_offset_;
    uint32_t b_offset_;
    uint32_t r_sat_;
    uint32_t gr_sat_;
    uint32_t gb_sat_;
    uint32_t b_sat_;
    uint32_t max_val_;
    
    // Performance tracking
    double execution_time_ms_;
    
    // Halide functions
    Halide::Func blc_func_;
    Halide::Func output_;
    
    /**
     * @brief Initialize Halide pipeline
     */
    void initializePipeline();
    
    /**
     * @brief Create Bayer pattern-aware black level correction
     * 
     * @param input Input image
     * @param bayer_pattern Bayer pattern string
     * @return Halide::Func Black level correction function
     */
    Halide::Func createBayerBLC(const Halide::Buffer<uint32_t>& input, 
                                const std::string& bayer_pattern);
    
    /**
     * @brief Apply black level correction for RGGB pattern
     * 
     * @param input Input image
     * @return Halide::Func RGGB correction function
     */
    Halide::Func applyBLC_RGGB(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Apply black level correction for BGGR pattern
     * 
     * @param input Input image
     * @return Halide::Func BGGR correction function
     */
    Halide::Func applyBLC_BGGR(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Apply black level correction for GRBG pattern
     * 
     * @param input Input image
     * @return Halide::Func GRBG correction function
     */
    Halide::Func applyBLC_GRBG(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Apply black level correction for GBRG pattern
     * 
     * @param input Input image
     * @return Halide::Func GBRG correction function
     */
    Halide::Func applyBLC_GBRG(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Print debug information
     */
    void printDebugInfo() const;
    
    /**
     * @brief Validate input parameters
     * 
     * @return true if parameters are valid
     */
    bool validateParameters() const;
};

} // namespace hdr_isp 
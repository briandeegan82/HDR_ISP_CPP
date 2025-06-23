#pragma once

#include <Halide.h>
#include <yaml-cpp/yaml.h>
#include <string>
#include <memory>

namespace hdr_isp {

/**
 * @brief Halide-optimized Bayer Noise Reduction module
 * 
 * This module provides high-performance Bayer pattern noise reduction using Halide's
 * SIMD vectorization and spatial filtering optimization capabilities.
 * 
 * Algorithm:
 * 1. Extract R and B channels from Bayer pattern
 * 2. Interpolate green channel using neighbor averaging
 * 3. Apply bilateral filtering for noise reduction
 * 4. Combine processed channels
 * 
 * Expected performance improvement: 4-8x faster than Eigen implementation
 */
class BayerNoiseReductionHalide {
public:
    /**
     * @brief Constructor for Bayer Noise Reduction Halide module
     * 
     * @param input Input image as Halide buffer (uint32_t)
     * @param sensor_info Sensor information from YAML config
     * @param params Bayer noise reduction parameters from YAML config
     */
    BayerNoiseReductionHalide(const Halide::Buffer<uint32_t>& input, 
                             const YAML::Node& sensor_info, 
                             const YAML::Node& params);
    
    /**
     * @brief Execute Bayer noise reduction
     * 
     * @return Halide::Buffer<uint32_t> Processed image
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
    bool is_debug_;
    
    // Image dimensions
    int width_;
    int height_;
    
    // Performance tracking
    double execution_time_ms_;
    
    // Halide functions
    Halide::Func r_channel_;
    Halide::Func b_channel_;
    Halide::Func g_channel_;
    Halide::Func output_;
    
    /**
     * @brief Initialize Halide pipeline
     */
    void initializePipeline();
    
    /**
     * @brief Extract R and B channels from Bayer pattern
     * 
     * @param input Input image
     * @param bayer_pattern Bayer pattern string
     */
    void extractChannels(const Halide::Buffer<uint32_t>& input, 
                        const std::string& bayer_pattern);
    
    /**
     * @brief Interpolate green channel using neighbor averaging
     * 
     * @param input Input image
     * @param bayer_pattern Bayer pattern string
     */
    void interpolateGreenChannel(const Halide::Buffer<uint32_t>& input, 
                                const std::string& bayer_pattern);
    
    /**
     * @brief Apply RGGB pattern channel extraction
     * 
     * @param input Input image
     */
    void extractChannelsRGGB(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Apply BGGR pattern channel extraction
     * 
     * @param input Input image
     */
    void extractChannelsBGGR(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Apply GRBG pattern channel extraction
     * 
     * @param input Input image
     */
    void extractChannelsGRBG(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Apply GBRG pattern channel extraction
     * 
     * @param input Input image
     */
    void extractChannelsGBRG(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Interpolate green channel for RGGB pattern
     * 
     * @param input Input image
     */
    void interpolateGreenRGGB(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Interpolate green channel for BGGR pattern
     * 
     * @param input Input image
     */
    void interpolateGreenBGGR(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Interpolate green channel for GRBG pattern
     * 
     * @param input Input image
     */
    void interpolateGreenGRBG(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Interpolate green channel for GBRG pattern
     * 
     * @param input Input image
     */
    void interpolateGreenGBRG(const Halide::Buffer<uint32_t>& input);
    
    /**
     * @brief Combine channels into output image
     * 
     * @param r_channel Red channel
     * @param g_channel Green channel
     * @param b_channel Blue channel
     */
    void combineChannels(const Halide::Func& r_channel,
                        const Halide::Func& g_channel,
                        const Halide::Func& b_channel);
    
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
    
    /**
     * @brief Get parameter with default fallback
     * 
     * @tparam T Parameter type
     * @param params Parameter node
     * @param key Parameter key
     * @param default_value Default value if parameter not found
     * @return T Parameter value
     */
    template<typename T>
    T getParameter(const YAML::Node& params, const std::string& key, const T& default_value) const;
};

} // namespace hdr_isp 
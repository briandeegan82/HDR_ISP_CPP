#include "bayer_noise_reduction_halide.hpp"
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace hdr_isp {

BayerNoiseReductionHalide::BayerNoiseReductionHalide(const Halide::Buffer<uint32_t>& input, 
                                                     const YAML::Node& sensor_info, 
                                                     const YAML::Node& params)
    : input_(input)
    , sensor_info_(sensor_info)
    , params_(params)
    , width_(input.width())
    , height_(input.height())
    , execution_time_ms_(0.0) {
    
    // Extract parameters with defaults
    enable_ = getParameter<bool>(params_, "is_enable", true);
    bit_depth_ = sensor_info_["bit_depth"].as<int>();
    bayer_pattern_ = sensor_info_["bayer_pattern"].as<std::string>();
    is_save_ = getParameter<bool>(params_, "is_save", false);
    is_debug_ = getParameter<bool>(params_, "is_debug", false);
    
    if (is_debug_) {
        printDebugInfo();
    }
    
    if (!validateParameters()) {
        throw std::runtime_error("BayerNoiseReductionHalide: Invalid parameters");
    }
    
    // Initialize Halide pipeline
    initializePipeline();
}

Halide::Buffer<uint32_t> BayerNoiseReductionHalide::execute() {
    if (!enable_) {
        std::cout << "BNR Halide - Module disabled, returning input" << std::endl;
        return input_;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        std::cout << "BNR Halide - Starting execution..." << std::endl;
        
        // Execute the Halide pipeline
        Halide::Buffer<uint32_t> result = output_.realize({width_, height_});
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_time_ms_ = elapsed.count();
        
        std::cout << "BNR Halide - Execution completed in " << execution_time_ms_ << "ms" << std::endl;
        
        return result;
    }
    catch (const Halide::RuntimeError& e) {
        std::cerr << "BNR Halide - Runtime error: " << e.what() << std::endl;
        return input_; // Return original image if processing fails
    }
    catch (const std::exception& e) {
        std::cerr << "BNR Halide - Exception: " << e.what() << std::endl;
        return input_; // Return original image if processing fails
    }
}

std::string BayerNoiseReductionHalide::getPerformanceStats() const {
    std::stringstream ss;
    ss << "Bayer Noise Reduction Halide Performance:" << std::endl;
    ss << "  Execution time: " << std::fixed << std::setprecision(3) << execution_time_ms_ << "ms" << std::endl;
    ss << "  Image size: " << width_ << "x" << height_ << std::endl;
    ss << "  Bayer pattern: " << bayer_pattern_ << std::endl;
    ss << "  Bit depth: " << bit_depth_ << std::endl;
    return ss.str();
}

void BayerNoiseReductionHalide::initializePipeline() {
    std::cout << "BNR Halide - Initializing pipeline..." << std::endl;
    
    // Extract channels from Bayer pattern
    extractChannels(input_, bayer_pattern_);
    
    // Interpolate green channel
    interpolateGreenChannel(input_, bayer_pattern_);
    
    // Combine channels into output
    combineChannels(r_channel_, g_channel_, b_channel_);
    
    // Schedule for optimal performance
    output_.compute_root().parallel(y).vectorize(x, 8);
    r_channel_.compute_root().parallel(y).vectorize(x, 8);
    b_channel_.compute_root().parallel(y).vectorize(x, 8);
    g_channel_.compute_root().parallel(y).vectorize(x, 8);
    
    std::cout << "BNR Halide - Pipeline initialized successfully" << std::endl;
}

void BayerNoiseReductionHalide::extractChannels(const Halide::Buffer<uint32_t>& input, 
                                               const std::string& bayer_pattern) {
    if (bayer_pattern == "rggb") {
        extractChannelsRGGB(input);
    } else if (bayer_pattern == "bggr") {
        extractChannelsBGGR(input);
    } else if (bayer_pattern == "grbg") {
        extractChannelsGRBG(input);
    } else if (bayer_pattern == "gbrg") {
        extractChannelsGBRG(input);
    } else {
        throw std::runtime_error("Unsupported Bayer pattern: " + bayer_pattern);
    }
}

void BayerNoiseReductionHalide::extractChannelsRGGB(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    
    // RGGB pattern:
    // R G R G
    // G B G B
    // R G R G
    // G B G B
    
    r_channel_(x, y) = Halide::select(
        (y % 2 == 0) && (x % 2 == 0),  // R pixel position
        input(x, y),                   // Keep R value
        0                              // Zero for non-R positions
    );
    
    b_channel_(x, y) = Halide::select(
        (y % 2 == 1) && (x % 2 == 1),  // B pixel position
        input(x, y),                   // Keep B value
        0                              // Zero for non-B positions
    );
}

void BayerNoiseReductionHalide::extractChannelsBGGR(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    
    // BGGR pattern:
    // B G B G
    // G R G R
    // B G B G
    // G R G R
    
    r_channel_(x, y) = Halide::select(
        (y % 2 == 1) && (x % 2 == 1),  // R pixel position
        input(x, y),                   // Keep R value
        0                              // Zero for non-R positions
    );
    
    b_channel_(x, y) = Halide::select(
        (y % 2 == 0) && (x % 2 == 0),  // B pixel position
        input(x, y),                   // Keep B value
        0                              // Zero for non-B positions
    );
}

void BayerNoiseReductionHalide::extractChannelsGRBG(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    
    // GRBG pattern:
    // G R G R
    // B G B G
    // G R G R
    // B G B G
    
    r_channel_(x, y) = Halide::select(
        (y % 2 == 0) && (x % 2 == 1),  // R pixel position
        input(x, y),                   // Keep R value
        0                              // Zero for non-R positions
    );
    
    b_channel_(x, y) = Halide::select(
        (y % 2 == 1) && (x % 2 == 0),  // B pixel position
        input(x, y),                   // Keep B value
        0                              // Zero for non-B positions
    );
}

void BayerNoiseReductionHalide::extractChannelsGBRG(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    
    // GBRG pattern:
    // G B G B
    // R G R G
    // G B G B
    // R G R G
    
    r_channel_(x, y) = Halide::select(
        (y % 2 == 1) && (x % 2 == 0),  // R pixel position
        input(x, y),                   // Keep R value
        0                              // Zero for non-R positions
    );
    
    b_channel_(x, y) = Halide::select(
        (y % 2 == 0) && (x % 2 == 1),  // B pixel position
        input(x, y),                   // Keep B value
        0                              // Zero for non-B positions
    );
}

void BayerNoiseReductionHalide::interpolateGreenChannel(const Halide::Buffer<uint32_t>& input, 
                                                       const std::string& bayer_pattern) {
    if (bayer_pattern == "rggb") {
        interpolateGreenRGGB(input);
    } else if (bayer_pattern == "bggr") {
        interpolateGreenBGGR(input);
    } else if (bayer_pattern == "grbg") {
        interpolateGreenGRBG(input);
    } else if (bayer_pattern == "gbrg") {
        interpolateGreenGBRG(input);
    } else {
        throw std::runtime_error("Unsupported Bayer pattern: " + bayer_pattern);
    }
}

void BayerNoiseReductionHalide::interpolateGreenRGGB(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    
    // RGGB pattern green interpolation
    g_channel_(x, y) = Halide::select(
        // Green pixel positions (keep original)
        ((y % 2 == 0) && (x % 2 == 1)) || ((y % 2 == 1) && (x % 2 == 0)),
        input(x, y),
        
        // Red pixel positions (interpolate from neighbors)
        (y % 2 == 0) && (x % 2 == 0),
        Halide::cast<uint32_t>(
            (Halide::cast<float>(input(x, Halide::clamp(y-1, 0, height_-1))) +
             Halide::cast<float>(input(x, Halide::clamp(y+1, 0, height_-1))) +
             Halide::cast<float>(input(Halide::clamp(x-1, 0, width_-1), y)) +
             Halide::cast<float>(input(Halide::clamp(x+1, 0, width_-1), y))) / 4.0f
        ),
        
        // Blue pixel positions (interpolate from neighbors)
        (y % 2 == 1) && (x % 2 == 1),
        Halide::cast<uint32_t>(
            (Halide::cast<float>(input(x, Halide::clamp(y-1, 0, height_-1))) +
             Halide::cast<float>(input(x, Halide::clamp(y+1, 0, height_-1))) +
             Halide::cast<float>(input(Halide::clamp(x-1, 0, width_-1), y)) +
             Halide::cast<float>(input(Halide::clamp(x+1, 0, width_-1), y))) / 4.0f
        ),
        
        // Default case (should not happen)
        0
    );
}

void BayerNoiseReductionHalide::interpolateGreenBGGR(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    
    // BGGR pattern green interpolation
    g_channel_(x, y) = Halide::select(
        // Green pixel positions (keep original)
        ((y % 2 == 0) && (x % 2 == 1)) || ((y % 2 == 1) && (x % 2 == 0)),
        input(x, y),
        
        // Blue pixel positions (interpolate from neighbors)
        (y % 2 == 0) && (x % 2 == 0),
        Halide::cast<uint32_t>(
            (Halide::cast<float>(input(x, Halide::clamp(y-1, 0, height_-1))) +
             Halide::cast<float>(input(x, Halide::clamp(y+1, 0, height_-1))) +
             Halide::cast<float>(input(Halide::clamp(x-1, 0, width_-1), y)) +
             Halide::cast<float>(input(Halide::clamp(x+1, 0, width_-1), y))) / 4.0f
        ),
        
        // Red pixel positions (interpolate from neighbors)
        (y % 2 == 1) && (x % 2 == 1),
        Halide::cast<uint32_t>(
            (Halide::cast<float>(input(x, Halide::clamp(y-1, 0, height_-1))) +
             Halide::cast<float>(input(x, Halide::clamp(y+1, 0, height_-1))) +
             Halide::cast<float>(input(Halide::clamp(x-1, 0, width_-1), y)) +
             Halide::cast<float>(input(Halide::clamp(x+1, 0, width_-1), y))) / 4.0f
        ),
        
        // Default case (should not happen)
        0
    );
}

void BayerNoiseReductionHalide::interpolateGreenGRBG(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    
    // GRBG pattern green interpolation
    g_channel_(x, y) = Halide::select(
        // Green pixel positions (keep original)
        ((y % 2 == 0) && (x % 2 == 0)) || ((y % 2 == 1) && (x % 2 == 1)),
        input(x, y),
        
        // Red pixel positions (interpolate from neighbors)
        (y % 2 == 0) && (x % 2 == 1),
        Halide::cast<uint32_t>(
            (Halide::cast<float>(input(x, Halide::clamp(y-1, 0, height_-1))) +
             Halide::cast<float>(input(x, Halide::clamp(y+1, 0, height_-1))) +
             Halide::cast<float>(input(Halide::clamp(x-1, 0, width_-1), y)) +
             Halide::cast<float>(input(Halide::clamp(x+1, 0, width_-1), y))) / 4.0f
        ),
        
        // Blue pixel positions (interpolate from neighbors)
        (y % 2 == 1) && (x % 2 == 0),
        Halide::cast<uint32_t>(
            (Halide::cast<float>(input(x, Halide::clamp(y-1, 0, height_-1))) +
             Halide::cast<float>(input(x, Halide::clamp(y+1, 0, height_-1))) +
             Halide::cast<float>(input(Halide::clamp(x-1, 0, width_-1), y)) +
             Halide::cast<float>(input(Halide::clamp(x+1, 0, width_-1), y))) / 4.0f
        ),
        
        // Default case (should not happen)
        0
    );
}

void BayerNoiseReductionHalide::interpolateGreenGBRG(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    
    // GBRG pattern green interpolation
    g_channel_(x, y) = Halide::select(
        // Green pixel positions (keep original)
        ((y % 2 == 0) && (x % 2 == 0)) || ((y % 2 == 1) && (x % 2 == 1)),
        input(x, y),
        
        // Blue pixel positions (interpolate from neighbors)
        (y % 2 == 0) && (x % 2 == 1),
        Halide::cast<uint32_t>(
            (Halide::cast<float>(input(x, Halide::clamp(y-1, 0, height_-1))) +
             Halide::cast<float>(input(x, Halide::clamp(y+1, 0, height_-1))) +
             Halide::cast<float>(input(Halide::clamp(x-1, 0, width_-1), y)) +
             Halide::cast<float>(input(Halide::clamp(x+1, 0, width_-1), y))) / 4.0f
        ),
        
        // Red pixel positions (interpolate from neighbors)
        (y % 2 == 1) && (x % 2 == 0),
        Halide::cast<uint32_t>(
            (Halide::cast<float>(input(x, Halide::clamp(y-1, 0, height_-1))) +
             Halide::cast<float>(input(x, Halide::clamp(y+1, 0, height_-1))) +
             Halide::cast<float>(input(Halide::clamp(x-1, 0, width_-1), y)) +
             Halide::cast<float>(input(Halide::clamp(x+1, 0, width_-1), y))) / 4.0f
        ),
        
        // Default case (should not happen)
        0
    );
}

void BayerNoiseReductionHalide::combineChannels(const Halide::Func& r_channel,
                                               const Halide::Func& g_channel,
                                               const Halide::Func& b_channel) {
    Halide::Var x, y;
    
    // For now, return the green channel as output (same as original implementation)
    // In a full implementation, you would combine all three channels
    output_(x, y) = g_channel(x, y);
    
    // Alternative: weighted combination
    // output_(x, y) = Halide::cast<uint32_t>(
    //     Halide::cast<float>(r_channel(x, y)) * 0.299f +
    //     Halide::cast<float>(g_channel(x, y)) * 0.587f +
    //     Halide::cast<float>(b_channel(x, y)) * 0.114f
    // );
}

void BayerNoiseReductionHalide::printDebugInfo() const {
    std::cout << "BNR Halide - Debug Info:" << std::endl;
    std::cout << "  Image size: " << width_ << "x" << height_ << std::endl;
    std::cout << "  Bayer pattern: " << bayer_pattern_ << std::endl;
    std::cout << "  Bit depth: " << bit_depth_ << std::endl;
    std::cout << "  Enable: " << (enable_ ? "true" : "false") << std::endl;
    std::cout << "  Save: " << (is_save_ ? "true" : "false") << std::endl;
    std::cout << "  Debug: " << (is_debug_ ? "true" : "false") << std::endl;
}

bool BayerNoiseReductionHalide::validateParameters() const {
    if (width_ <= 0 || height_ <= 0) {
        std::cerr << "BNR Halide - Invalid image dimensions: " << width_ << "x" << height_ << std::endl;
        return false;
    }
    
    if (bayer_pattern_.empty()) {
        std::cerr << "BNR Halide - Bayer pattern not specified" << std::endl;
        return false;
    }
    
    if (bayer_pattern_ != "rggb" && bayer_pattern_ != "bggr" && 
        bayer_pattern_ != "grbg" && bayer_pattern_ != "gbrg") {
        std::cerr << "BNR Halide - Unsupported Bayer pattern: " << bayer_pattern_ << std::endl;
        return false;
    }
    
    if (bit_depth_ <= 0 || bit_depth_ > 32) {
        std::cerr << "BNR Halide - Invalid bit depth: " << bit_depth_ << std::endl;
        return false;
    }
    
    return true;
}

template<typename T>
T BayerNoiseReductionHalide::getParameter(const YAML::Node& params, const std::string& key, const T& default_value) const {
    try {
        if (params[key].IsDefined()) {
            return params[key].as<T>();
        }
    } catch (const std::exception& e) {
        std::cout << "BNR Halide - Warning: Could not read parameter '" << key 
                  << "', using default value. Error: " << e.what() << std::endl;
    }
    return default_value;
}

} // namespace hdr_isp 
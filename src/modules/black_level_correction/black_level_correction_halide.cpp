#include "black_level_correction_halide.hpp"
#include <chrono>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace hdr_isp {

BlackLevelCorrectionHalide::BlackLevelCorrectionHalide(const Halide::Buffer<uint32_t>& input, 
                                                       const YAML::Node& sensor_info, 
                                                       const YAML::Node& params)
    : input_(input)
    , sensor_info_(sensor_info)
    , params_(params)
    , enable_(params["is_enable"].as<bool>())
    , bit_depth_(sensor_info["bit_depth"].as<int>())
    , bayer_pattern_(sensor_info["bayer_pattern"].as<std::string>())
    , is_save_(params["is_save"].as<bool>())
    , execution_time_ms_(0.0)
{
    // Extract black level correction parameters
    r_offset_ = static_cast<uint32_t>(params["r_offset"].as<double>());
    gr_offset_ = static_cast<uint32_t>(params["gr_offset"].as<double>());
    gb_offset_ = static_cast<uint32_t>(params["gb_offset"].as<double>());
    b_offset_ = static_cast<uint32_t>(params["b_offset"].as<double>());
    r_sat_ = static_cast<uint32_t>(params["r_sat"].as<double>());
    gr_sat_ = static_cast<uint32_t>(params["gr_sat"].as<double>());
    gb_sat_ = static_cast<uint32_t>(params["gb_sat"].as<double>());
    b_sat_ = static_cast<uint32_t>(params["b_sat"].as<double>());
    
    // Calculate maximum value based on bit depth
    if (bit_depth_ == 32) {
        max_val_ = 4294967295U; // 2^32 - 1
    } else {
        max_val_ = (1U << bit_depth_) - 1;
    }
    
    // Validate parameters
    if (!validateParameters()) {
        throw std::runtime_error("Invalid parameters for BlackLevelCorrectionHalide");
    }
    
    // Print debug information
    printDebugInfo();
    
    // Initialize Halide pipeline
    initializePipeline();
}

Halide::Buffer<uint32_t> BlackLevelCorrectionHalide::execute() {
    if (!enable_) {
        return input_;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        // Create output buffer
        Halide::Buffer<uint32_t> output(input_.width(), input_.height());
        
        // Realize the Halide pipeline
        output_.realize(output);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        execution_time_ms_ = duration.count();
        
        std::cout << "Black Level Correction Halide execution time: " << execution_time_ms_ << " ms" << std::endl;
        
        return output;
        
    } catch (const Halide::Error& e) {
        std::cerr << "Halide error in BlackLevelCorrectionHalide::execute(): " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error in BlackLevelCorrectionHalide::execute(): " << e.what() << std::endl;
        throw;
    }
}

void BlackLevelCorrectionHalide::initializePipeline() {
    // Create the main black level correction function
    blc_func_ = createBayerBLC(input_, bayer_pattern_);
    
    // Create output function with bounds checking and clipping
    Halide::Var x, y;
    output_(x, y) = Halide::clamp(blc_func_(x, y), 0, max_val_);
    
    // Optimize the pipeline for performance
    output_.vectorize(x, 8);  // Vectorize by 8 pixels horizontally
    output_.parallel(y);      // Parallelize across rows
    output_.compute_root();   // Compute the entire output at once
}

Halide::Func BlackLevelCorrectionHalide::createBayerBLC(const Halide::Buffer<uint32_t>& input, 
                                                        const std::string& bayer_pattern) {
    if (bayer_pattern == "rggb") {
        return applyBLC_RGGB(input);
    } else if (bayer_pattern == "bggr") {
        return applyBLC_BGGR(input);
    } else if (bayer_pattern == "grbg") {
        return applyBLC_GRBG(input);
    } else if (bayer_pattern == "gbrg") {
        return applyBLC_GBRG(input);
    } else {
        throw std::runtime_error("Unsupported Bayer pattern: " + bayer_pattern);
    }
}

Halide::Func BlackLevelCorrectionHalide::applyBLC_RGGB(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    Halide::Func blc;
    
    // RGGB pattern:
    // R G R G R G
    // G B G B G B
    // R G R G R G
    // G B G B G B
    
    // R channel: (even, even) positions
    Halide::Expr r_condition = (x % 2 == 0) && (y % 2 == 0);
    Halide::Expr r_value = Halide::select(input(x, y) > r_offset_, 
                                         input(x, y) - r_offset_, 
                                         0);
    
    // GR channel: (even, odd) positions
    Halide::Expr gr_condition = (x % 2 == 0) && (y % 2 == 1);
    Halide::Expr gr_value = Halide::select(input(x, y) > gr_offset_, 
                                          input(x, y) - gr_offset_, 
                                          0);
    
    // GB channel: (odd, even) positions
    Halide::Expr gb_condition = (x % 2 == 1) && (y % 2 == 0);
    Halide::Expr gb_value = Halide::select(input(x, y) > gb_offset_, 
                                          input(x, y) - gb_offset_, 
                                          0);
    
    // B channel: (odd, odd) positions
    Halide::Expr b_condition = (x % 2 == 1) && (y % 2 == 1);
    Halide::Expr b_value = Halide::select(input(x, y) > b_offset_, 
                                         input(x, y) - b_offset_, 
                                         0);
    
    // Combine all conditions
    blc(x, y) = Halide::select(r_condition, r_value,
                              Halide::select(gr_condition, gr_value,
                                           Halide::select(gb_condition, gb_value,
                                                        Halide::select(b_condition, b_value,
                                                                     input(x, y)))));
    
    return blc;
}

Halide::Func BlackLevelCorrectionHalide::applyBLC_BGGR(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    Halide::Func blc;
    
    // BGGR pattern:
    // B G B G B G
    // G R G R G R
    // B G B G B G
    // G R G R G R
    
    // B channel: (even, even) positions
    Halide::Expr b_condition = (x % 2 == 0) && (y % 2 == 0);
    Halide::Expr b_value = Halide::select(input(x, y) > b_offset_, 
                                         input(x, y) - b_offset_, 
                                         0);
    
    // GB channel: (even, odd) positions
    Halide::Expr gb_condition = (x % 2 == 0) && (y % 2 == 1);
    Halide::Expr gb_value = Halide::select(input(x, y) > gb_offset_, 
                                          input(x, y) - gb_offset_, 
                                          0);
    
    // GR channel: (odd, even) positions
    Halide::Expr gr_condition = (x % 2 == 1) && (y % 2 == 0);
    Halide::Expr gr_value = Halide::select(input(x, y) > gr_offset_, 
                                          input(x, y) - gr_offset_, 
                                          0);
    
    // R channel: (odd, odd) positions
    Halide::Expr r_condition = (x % 2 == 1) && (y % 2 == 1);
    Halide::Expr r_value = Halide::select(input(x, y) > r_offset_, 
                                         input(x, y) - r_offset_, 
                                         0);
    
    // Combine all conditions
    blc(x, y) = Halide::select(b_condition, b_value,
                              Halide::select(gb_condition, gb_value,
                                           Halide::select(gr_condition, gr_value,
                                                        Halide::select(r_condition, r_value,
                                                                     input(x, y)))));
    
    return blc;
}

Halide::Func BlackLevelCorrectionHalide::applyBLC_GRBG(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    Halide::Func blc;
    
    // GRBG pattern:
    // G R G R G R
    // B G B G B G
    // G R G R G R
    // B G B G B G
    
    // GR channel: (even, even) positions
    Halide::Expr gr_condition = (x % 2 == 0) && (y % 2 == 0);
    Halide::Expr gr_value = Halide::select(input(x, y) > gr_offset_, 
                                          input(x, y) - gr_offset_, 
                                          0);
    
    // R channel: (even, odd) positions
    Halide::Expr r_condition = (x % 2 == 0) && (y % 2 == 1);
    Halide::Expr r_value = Halide::select(input(x, y) > r_offset_, 
                                         input(x, y) - r_offset_, 
                                         0);
    
    // B channel: (odd, even) positions
    Halide::Expr b_condition = (x % 2 == 1) && (y % 2 == 0);
    Halide::Expr b_value = Halide::select(input(x, y) > b_offset_, 
                                         input(x, y) - b_offset_, 
                                         0);
    
    // GB channel: (odd, odd) positions
    Halide::Expr gb_condition = (x % 2 == 1) && (y % 2 == 1);
    Halide::Expr gb_value = Halide::select(input(x, y) > gb_offset_, 
                                          input(x, y) - gb_offset_, 
                                          0);
    
    // Combine all conditions
    blc(x, y) = Halide::select(gr_condition, gr_value,
                              Halide::select(r_condition, r_value,
                                           Halide::select(b_condition, b_value,
                                                        Halide::select(gb_condition, gb_value,
                                                                     input(x, y)))));
    
    return blc;
}

Halide::Func BlackLevelCorrectionHalide::applyBLC_GBRG(const Halide::Buffer<uint32_t>& input) {
    Halide::Var x, y;
    Halide::Func blc;
    
    // GBRG pattern:
    // G B G B G B
    // R G R G R G
    // G B G B G B
    // R G R G R G
    
    // GB channel: (even, even) positions
    Halide::Expr gb_condition = (x % 2 == 0) && (y % 2 == 0);
    Halide::Expr gb_value = Halide::select(input(x, y) > gb_offset_, 
                                          input(x, y) - gb_offset_, 
                                          0);
    
    // B channel: (even, odd) positions
    Halide::Expr b_condition = (x % 2 == 0) && (y % 2 == 1);
    Halide::Expr b_value = Halide::select(input(x, y) > b_offset_, 
                                         input(x, y) - b_offset_, 
                                         0);
    
    // R channel: (odd, even) positions
    Halide::Expr r_condition = (x % 2 == 1) && (y % 2 == 0);
    Halide::Expr r_value = Halide::select(input(x, y) > r_offset_, 
                                         input(x, y) - r_offset_, 
                                         0);
    
    // GR channel: (odd, odd) positions
    Halide::Expr gr_condition = (x % 2 == 1) && (y % 2 == 1);
    Halide::Expr gr_value = Halide::select(input(x, y) > gr_offset_, 
                                          input(x, y) - gr_offset_, 
                                          0);
    
    // Combine all conditions
    blc(x, y) = Halide::select(gb_condition, gb_value,
                              Halide::select(b_condition, b_value,
                                           Halide::select(r_condition, r_value,
                                                        Halide::select(gr_condition, gr_value,
                                                                     input(x, y)))));
    
    return blc;
}

void BlackLevelCorrectionHalide::printDebugInfo() const {
    std::cout << "BLC Halide - Parameters:" << std::endl;
    std::cout << "  R offset: " << r_offset_ << ", saturation: " << r_sat_ << std::endl;
    std::cout << "  GR offset: " << gr_offset_ << ", saturation: " << gr_sat_ << std::endl;
    std::cout << "  GB offset: " << gb_offset_ << ", saturation: " << gb_sat_ << std::endl;
    std::cout << "  B offset: " << b_offset_ << ", saturation: " << b_sat_ << std::endl;
    std::cout << "  Bayer pattern: " << bayer_pattern_ << std::endl;
    std::cout << "  Bit depth: " << bit_depth_ << ", Max value: " << max_val_ << std::endl;
    std::cout << "  Image size: " << input_.width() << "x" << input_.height() << std::endl;
    std::cout << "  Enabled: " << (enable_ ? "Yes" : "No") << std::endl;
}

bool BlackLevelCorrectionHalide::validateParameters() const {
    // Check if input buffer is valid
    if (!input_.defined()) {
        std::cerr << "Error: Input buffer is not defined" << std::endl;
        return false;
    }
    
    // Check if Bayer pattern is supported
    if (bayer_pattern_ != "rggb" && bayer_pattern_ != "bggr" && 
        bayer_pattern_ != "grbg" && bayer_pattern_ != "gbrg") {
        std::cerr << "Error: Unsupported Bayer pattern: " << bayer_pattern_ << std::endl;
        return false;
    }
    
    // Check bit depth
    if (bit_depth_ <= 0 || bit_depth_ > 32) {
        std::cerr << "Error: Invalid bit depth: " << bit_depth_ << std::endl;
        return false;
    }
    
    // Check if offsets are reasonable
    if (r_offset_ > max_val_ || gr_offset_ > max_val_ || 
        gb_offset_ > max_val_ || b_offset_ > max_val_) {
        std::cerr << "Error: Offset values exceed maximum value" << std::endl;
        return false;
    }
    
    return true;
}

std::string BlackLevelCorrectionHalide::getPerformanceStats() const {
    std::ostringstream oss;
    oss << "Black Level Correction Halide Performance:" << std::endl;
    oss << "  Execution time: " << execution_time_ms_ << " ms" << std::endl;
    oss << "  Image size: " << input_.width() << "x" << input_.height() << std::endl;
    oss << "  Pixels processed: " << (input_.width() * input_.height()) << std::endl;
    oss << "  Pixels per second: " << (input_.width() * input_.height() / (execution_time_ms_ / 1000.0)) << std::endl;
    return oss.str();
}

} // namespace hdr_isp 
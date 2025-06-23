#include "fixed_point_utils.hpp"
#include <cmath>
#include <iostream>

namespace hdr_isp {

FixedPointConfig::FixedPointConfig(const YAML::Node& config) {
    // Default values
    precision_mode_ = FixedPointPrecision::FAST_8BIT;
    fractional_bits_8bit_ = 6;
    fractional_bits_16bit_ = 12;
    enable_fixed_point_ = true;
    
    // Load from config if available
    if (config["fixed_point_config"].IsDefined()) {
        YAML::Node fp_config = config["fixed_point_config"];
        
        // Load precision mode
        if (fp_config["precision_mode"].IsDefined()) {
            std::string mode = fp_config["precision_mode"].as<std::string>();
            if (mode == "8bit") {
                precision_mode_ = FixedPointPrecision::FAST_8BIT;
            } else if (mode == "16bit") {
                precision_mode_ = FixedPointPrecision::PRECISE_16BIT;
            } else {
                std::cout << "Warning: Unknown precision mode '" << mode << "', using 8bit" << std::endl;
            }
        }
        
        // Load fractional bits
        if (fp_config["fractional_bits_8bit"].IsDefined()) {
            fractional_bits_8bit_ = fp_config["fractional_bits_8bit"].as<int>();
            // Clamp to valid range
            fractional_bits_8bit_ = std::max(4, std::min(7, fractional_bits_8bit_));
        }
        
        if (fp_config["fractional_bits_16bit"].IsDefined()) {
            fractional_bits_16bit_ = fp_config["fractional_bits_16bit"].as<int>();
            // Clamp to valid range
            fractional_bits_16bit_ = std::max(8, std::min(14, fractional_bits_16bit_));
        }
        
        // Load enable flag
        if (fp_config["enable_fixed_point"].IsDefined()) {
            enable_fixed_point_ = fp_config["enable_fixed_point"].as<bool>();
        }
    }
    
    // Print configuration
    std::cout << "Fixed-Point Configuration:" << std::endl;
    std::cout << "  Mode: " << getPrecisionModeString() << std::endl;
    std::cout << "  Fractional bits: " << getFractionalBits() << std::endl;
    std::cout << "  Scale factor: " << getScaleFactor() << std::endl;
    std::cout << "  Precision: " << getPrecision() << std::endl;
    std::cout << "  Enabled: " << (enable_fixed_point_ ? "Yes" : "No") << std::endl;
}

int FixedPointConfig::getFractionalBits() const {
    switch (precision_mode_) {
        case FixedPointPrecision::FAST_8BIT:
            return fractional_bits_8bit_;
        case FixedPointPrecision::PRECISE_16BIT:
            return fractional_bits_16bit_;
        default:
            return fractional_bits_8bit_;
    }
}

int FixedPointConfig::getScaleFactor() const {
    return 1 << getFractionalBits();
}

std::string FixedPointConfig::getPrecisionModeString() const {
    switch (precision_mode_) {
        case FixedPointPrecision::FAST_8BIT:
            return "8-bit (Fast)";
        case FixedPointPrecision::PRECISE_16BIT:
            return "16-bit (Precise)";
        default:
            return "Unknown";
    }
}

float FixedPointConfig::getPrecision() const {
    return 1.0f / static_cast<float>(getScaleFactor());
}

} // namespace hdr_isp 
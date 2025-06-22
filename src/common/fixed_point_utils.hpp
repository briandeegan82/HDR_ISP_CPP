#pragma once

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <string>
#include <cstdint>

namespace hdr_isp {

enum class FixedPointPrecision {
    FAST_8BIT = 8,
    PRECISE_16BIT = 16
};

class FixedPointConfig {
public:
    FixedPointConfig(const YAML::Node& config);
    
    // Get precision mode
    FixedPointPrecision getPrecisionMode() const { return precision_mode_; }
    
    // Get fractional bits for current mode
    int getFractionalBits() const;
    
    // Get scale factor for current mode
    int getScaleFactor() const;
    
    // Check if fixed-point is enabled
    bool isEnabled() const { return enable_fixed_point_; }
    
    // Get precision mode as string
    std::string getPrecisionModeString() const;
    
    // Get precision value (e.g., 0.0156 for 6 fractional bits)
    float getPrecision() const;

private:
    FixedPointPrecision precision_mode_;
    int fractional_bits_8bit_;
    int fractional_bits_16bit_;
    bool enable_fixed_point_;
};

// Fixed-point arithmetic utilities
class FixedPointUtils {
public:
    // Convert float to fixed-point
    template<typename T>
    static T floatToFixed(float value, int fractional_bits);
    
    // Convert fixed-point to float
    template<typename T>
    static float fixedToFloat(T value, int fractional_bits);
    
    // Matrix multiplication with fixed-point arithmetic
    template<typename T>
    static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> 
    fixedPointMatrixMultiply(const Eigen::Matrix<T, 3, 3>& matrix,
                            const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& input,
                            int fractional_bits);
    
    // Element-wise multiplication with fixed-point
    template<typename T>
    static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    fixedPointElementWiseMultiply(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& a,
                                 const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b,
                                 int fractional_bits);
    
    // Element-wise addition with fixed-point
    template<typename T>
    static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    fixedPointElementWiseAdd(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& a,
                            const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b);
    
    // Apply fixed-point scaling to a matrix
    template<typename T>
    static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    applyFixedPointScaling(const Eigen::MatrixXf& float_matrix, int fractional_bits);
    
    // Convert fixed-point result back to float with proper rounding
    template<typename T>
    static Eigen::MatrixXf convertFixedToFloat(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& fixed_matrix,
                                              int fractional_bits);
};

// Template implementations
template<typename T>
T FixedPointUtils::floatToFixed(float value, int fractional_bits) {
    int scale_factor = 1 << fractional_bits;
    return static_cast<T>(std::round(value * scale_factor));
}

template<typename T>
float FixedPointUtils::fixedToFloat(T value, int fractional_bits) {
    int scale_factor = 1 << fractional_bits;
    return static_cast<float>(value) / scale_factor;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> 
FixedPointUtils::fixedPointMatrixMultiply(const Eigen::Matrix<T, 3, 3>& matrix,
                                         const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& input,
                                         int fractional_bits) {
    // Use larger intermediate type to prevent overflow
    using IntermediateType = typename std::conditional<sizeof(T) <= 1, int32_t, int64_t>::type;
    
    Eigen::Matrix<IntermediateType, Eigen::Dynamic, Eigen::Dynamic> result = 
        matrix.template cast<IntermediateType>() * input.template cast<IntermediateType>();
    
    // Apply proper rounding
    int half_scale = 1 << (fractional_bits - 1);
    result = (result + half_scale) >> fractional_bits;
    
    return result.template cast<T>();
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
FixedPointUtils::fixedPointElementWiseMultiply(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& a,
                                              const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b,
                                              int fractional_bits) {
    using IntermediateType = typename std::conditional<sizeof(T) <= 1, int32_t, int64_t>::type;
    
    Eigen::Matrix<IntermediateType, Eigen::Dynamic, Eigen::Dynamic> result = 
        a.template cast<IntermediateType>().cwiseProduct(b.template cast<IntermediateType>());
    
    // Apply proper rounding
    int half_scale = 1 << (fractional_bits - 1);
    result = (result + half_scale) >> fractional_bits;
    
    return result.template cast<T>();
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
FixedPointUtils::fixedPointElementWiseAdd(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& a,
                                         const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b) {
    return a + b; // No scaling needed for addition
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
FixedPointUtils::applyFixedPointScaling(const Eigen::MatrixXf& float_matrix, int fractional_bits) {
    int scale_factor = 1 << fractional_bits;
    return (float_matrix * scale_factor).template cast<T>();
}

template<typename T>
Eigen::MatrixXf FixedPointUtils::convertFixedToFloat(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& fixed_matrix,
                                                    int fractional_bits) {
    int scale_factor = 1 << fractional_bits;
    return fixed_matrix.template cast<float>() / scale_factor;
}

} // namespace hdr_isp 
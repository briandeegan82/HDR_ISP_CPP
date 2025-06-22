# Color Correction Matrix Fixed-Point Arithmetic

## Overview

The Color Correction Matrix (CCM) module now supports both floating-point and fixed-point arithmetic. Fixed-point arithmetic can provide better performance on hardware that doesn't have floating-point units, and can be more predictable in terms of precision and timing.

## Configuration

To enable fixed-point arithmetic, add the following parameters to your configuration file:

```yaml
color_correction_matrix:
  is_enable: true
  corrected_red: [1.660, -0.527, -0.133]
  corrected_green: [-0.408, 1.563, -0.082]
  corrected_blue: [-0.055, -1.641, 2.695]
  is_save: false
  # Fixed-point arithmetic options
  use_fixed_point: true    # Set to true to use fixed-point arithmetic
  fixed_point_bits: 16     # Number of fractional bits
```

## Parameters

### `use_fixed_point`
- **Type**: boolean
- **Default**: false
- **Description**: When set to `true`, the module uses fixed-point arithmetic instead of floating-point arithmetic.

### `fixed_point_bits`
- **Type**: integer
- **Default**: 16
- **Description**: Number of fractional bits used for fixed-point representation. Common values:
  - **8 bits**: Lower precision, smaller memory footprint
  - **12 bits**: Good balance for most applications
  - **16 bits**: High precision (recommended)
  - **20 bits**: Very high precision, larger memory footprint

## How It Works

### Fixed-Point Representation
The fixed-point implementation works as follows:

1. **Scaling**: Floating-point CCM matrix values are scaled by `2^fixed_point_bits`
2. **Conversion**: Input image values are also scaled to fixed-point representation
3. **Multiplication**: Matrix multiplication is performed using integer arithmetic
4. **Scaling Back**: Results are scaled back down and converted to floating-point

### Example
For a CCM matrix value of 1.660 with 16 fractional bits:
- Fixed-point representation: `1.660 * 2^16 = 108,789`
- This provides precision of approximately 1/65536 ≈ 0.000015

### Precision vs Performance Trade-offs

| Fractional Bits | Precision | Memory Usage | Performance |
|----------------|-----------|--------------|-------------|
| 8              | ~0.004    | Low          | Fast        |
| 12             | ~0.0002   | Medium       | Medium      |
| 16             | ~0.000015 | Medium       | Medium      |
| 20             | ~0.000001 | High         | Slower      |

## Implementation Details

### Overflow Protection
The implementation uses 64-bit integers for intermediate calculations to prevent overflow during matrix multiplication.

### Rounding
Proper rounding is applied by adding half the scale factor before division to minimize quantization errors.

### Memory Usage
Fixed-point arithmetic requires additional memory for:
- Fixed-point CCM matrix (3x3 integer matrix)
- Intermediate 64-bit calculation buffers

## Comparison with Floating-Point

### Advantages of Fixed-Point
- **Deterministic**: Same input always produces same output
- **Hardware Friendly**: Better suited for FPGAs and DSPs
- **Performance**: Can be faster on systems without FPUs
- **Memory**: Predictable memory usage

### Advantages of Floating-Point
- **Precision**: Higher dynamic range and precision
- **Ease of Use**: No need to manage scaling factors
- **Flexibility**: Easier to work with varying input ranges

## Usage Examples

### Basic Fixed-Point Configuration
```yaml
color_correction_matrix:
  is_enable: true
  corrected_red: [1.660, -0.527, -0.133]
  corrected_green: [-0.408, 1.563, -0.082]
  corrected_blue: [-0.055, -1.641, 2.695]
  use_fixed_point: true
  fixed_point_bits: 16
```

### High Precision Fixed-Point
```yaml
color_correction_matrix:
  is_enable: true
  corrected_red: [1.660, -0.527, -0.133]
  corrected_green: [-0.408, 1.563, -0.082]
  corrected_blue: [-0.055, -1.641, 2.695]
  use_fixed_point: true
  fixed_point_bits: 20
```

### Performance-Optimized Fixed-Point
```yaml
color_correction_matrix:
  is_enable: true
  corrected_red: [1.660, -0.527, -0.133]
  corrected_green: [-0.408, 1.563, -0.082]
  corrected_blue: [-0.055, -1.641, 2.695]
  use_fixed_point: true
  fixed_point_bits: 12
```

## Debug Output

When using fixed-point arithmetic, the module provides detailed debug information:

```
Using fixed-point arithmetic for CCM (fractional bits: 16)
Fixed-point CCM Matrix (scaled by 2^16):
108789  -34539   -8715
-26739   102400   -5374
-3604   -107544   176619
Original floating-point CCM Matrix:
1.660   -0.527   -0.133
-0.408   1.563   -0.082
-0.055   -1.641   2.695
```

## Recommendations

1. **Start with 16 bits**: Provides good precision for most applications
2. **Test with your data**: Verify that the precision is sufficient for your use case
3. **Consider hardware**: Use fixed-point for FPGA/DSP implementations
4. **Profile performance**: Compare execution times between floating-point and fixed-point modes
5. **Validate results**: Ensure the fixed-point results are within acceptable error bounds compared to floating-point 
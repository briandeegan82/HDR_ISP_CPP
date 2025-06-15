#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

namespace common {

// Common structures
struct PlatformInfo {
    std::string name;
    std::string version;
    bool is_hardware;
};

struct SensorInfo {
    int width;
    int height;
    int bit_depth;
    std::string bayer_pattern;
};

// Common constants
constexpr int DEFAULT_BIT_DEPTH = 12;
constexpr float DEFAULT_MEMORY_THRESHOLD = 100.0f;  // MB

// Pipeline configuration structure
struct PipelineConfig {
    std::string config_path;
    std::string data_path;
    std::string filename;
    bool save_intermediate;
    bool use_memory_map;
    float memory_threshold;
};

// Common utility functions
cv::Mat load_raw_image(const std::string& filename, int width, int height, int bit_depth);
void save_image(const cv::Mat& img, const std::string& filename);
std::string get_output_filename(const std::string& input_filename, const std::string& suffix);
PipelineConfig parse_arguments(int argc, char* argv[]);
void create_output_directories(bool save_intermediate);
cv::Mat load_raw_image_with_mmap(const std::string& filename, int width, int height, int bit_depth, bool use_mmap);
void save_intermediate_image(const cv::Mat& img, const std::string& module_name, bool save_intermediate);
YAML::Node load_yaml_config(const std::string& config_path);

// Common error handling
class ISPRuntimeError : public std::runtime_error {
public:
    explicit ISPRuntimeError(const std::string& message) : std::runtime_error(message) {}
};

} // namespace common 
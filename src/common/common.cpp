#include "common.hpp"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <iostream>

namespace fs = std::filesystem;

namespace common {

cv::Mat load_raw_image(const std::string& filename, int width, int height, int bit_depth) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw ISPRuntimeError("Failed to open file: " + filename);
    }

    // Calculate the number of bytes per pixel
    int bytes_per_pixel = (bit_depth + 7) / 8;
    size_t file_size = std::filesystem::file_size(filename);
    size_t expected_size = width * height * bytes_per_pixel;

    if (file_size != expected_size) {
        throw ISPRuntimeError("File size mismatch for: " + filename);
    }

    // Create a matrix to hold the raw data
    cv::Mat raw_data(height, width, CV_16UC1);
    
    // Read the raw data
    file.read(reinterpret_cast<char*>(raw_data.data), file_size);
    
    if (!file) {
        throw ISPRuntimeError("Error reading file: " + filename);
    }

    return raw_data;
}

void save_image(const cv::Mat& img, const std::string& filename) {
    // Create output directory if it doesn't exist
    std::filesystem::path output_path(filename);
    std::filesystem::create_directories(output_path.parent_path());

    // Save the image
    if (!cv::imwrite(filename, img)) {
        throw ISPRuntimeError("Failed to save image: " + filename);
    }
}

std::string get_output_filename(const std::string& input_filename, const std::string& suffix) {
    std::filesystem::path path(input_filename);
    std::string stem = path.stem().string();
    std::string extension = path.extension().string();
    
    return stem + "_" + suffix + extension;
}

PipelineConfig parse_arguments(int argc, char* argv[]) {
    PipelineConfig config;
    
    // Default values
    config.config_path = fs::path("./config/svs_cam.yml").make_preferred().string();
    config.data_path = fs::path("./in_frames/normal").make_preferred().string();
    config.filename = "ColorChecker_2592x1536_12bits_RGGB.raw";
    config.save_intermediate = false;
    config.use_memory_map = false;
    config.memory_threshold = DEFAULT_MEMORY_THRESHOLD;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config.config_path = fs::path(argv[++i]).make_preferred().string();
        }
        else if (arg == "--data" && i + 1 < argc) {
            config.data_path = fs::path(argv[++i]).make_preferred().string();
        }
        else if (arg == "--file" && i + 1 < argc) {
            config.filename = argv[++i];
        }
        else if (arg == "--save-intermediate") {
            config.save_intermediate = true;
        }
        else if (arg == "--use-mmap") {
            config.use_memory_map = true;
        }
        else if (arg == "--memory-threshold" && i + 1 < argc) {
            config.memory_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                     << "Options:\n"
                     << "  --config <path>           Path to configuration file\n"
                     << "  --data <path>             Path to data directory\n"
                     << "  --file <filename>         Input filename\n"
                     << "  --save-intermediate       Save intermediate results\n"
                     << "  --use-mmap               Use memory mapping for large files\n"
                     << "  --memory-threshold <MB>   Memory threshold in MB\n"
                     << "  --help                   Show this help message\n";
            exit(0);
        }
    }

    return config;
}

void create_output_directories(bool save_intermediate) {
    std::filesystem::create_directories("out_frames");
    if (save_intermediate) {
        std::filesystem::create_directories("out_frames/intermediate");
    }
}

cv::Mat load_raw_image_with_mmap(const std::string& filename, int width, int height, int bit_depth, bool use_mmap) {
    std::filesystem::path path_object(filename);
    size_t file_size = std::filesystem::file_size(path_object);
    bool should_use_mmap = use_mmap && (file_size > DEFAULT_MEMORY_THRESHOLD * 1024 * 1024);

    if (should_use_mmap) {
        // TODO: Implement memory mapping
        throw ISPRuntimeError("Memory mapping not implemented yet");
    }

    if (path_object.extension() == ".raw") {
        return load_raw_image(filename, width, height, bit_depth);
    }
    else if (path_object.extension() == ".tiff") {
        cv::Mat raw = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if (raw.channels() == 3) {
            std::vector<cv::Mat> channels;
            cv::split(raw, channels);
            return channels[0];
        }
        return raw;
    }
    else {
        return cv::imread(filename, cv::IMREAD_UNCHANGED);
    }
}

void save_intermediate_image(const cv::Mat& img, const std::string& module_name, bool save_intermediate) {
    if (save_intermediate) {
        std::filesystem::path output_path = std::filesystem::path("out_frames/intermediate") / (module_name + ".png");
        save_image(img, output_path.string());
    }
}

YAML::Node load_yaml_config(const std::string& config_path) {
    try {
        return YAML::LoadFile(config_path);
    }
    catch (const std::exception& e) {
        throw ISPRuntimeError("Error loading config: " + std::string(e.what()));
    }
}

} // namespace common 
#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "infinite_isp.hpp"

namespace fs = std::filesystem;

struct PipelineConfig {
    std::string config_path;
    std::string data_path;
    std::string filename;
    bool save_intermediate;
    bool use_memory_map;
    float memory_threshold;
};

PipelineConfig parse_arguments(int argc, char* argv[]) {
    PipelineConfig config;
    
    // Default values
    config.config_path = "./config/svs_cam.yml";
    config.data_path = "./in_frames/normal";
    config.filename = "ColorChecker_2592x1536_12bits_RGGB.raw";
    config.save_intermediate = false;
    config.use_memory_map = false;
    config.memory_threshold = 100.0f;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config.config_path = argv[++i];
        }
        else if (arg == "--data" && i + 1 < argc) {
            config.data_path = argv[++i];
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

    // Print configuration
    std::cout << "HDR ISP C++ Pipeline Configuration:" << std::endl;
    std::cout << "Config Path: " << config.config_path << std::endl;
    std::cout << "Data Path: " << config.data_path << std::endl;
    std::cout << "Filename: " << config.filename << std::endl;
    std::cout << "Save Intermediate: " << (config.save_intermediate ? "Yes" : "No") << std::endl;
    std::cout << "Use Memory Map: " << (config.use_memory_map ? "Yes" : "No") << std::endl;
    std::cout << "Memory Threshold: " << config.memory_threshold << " MB" << std::endl;

    return config;
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        PipelineConfig config = parse_arguments(argc, argv);

        // Check if input file exists
        fs::path input_file = fs::path(config.data_path) / config.filename;
        if (!fs::exists(input_file)) {
            std::cerr << "Error: Input file not found: " << input_file << std::endl;
            return 1;
        }

        // Check if config file exists
        if (!fs::exists(config.config_path)) {
            std::cerr << "Error: Configuration file not found: " << config.config_path << std::endl;
            return 1;
        }

        // Create output directories
        fs::create_directories("out_frames");
        if (config.save_intermediate) {
            fs::create_directories("out_frames/intermediate");
        }

        // Initialize and run the ISP pipeline
        std::cout << "Starting HDR ISP Pipeline..." << std::endl;
        
        InfiniteISP isp(config.data_path, config.config_path);
        isp.execute(config.filename, config.save_intermediate);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 
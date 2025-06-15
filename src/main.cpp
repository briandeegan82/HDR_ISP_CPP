#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "infinite_isp.hpp"
#include "common/common.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        common::PipelineConfig config = common::parse_arguments(argc, argv);

        // Check if input file exists
        fs::path input_file = fs::path(config.data_path).make_preferred() / config.filename;
        if (!fs::exists(input_file)) {
            std::cerr << "Error: Input file not found: " << input_file.string() << std::endl;
            return 1;
        }

        // Check if config file exists
        fs::path config_path = fs::path(config.config_path).make_preferred();
        if (!fs::exists(config_path)) {
            std::cerr << "Error: Configuration file not found: " << config_path.string() << std::endl;
            return 1;
        }

        // Create output directories
        common::create_output_directories(config.save_intermediate);

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
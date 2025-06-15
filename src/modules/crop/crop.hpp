#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

class Crop {
public:
    Crop(const cv::Mat& img, const YAML::Node& platform, 
         const YAML::Node& sensor_info, const YAML::Node& parm_cro);

    cv::Mat execute();

private:
    void update_sensor_info(YAML::Node& dictionary);
    cv::Mat crop(const cv::Mat& img, int rows_to_crop, int cols_to_crop);
    cv::Mat apply_cropping();
    void save(const std::string& filename_tag);

    cv::Mat img_;
    YAML::Node platform_;
    YAML::Node sensor_info_;
    YAML::Node parm_cro_;
    std::pair<int, int> old_size_;
    std::pair<int, int> new_size_;
    bool enable_;
    bool is_debug_;
    bool is_save_;
}; 
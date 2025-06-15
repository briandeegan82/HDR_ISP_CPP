#include "pca.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>

PCAIlluminEstimation::PCAIlluminEstimation(const cv::Mat& flatten_img, float pixel_percentage)
    : flatten_img_(flatten_img)
    , pixel_percentage_(pixel_percentage) {
}

std::tuple<double, double> PCAIlluminEstimation::calculate_gains() {
    // Convert to float for calculations
    cv::Mat flat_img;
    flatten_img_.convertTo(flat_img, CV_32F);
    
    // Calculate mean RGB vector
    cv::Scalar mean_rgb = cv::mean(flat_img);
    cv::Mat mean_vector = cv::Mat(mean_rgb).reshape(1, 3);
    double norm = cv::norm(mean_vector);
    mean_vector /= norm;
    
    // Calculate projected distances
    std::vector<float> data_p(flat_img.rows);
    for (int i = 0; i < flat_img.rows; ++i) {
        data_p[i] = 0;
        for (int j = 0; j < 3; ++j) {
            data_p[i] += flat_img.at<float>(i, j) * mean_vector.at<float>(j, 0);
        }
    }
    
    // Sort indices based on projected distances
    std::vector<int> indices(flat_img.rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&data_p](int i1, int i2) { return data_p[i1] < data_p[i2]; });
    
    // Select dark and light pixels
    int index = std::ceil(flat_img.rows * (pixel_percentage_ / 100.0f));
    std::vector<int> filtered_indices;
    filtered_indices.insert(filtered_indices.end(), indices.begin(), indices.begin() + index);
    filtered_indices.insert(filtered_indices.end(), indices.end() - index, indices.end());
    
    // Create filtered data matrix
    cv::Mat filtered_data(filtered_indices.size(), 3, CV_32F);
    for (size_t i = 0; i < filtered_indices.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            filtered_data.at<float>(i, j) = flat_img.at<float>(filtered_indices[i], j);
        }
    }
    
    // Calculate covariance matrix
    cv::Mat sigma = filtered_data.t() * filtered_data;
    
    // Calculate eigenvalues and eigenvectors
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(sigma, eigenvalues, eigenvectors);
    
    // Get eigenvector with maximum eigenvalue
    cv::Mat avg_rgb = cv::abs(eigenvectors.row(2));
    
    // Calculate white balance gains
    double rgain = std::isnan(avg_rgb.at<float>(0, 1) / avg_rgb.at<float>(0, 0)) ? 0.0 : 
                   avg_rgb.at<float>(0, 1) / avg_rgb.at<float>(0, 0);
    double bgain = std::isnan(avg_rgb.at<float>(0, 1) / avg_rgb.at<float>(0, 2)) ? 0.0 : 
                   avg_rgb.at<float>(0, 1) / avg_rgb.at<float>(0, 2);
    
    return {rgain, bgain};
} 
cmake_minimum_required(VERSION 3.16)
project(bayer_noise_reduction_halide_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable hybrid backend
option(USE_HYBRID_BACKEND "Enable hybrid backend optimization" ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

if(USE_HYBRID_BACKEND)
    find_package(Halide REQUIRED)
    add_compile_definitions(USE_HYBRID_BACKEND)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Create common library
add_library(hdr_isp_common
    src/common/eigen_utils.cpp
)

target_include_directories(hdr_isp_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(hdr_isp_common
    PUBLIC
        ${OpenCV_LIBS}
        yaml-cpp::yaml-cpp
)

# Create Bayer Noise Reduction Halide library
add_library(bayer_noise_reduction_halide
    src/modules/bayer_noise_reduction/bayer_noise_reduction_halide.cpp
)

target_include_directories(bayer_noise_reduction_halide
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(bayer_noise_reduction_halide
    PUBLIC
        hdr_isp_common
        ${OpenCV_LIBS}
        yaml-cpp::yaml-cpp
        ${HALIDE_LIBRARIES}
)

# Create test executable
add_executable(test_bayer_noise_reduction_halide
    test_bayer_noise_reduction_halide.cpp
)

target_link_libraries(test_bayer_noise_reduction_halide
    bayer_noise_reduction_halide
    hdr_isp_common
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    ${HALIDE_LIBRARIES}
) 
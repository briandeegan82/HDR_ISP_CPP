cmake_minimum_required(VERSION 3.10)
project(HDR_ISP_CPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(LibRaw REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(GSL REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${LibRaw_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${FFTW3_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIRS}
)

# Add source directory
add_subdirectory(src)

# Create the main library
add_library(hdr_isp_common
    src/common/image_processing.cpp
    src/common/image_processing.hpp
)

# Link dependencies to the main library
target_link_libraries(hdr_isp_common
    PUBLIC
        ${OpenCV_LIBS}
        Eigen3::Eigen
        ${LibRaw_LIBRARIES}
        yaml-cpp
        ${FFTW3_LIBRARIES}
        ${GSL_LIBRARIES}
) 
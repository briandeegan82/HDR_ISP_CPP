cmake_minimum_required(VERSION 3.10)

# Add vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project(HDR_ISP_CPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define project root directory for use in C++ code
add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

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

# Create executable
add_executable(hdr_isp_pipeline src/main.cpp)

# Link dependencies to the executable
target_link_libraries(hdr_isp_pipeline
    PRIVATE
        common
        infinite_isp
        isp_modules
        ${OpenCV_LIBS}
        Eigen3::Eigen
        ${LibRaw_LIBRARIES}
        yaml-cpp::yaml-cpp
        FFTW3::fftw3
        ${GSL_LIBRARIES}
) 
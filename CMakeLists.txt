cmake_minimum_required(VERSION 3.10)

# Add vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project(HDR_ISP_CPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define project root directory for use in C++ code
add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

# Optional hybrid backend support
option(USE_HYBRID_BACKEND "Enable hybrid backend with Halide support" OFF)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(LibRaw REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(GSL REQUIRED)

# Optional Halide package
if(USE_HYBRID_BACKEND)
    find_package(Halide QUIET)
    if(Halide_FOUND)
        add_compile_definitions(USE_HYBRID_BACKEND)
        message(STATUS "Halide found - hybrid backend enabled")
    else()
        message(WARNING "Halide not found - hybrid backend disabled")
        set(USE_HYBRID_BACKEND OFF)
    endif()
endif()

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

# Add Halide include directories if available
if(USE_HYBRID_BACKEND AND Halide_FOUND)
    include_directories(${HALIDE_INCLUDE_DIRS})
endif()

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

# Add Halide libraries if available
if(USE_HYBRID_BACKEND AND Halide_FOUND)
    target_link_libraries(hdr_isp_pipeline PRIVATE ${HALIDE_LIBRARIES})
endif() 
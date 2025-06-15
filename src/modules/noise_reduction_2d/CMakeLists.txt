# 2D Noise Reduction Module
add_library(noise_reduction_2d
    noise_reduction_2d.cpp
    noise_reduction_2d.hpp
)

target_include_directories(noise_reduction_2d
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(noise_reduction_2d
    PUBLIC
        hdr_isp_common
) 
# 2D Noise Reduction Module
add_library(2d_noise_reduction
    2d_noise_reduction.cpp
    2d_noise_reduction.hpp
)

target_include_directories(2d_noise_reduction
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(2d_noise_reduction
    PUBLIC
        common
) 
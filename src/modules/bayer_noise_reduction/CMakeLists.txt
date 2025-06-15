# Bayer Noise Reduction Module
add_library(bayer_noise_reduction
    bayer_noise_reduction.cpp
    bayer_noise_reduction.hpp
)

target_include_directories(bayer_noise_reduction
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(bayer_noise_reduction
    PUBLIC
        hdr_isp_common
        ${OpenCV_LIBS}
        yaml-cpp::yaml-cpp
) 
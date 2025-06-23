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

# Add Halide version if hybrid backend is enabled
if(USE_HYBRID_BACKEND)
    add_library(bayer_noise_reduction_halide
        bayer_noise_reduction_halide.cpp
        bayer_noise_reduction_halide.hpp
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
    
    target_compile_definitions(bayer_noise_reduction_halide
        PRIVATE
            USE_HYBRID_BACKEND
    )
endif() 
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

# Add Halide hybrid version if hybrid backend is enabled
if(USE_HYBRID_BACKEND)
    add_library(2d_noise_reduction_hybrid
        2d_noise_reduction_hybrid.cpp
        2d_noise_reduction_hybrid.hpp
    )

    target_include_directories(2d_noise_reduction_hybrid
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(2d_noise_reduction_hybrid
        PUBLIC
            common
            ${HALIDE_LIBRARIES}
    )
endif() 
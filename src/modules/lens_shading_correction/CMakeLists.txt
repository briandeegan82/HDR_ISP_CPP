# Lens Shading Correction Module
add_library(lens_shading_correction
    lens_shading_correction.cpp
    lens_shading_correction.hpp
)

target_include_directories(lens_shading_correction
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(lens_shading_correction
    PUBLIC
        hdr_isp_common
)

# Add Halide version if hybrid backend is enabled
if(USE_HYBRID_BACKEND)
    add_library(lens_shading_correction_halide
        lens_shading_correction_halide.cpp
        lens_shading_correction_halide.hpp
    )

    target_include_directories(lens_shading_correction_halide
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(lens_shading_correction_halide
        PUBLIC
            hdr_isp_common
            ${HALIDE_LIBRARIES}
    )
endif() 
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
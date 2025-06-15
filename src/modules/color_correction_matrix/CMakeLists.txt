# Color Correction Matrix Module
add_library(color_correction_matrix
    color_correction_matrix.cpp
    color_correction_matrix.hpp
)

target_include_directories(color_correction_matrix
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(color_correction_matrix
    PUBLIC
        hdr_isp_common
) 
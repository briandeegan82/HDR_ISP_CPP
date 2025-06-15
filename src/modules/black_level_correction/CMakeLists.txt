# Black Level Correction Module
add_library(black_level_correction
    black_level_correction.cpp
    black_level_correction.hpp
)

target_include_directories(black_level_correction
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(black_level_correction
    PUBLIC
        hdr_isp_common
) 
# Dead Pixel Correction Module
add_library(dead_pixel_correction
    dead_pixel_correction.cpp
    dead_pixel_correction.hpp
)

target_include_directories(dead_pixel_correction
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(dead_pixel_correction
    PUBLIC
        common
) 
# RGB Conversion Module
add_library(rgb_conversion
    rgb_conversion.cpp
    rgb_conversion.hpp
)

target_include_directories(rgb_conversion
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rgb_conversion
    PUBLIC
        common
) 
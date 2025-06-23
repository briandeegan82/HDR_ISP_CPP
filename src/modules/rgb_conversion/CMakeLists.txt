# RGB Conversion Module
set(RGB_CONVERSION_SOURCES
    rgb_conversion.cpp
    rgb_conversion.hpp
)

# Add hybrid implementation if enabled
if(USE_HYBRID_BACKEND AND Halide_FOUND)
    list(APPEND RGB_CONVERSION_SOURCES 
        rgb_conversion_hybrid.cpp
        rgb_conversion_hybrid.hpp
    )
endif()

add_library(rgb_conversion ${RGB_CONVERSION_SOURCES})

target_include_directories(rgb_conversion
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rgb_conversion
    PUBLIC
        common
) 
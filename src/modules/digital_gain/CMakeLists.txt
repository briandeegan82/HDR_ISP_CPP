# Digital Gain Module
add_library(digital_gain
    digital_gain.cpp
    digital_gain.hpp
)

target_include_directories(digital_gain
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(digital_gain
    PUBLIC
        common
) 
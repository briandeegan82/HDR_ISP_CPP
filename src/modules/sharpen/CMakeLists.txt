# Sharpen Module
add_library(sharpen
    sharpen.cpp
    sharpen.hpp
)

target_include_directories(sharpen
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sharpen
    PUBLIC
        hdr_isp_common
) 
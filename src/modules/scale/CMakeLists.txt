# Scale Module
add_library(scale
    scale.cpp
    scale.hpp
)

target_include_directories(scale
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(scale
    PUBLIC
        hdr_isp_common
) 
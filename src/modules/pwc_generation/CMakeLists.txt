# PWC Generation Module
add_library(pwc_generation
    pwc_generation.cpp
    pwc_generation.hpp
)

target_include_directories(pwc_generation
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pwc_generation
    PUBLIC
        hdr_isp_common
) 
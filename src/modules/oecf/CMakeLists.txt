# OECF Module
add_library(oecf
    oecf.cpp
    oecf.hpp
)

target_include_directories(oecf
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(oecf
    PUBLIC
        hdr_isp_common
) 
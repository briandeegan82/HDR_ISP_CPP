# YUV Conversion Format Module
add_library(yuv_conv_format
    yuv_conv_format.cpp
    yuv_conv_format.hpp
)

target_include_directories(yuv_conv_format
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(yuv_conv_format
    PUBLIC
        hdr_isp_common
) 
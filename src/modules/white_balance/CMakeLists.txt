# White Balance Module
add_library(white_balance
    white_balance.cpp
    white_balance.hpp
)

target_include_directories(white_balance
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(white_balance
    PUBLIC
        hdr_isp_common
) 
# Auto White Balance Module
add_library(auto_white_balance
    auto_white_balance.cpp
    auto_white_balance.hpp
    gray_world.cpp
    gray_world.hpp
    norm_gray_world.cpp
    norm_gray_world.hpp
    pca.cpp
    pca.hpp
)

target_include_directories(auto_white_balance
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(auto_white_balance
    PUBLIC
        hdr_isp_common
) 
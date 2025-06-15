# Auto Exposure Module
add_library(auto_exposure
    auto_exposure.cpp
    auto_exposure.hpp
)

target_include_directories(auto_exposure
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(auto_exposure
    PUBLIC
        hdr_isp_common
) 
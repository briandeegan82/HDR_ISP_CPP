# Auto Exposure Module
add_library(auto_exposure
    auto_exposure.cpp
    auto_exposure.hpp
)

target_include_directories(auto_exposure
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(auto_exposure
    PUBLIC
        hdr_isp_common
        ${OpenCV_LIBS}
        yaml-cpp::yaml-cpp
) 
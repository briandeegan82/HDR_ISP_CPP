# Add modules directory
add_subdirectory(modules)

# Add common directory
add_subdirectory(common)

# Add infinite_isp library
add_library(infinite_isp
    infinite_isp.cpp
)

target_include_directories(infinite_isp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(infinite_isp
    PUBLIC
        common
        ${OpenCV_LIBS}
        yaml-cpp::yaml-cpp
) 
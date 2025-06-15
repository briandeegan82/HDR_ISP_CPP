# Crop Module
add_library(crop
    crop.cpp
    crop.hpp
)

target_include_directories(crop
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(crop
    PUBLIC
        common
) 
add_library(color_space_conversion
    color_space_conversion.cpp
)

target_include_directories(color_space_conversion
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(color_space_conversion
    PUBLIC
        ${OpenCV_LIBS}
) 
set(CSC_SOURCES
    color_space_conversion.cpp
)

if(USE_HYBRID_BACKEND AND Halide_FOUND)
    list(APPEND CSC_SOURCES color_space_conversion_hybrid.cpp color_space_conversion_hybrid.hpp)
endif()

add_library(color_space_conversion ${CSC_SOURCES})

target_include_directories(color_space_conversion
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(color_space_conversion
    PUBLIC
        ${OpenCV_LIBS}
) 
# Digital Gain Module
add_library(digital_gain
    digital_gain.cpp
    digital_gain.hpp
)

target_include_directories(digital_gain
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(digital_gain
    PUBLIC
        common
)

# Add Halide Digital Gain when hybrid backend is enabled
if(USE_HYBRID_BACKEND)
    add_library(digital_gain_halide
        digital_gain_halide.cpp
        digital_gain_halide.hpp
    )

    target_include_directories(digital_gain_halide
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(digital_gain_halide
        PUBLIC
            common
            halide_utils
    )

    if(Halide_FOUND)
        target_link_libraries(digital_gain_halide
            PRIVATE
                ${HALIDE_LIBRARIES}
        )
    endif()
endif() 
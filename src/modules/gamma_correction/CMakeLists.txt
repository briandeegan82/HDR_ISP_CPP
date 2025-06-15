# Gamma Correction Module
add_library(gamma_correction
    gamma_correction.cpp
    gamma_correction.hpp
)

target_include_directories(gamma_correction
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(gamma_correction
    PUBLIC
        common
) 
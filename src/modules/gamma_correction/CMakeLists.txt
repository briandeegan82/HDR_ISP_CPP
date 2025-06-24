# Gamma Correction Module
set(GAMMA_CORRECTION_SOURCES
    gamma_correction.cpp
    gamma_correction.hpp
)

# Add hybrid implementation if enabled
if(USE_HYBRID_BACKEND AND Halide_FOUND)
    list(APPEND GAMMA_CORRECTION_SOURCES 
        gamma_correction_hybrid.cpp
        gamma_correction_hybrid.hpp
    )
endif()

add_library(gamma_correction ${GAMMA_CORRECTION_SOURCES})

target_include_directories(gamma_correction
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(gamma_correction
    PUBLIC
        common
)

# Add Halide libraries if available
if(USE_HYBRID_BACKEND AND Halide_FOUND)
    target_link_libraries(gamma_correction
        PRIVATE
            ${HALIDE_LIBRARIES}
    )
    
    target_compile_definitions(gamma_correction
        PRIVATE
            USE_HYBRID_BACKEND
    )
endif() 
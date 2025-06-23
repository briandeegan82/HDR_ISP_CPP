# Black Level Correction Module
add_library(black_level_correction
    black_level_correction.cpp
    black_level_correction.hpp
)

# Add Halide implementation if hybrid backend is enabled
if(USE_HYBRID_BACKEND)
    target_sources(black_level_correction
        PRIVATE
            black_level_correction_halide.cpp
            black_level_correction_halide.hpp
    )
    
    target_compile_definitions(black_level_correction
        PRIVATE
            USE_HYBRID_BACKEND
    )
    
    target_link_libraries(black_level_correction
        PRIVATE
            ${HALIDE_LIBRARIES}
    )
endif()

target_include_directories(black_level_correction
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(black_level_correction
    PUBLIC
        hdr_isp_common
) 
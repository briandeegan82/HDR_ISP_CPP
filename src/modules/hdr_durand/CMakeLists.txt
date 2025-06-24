add_library(hdr_durand
    hdr_durand.cpp
    hdr_durand_hybrid.cpp
)

target_include_directories(hdr_durand
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(hdr_durand
    PUBLIC
        ${OpenCV_LIBS}
)

# Add Halide support if hybrid backend is enabled
if(USE_HYBRID_BACKEND)
    target_link_libraries(hdr_durand
        PUBLIC
            ${HALIDE_LIBRARIES}
    )
    
    target_compile_definitions(hdr_durand
        PUBLIC
            USE_HYBRID_BACKEND
    )
endif() 
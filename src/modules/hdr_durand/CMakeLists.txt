add_library(hdr_durand
    hdr_durand.cpp
)

target_include_directories(hdr_durand
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(hdr_durand
    PUBLIC
        ${OpenCV_LIBS}
) 
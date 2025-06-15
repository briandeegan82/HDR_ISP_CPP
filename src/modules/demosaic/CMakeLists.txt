add_library(demosaic
    demosaic.cpp
)

target_include_directories(demosaic
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(demosaic
    PUBLIC
        ${OpenCV_LIBS}
) 
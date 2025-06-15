add_library(ldci
    ldci.cpp
)

target_include_directories(ldci
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(ldci
    PUBLIC
        ${OpenCV_LIBS}
) 
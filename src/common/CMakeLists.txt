set(COMMON_SOURCES
    common.cpp
)

set(COMMON_HEADERS
    common.hpp
)

add_library(common STATIC ${COMMON_SOURCES} ${COMMON_HEADERS})

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(common
    PRIVATE
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
) 
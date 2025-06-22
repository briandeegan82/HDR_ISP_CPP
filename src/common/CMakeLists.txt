set(COMMON_SOURCES
    common.cpp
    eigen_utils.cpp
    fixed_point_utils.cpp
)

set(COMMON_HEADERS
    common.hpp
    eigen_utils.hpp
    fixed_point_utils.hpp
)

add_library(common STATIC ${COMMON_SOURCES} ${COMMON_HEADERS})

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(common
    PRIVATE
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    Eigen3::Eigen
) 
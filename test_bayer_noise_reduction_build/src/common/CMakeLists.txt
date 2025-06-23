set(COMMON_SOURCES
    common.cpp
    eigen_utils.cpp
    fixed_point_utils.cpp
    isp_backend_wrapper.cpp
    module_ab_test.cpp
)

set(COMMON_HEADERS
    common.hpp
    eigen_utils.hpp
    fixed_point_utils.hpp
    ../../include/isp_backend_wrapper.hpp
    ../../include/module_ab_test.hpp
)

# Add hybrid backend sources if enabled
if(USE_HYBRID_BACKEND AND Halide_FOUND)
    list(APPEND COMMON_SOURCES 
        hybrid_backend.cpp
        halide_utils.cpp
    )
    list(APPEND COMMON_HEADERS 
        ../../include/hybrid_backend.hpp
        halide_utils.hpp
    )
endif()

add_library(common STATIC ${COMMON_SOURCES} ${COMMON_HEADERS})

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(common
    PRIVATE
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    Eigen3::Eigen
)

# Add Halide libraries if available
if(USE_HYBRID_BACKEND AND Halide_FOUND)
    target_link_libraries(common PRIVATE ${HALIDE_LIBRARIES})
endif() 
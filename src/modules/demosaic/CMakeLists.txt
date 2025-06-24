# Demosaic Module
add_library(demosaic
    demosaic.cpp
    demosaic.hpp
)

# Add hybrid version if hybrid backend is enabled
if(USE_HYBRID_BACKEND)
    target_sources(demosaic
        PRIVATE
            demosaic_hybrid.cpp
            demosaic_hybrid.hpp
    )
endif()

target_include_directories(demosaic
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(demosaic
    PUBLIC
        hdr_isp_common
) 
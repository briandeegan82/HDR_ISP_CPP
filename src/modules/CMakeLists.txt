# Add subdirectories for each module
add_subdirectory(2d_noise_reduction)
add_subdirectory(auto_exposure)
add_subdirectory(auto_white_balance)
add_subdirectory(bayer_noise_reduction)
add_subdirectory(black_level_correction)
add_subdirectory(color_correction_matrix)
add_subdirectory(color_space_conversion)
add_subdirectory(crop)
add_subdirectory(dead_pixel_correction)
add_subdirectory(demosaic)
add_subdirectory(digital_gain)
add_subdirectory(gamma_correction)
add_subdirectory(hdr_durand)
add_subdirectory(ldci)
add_subdirectory(lens_shading_correction)
add_subdirectory(oecf)
add_subdirectory(pwc_generation)
add_subdirectory(rgb_conversion)
add_subdirectory(scale)
add_subdirectory(sharpen)
add_subdirectory(white_balance)
add_subdirectory(yuv_conv_format)

# Create a library containing all modules
add_library(isp_modules STATIC
    $<TARGET_OBJECTS:2d_noise_reduction>
    $<TARGET_OBJECTS:auto_exposure>
    $<TARGET_OBJECTS:auto_white_balance>
    $<TARGET_OBJECTS:bayer_noise_reduction>
    $<TARGET_OBJECTS:black_level_correction>
    $<TARGET_OBJECTS:color_correction_matrix>
    $<TARGET_OBJECTS:color_space_conversion>
    $<TARGET_OBJECTS:crop>
    $<TARGET_OBJECTS:dead_pixel_correction>
    $<TARGET_OBJECTS:demosaic>
    $<TARGET_OBJECTS:digital_gain>
    $<TARGET_OBJECTS:gamma_correction>
    $<TARGET_OBJECTS:hdr_durand>
    $<TARGET_OBJECTS:ldci>
    $<TARGET_OBJECTS:lens_shading_correction>
    $<TARGET_OBJECTS:oecf>
    $<TARGET_OBJECTS:pwc_generation>
    $<TARGET_OBJECTS:rgb_conversion>
    $<TARGET_OBJECTS:scale>
    $<TARGET_OBJECTS:sharpen>
    $<TARGET_OBJECTS:white_balance>
    $<TARGET_OBJECTS:yuv_conv_format>
)

# Add Halide modules if hybrid backend is enabled
if(USE_HYBRID_BACKEND)
    target_link_libraries(isp_modules
        PRIVATE
            ${OpenCV_LIBS}
            yaml-cpp::yaml-cpp
            lens_shading_correction_halide
            black_level_correction_halide
            digital_gain_halide
            bayer_noise_reduction_halide
            rgb_conversion_hybrid
            color_space_conversion_hybrid
    )
else()
    # Link OpenCV and other dependencies to the modules library
    target_link_libraries(isp_modules
        PRIVATE
            ${OpenCV_LIBS}
            yaml-cpp::yaml-cpp
    )
endif() 
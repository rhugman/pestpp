# pestpp_dsi: native C++ DSI surrogate for pestpp-ies lambda testing.
# This library is intentionally decoupled from pestpp_common so that DSI
# code can be unit-tested without dragging the ensemble-method stack.
# Cycle-free dependency: pestpp_common -> pestpp_dsi (not the other way).

add_library(pestpp_dsi STATIC
  DsiTransforms.cpp
  DsiTransformPipeline.cpp
  DsiRowWiseScaler.cpp
  DsiTrainingStore.cpp
  DsiEmulator.cpp
)

target_include_directories(pestpp_dsi
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(pestpp_dsi PUBLIC Eigen)

target_compile_options(pestpp_dsi PRIVATE ${PESTPP_CXX_WARN_FLAGS})

if(BUILD_SHARED_LIBS)
  set_property(TARGET pestpp_dsi PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

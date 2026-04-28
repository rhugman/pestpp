# pestpp_dsi unit tests.
#
# Each test is a standalone executable returning 0 on success. We do
# not pull in a test framework dependency (Catch2/gtest) to keep the
# upstream-merge cost minimal — see plan §11.8.

set(_dsi_tests
  test_log10_roundtrip
  test_pipeline
  test_training_store
  test_pmat_orientation
  test_dsi_fit_predict
  test_dsi_energy_truncation
  test_dsi_with_log10
  test_normal_score
  test_normal_score_clustered_boundary
  test_rowwise_scaler
  test_phi_drop_gate
  test_active_refine_stop
)

foreach(_t IN LISTS _dsi_tests)
  add_executable(${_t} ${_t}.cpp)
  target_link_libraries(${_t} PRIVATE pestpp_dsi)
  target_include_directories(${_t} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME ${_t} COMMAND ${_t})
  set_tests_properties(${_t} PROPERTIES LABELS "dsi")
endforeach()

# Phase 2 golden test against pyemu. Reads artefacts produced by the
# Python oracle at tests/golden/run_python_oracle.py. The artefact
# directory is configurable; default lives in the build tree. The test
# returns 77 (CTest skip code) if artefacts are missing — that lets
# devs without pyemu run the rest of the suite without false fails.
set(DSI_GOLDEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/golden_artifacts"
    CACHE PATH "Directory holding pyemu-trained DSI golden artefacts.")
add_executable(test_golden_pyemu test_golden_pyemu.cpp)
target_link_libraries(test_golden_pyemu PRIVATE pestpp_dsi)
target_include_directories(test_golden_pyemu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME test_golden_pyemu
         COMMAND test_golden_pyemu "${DSI_GOLDEN_DIR}")
set_tests_properties(test_golden_pyemu PROPERTIES
  LABELS "dsi"
  SKIP_RETURN_CODE 77)

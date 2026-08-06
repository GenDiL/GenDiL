gendil_add_target_compile_failure_test(
  test-sgbsr-rejects-rectangular-spaces
  rectangular_sgbsr.cpp
  "SGBSR GenericAssembly currently requires matching trial/test FE spaces"
)

gendil_add_target_compile_failure_test(
  test-element-sgbsr-rejects-rectangular-spaces
  rectangular_sgbsr.cpp
  "SGBSR element block-diagonal assembly currently requires matching trial/test FE spaces"
)
target_compile_definitions(
  test-element-sgbsr-rejects-rectangular-spaces-probe
  PRIVATE
  GENDIL_TEST_ELEMENT_SGBSR
)

gendil_add_target_compile_failure_test(
  test-cusparse-bsr-rejects-rectangular-spaces
  rectangular_vendor_bsr.cpp
  "Rectangular BSR assembly requires HostBSRBackend or NativeDeviceBSRBackend"
)
target_compile_definitions(
  test-cusparse-bsr-rejects-rectangular-spaces-probe
  PRIVATE
  GENDIL_TEST_CUSPARSE_BSR
)

gendil_add_target_compile_failure_test(
  test-rocsparse-bsr-rejects-rectangular-spaces
  rectangular_vendor_bsr.cpp
  "Rectangular BSR assembly requires HostBSRBackend or NativeDeviceBSRBackend"
)


# Mass
gendil_add_test(test-mass-3d-single-element test-mass-3d-single-element.cpp)
gendil_add_test(test-mass-3d test-mass-3d.cpp)
gendil_add_test(test-mass-1d-mfem-mesh test-mass-1d-mfem-mesh.cpp)
gendil_add_test(test-mass-2d-mfem-mesh test-mass-2d-mfem-mesh.cpp)
gendil_add_test(test-mass-3d-mfem-mesh test-mass-3d-mfem-mesh.cpp)
gendil_add_test(test-mass-4d-mfem-mesh test-mass-4d-mfem-mesh.cpp)
gendil_add_test(test-mass-H1-3d test-mass-H1-3d.cpp)
gendil_add_non_nvcc_test(test-mass-H1-bsr-3d test-mass-H1-bsr-3d.cpp)
gendil_add_non_nvcc_test(test-interior-coefficient-field-inputs-h1 test-interior-coefficient-field-inputs-h1.cpp)
gendil_add_test(test-mass-inverse test-mass-inverse.cpp)
gendil_add_test(test-vector-mass-3d test-vector-mass-3d.cpp)
# Advection
gendil_add_test(test-advection-3d-single-element test-advection-3d-single-element.cpp)
gendil_add_test(test-advection-1d test-advection-1d.cpp)
gendil_add_test(test-advection-2d test-advection-2d.cpp)
gendil_add_test(test-advection-3d test-advection-3d.cpp)
gendil_add_test(test-advection-2d-mfem-mesh test-advection-2d-mfem-mesh.cpp)
gendil_add_test(test-advection-3d-mfem-mesh test-advection-3d-mfem-mesh.cpp)
gendil_add_test(test-advection-6d-mfem-mesh test-advection-6d-mfem-mesh.cpp)
gendil_add_test(test-mass-advection-3d test-mass-advection-3d.cpp)
# Diffusion
gendil_add_test(test-diffusion-3d test-diffusion-3d.cpp)
gendil_add_non_nvcc_test(test-diffusion-3d-unstructured test-diffusion-3d-unstructured.cpp)
gendil_add_non_nvcc_test(test-diffusion-legacy-plus-jacobian test-diffusion-legacy-plus-jacobian.cpp)
gendil_add_non_nvcc_test(test-diffusion-4d-tensor-product-H1 test-diffusion-4d-tensor-product-H1.cpp)
gendil_add_non_nvcc_test(test-vector-interior-orientation-bsr-3d test-vector-interior-orientation-bsr-3d.cpp)
# Other
gendil_add_test(test-periodic-mesh test-periodic-mesh.cpp)
gendil_add_test(test-nested-product-mesh test-nested-product-mesh.cpp)
gendil_add_test(test-many-product-mesh test-many-product-mesh.cpp)
gendil_add_test(test-projections-6d test-projections-6d.cpp)

# MFEM global-face connectivity
function(gendil_add_mfem_global_face_test target compile_definition)
  gendil_add_non_nvcc_test(${target} test-mfem-global-face.cpp)
  if(TARGET ${target})
    target_compile_definitions(${target} PRIVATE ${compile_definition})
  endif()
endfunction()

gendil_add_mfem_global_face_test(
  test-mfem-global-face-quad-connectivity
  TEST_QUAD_CONNECTIVITY)

gendil_add_mfem_global_face_test(
  test-mfem-global-face-hex-connectivity
  TEST_HEX_CONNECTIVITY)

gendil_add_mfem_global_face_test(
  test-mfem-global-face-generic-operator
  TEST_GENERIC_OPERATOR)

gendil_add_mfem_global_face_test(
  test-mfem-global-face-file-sipdg
  TEST_FILE_SIPDG_LOCAL_VS_GLOBAL)

gendil_add_mfem_global_face_test(
  test-mfem-global-boundary-connectivity
  TEST_BOUNDARY)

gendil_add_mfem_global_face_test(
  test-mfem-global-face-smoke
  TEST_SMOKE)

gendil_add_mfem_global_face_test(
  test-mfem-global-face-periodic
  TEST_PERIODIC)

gendil_add_test(
  test-mfem-nonconforming-face-decoder
  test-mfem-nonconforming-face-decoder.cpp
)

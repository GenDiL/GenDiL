
# Mass
add_my_test(test-mass-3d-single-element test-mass-3d-single-element.cpp)
add_my_test(test-mass-3d test-mass-3d.cpp)
add_my_test(test-mass-1d-mfem-mesh test-mass-1d-mfem-mesh.cpp)
add_my_test(test-mass-2d-mfem-mesh test-mass-2d-mfem-mesh.cpp)
add_my_test(test-mass-3d-mfem-mesh test-mass-3d-mfem-mesh.cpp)
add_my_test(test-mass-4d-mfem-mesh test-mass-4d-mfem-mesh.cpp)
add_my_test(test-mass-H1-3d test-mass-H1-3d.cpp)
add_non_nvcc_test(test-mass-H1-bsr-3d test-mass-H1-bsr-3d.cpp)
add_non_nvcc_test(test-interior-coefficient-field-inputs-h1 test-interior-coefficient-field-inputs-h1.cpp)
add_my_test(test-mass-inverse test-mass-inverse.cpp)
add_my_test(test-vector-mass-3d test-vector-mass-3d.cpp)
# Advection
add_my_test(test-advection-3d-single-element test-advection-3d-single-element.cpp)
add_my_test(test-advection-1d test-advection-1d.cpp)
add_my_test(test-advection-2d test-advection-2d.cpp)
add_my_test(test-advection-3d test-advection-3d.cpp)
add_my_test(test-advection-2d-mfem-mesh test-advection-2d-mfem-mesh.cpp)
add_my_test(test-advection-3d-mfem-mesh test-advection-3d-mfem-mesh.cpp)
add_my_test(test-advection-6d-mfem-mesh test-advection-6d-mfem-mesh.cpp)
add_my_test(test-mass-advection-3d test-mass-advection-3d.cpp)
# Diffusion
add_my_test(test-diffusion-3d test-diffusion-3d.cpp)
add_non_nvcc_test(test-diffusion-3d-unstructured test-diffusion-3d-unstructured.cpp)
add_non_nvcc_test(test-diffusion-legacy-plus-jacobian test-diffusion-legacy-plus-jacobian.cpp)
add_non_nvcc_test(test-diffusion-4d-tensor-product-H1 test-diffusion-4d-tensor-product-H1.cpp)
add_non_nvcc_test(test-vector-interior-orientation-bsr-3d test-vector-interior-orientation-bsr-3d.cpp)
# Other
add_my_test(test-periodic-mesh test-periodic-mesh.cpp)
add_my_test(test-nested-product-mesh test-nested-product-mesh.cpp)
add_my_test(test-many-product-mesh test-many-product-mesh.cpp)
add_my_test(test-projections-6d test-projections-6d.cpp)

# MFEM global-face connectivity
set(mfem_global_face_source test-mfem-global-face.cpp)

add_non_nvcc_test(test-mfem-global-face-quad-connectivity ${mfem_global_face_source})
target_compile_definitions(test-mfem-global-face-quad-connectivity PRIVATE TEST_QUAD_CONNECTIVITY)

add_non_nvcc_test(test-mfem-global-face-hex-connectivity ${mfem_global_face_source})
target_compile_definitions(test-mfem-global-face-hex-connectivity PRIVATE TEST_HEX_CONNECTIVITY)

add_non_nvcc_test(test-mfem-global-face-generic-operator ${mfem_global_face_source})
target_compile_definitions(test-mfem-global-face-generic-operator PRIVATE TEST_GENERIC_OPERATOR)

add_non_nvcc_test(test-mfem-global-face-file-sipdg ${mfem_global_face_source})
target_compile_definitions(test-mfem-global-face-file-sipdg PRIVATE TEST_FILE_SIPDG_LOCAL_VS_GLOBAL)

add_non_nvcc_test(test-mfem-global-boundary-connectivity ${mfem_global_face_source})
target_compile_definitions(test-mfem-global-boundary-connectivity PRIVATE TEST_BOUNDARY)
add_my_test(
  test-mfem-nonconforming-face-decoder
  test-mfem-nonconforming-face-decoder.cpp
)


gendil_add_non_nvcc_test(test-mass-3d-generic test-mass-3d-generic.cpp)
gendil_add_test(
  test-nodal-subspace-projection
  test-nodal-subspace-projection.cpp
)
if(USE_CUDA OR USE_HIP)
  gendil_add_test(test-batched-mass-operator test-batched-mass-operator.cpp)
  gendil_add_test(
    test-batched-speed-of-light-operator
    test-batched-speed-of-light-operator.cpp
  )
  gendil_add_test(
    test-batched-global-face-speed-of-light-operator
    test-batched-global-face-speed-of-light-operator.cpp
  )
  gendil_add_test(
    test-batched-global-face-advection-operator
    test-batched-global-face-advection-operator.cpp
  )
  gendil_add_test(
    test-batched-mixed-mass-operator
    test-batched-mixed-mass-operator.cpp
  )
  gendil_add_test(
    test-batched-l2-projection-operator
    test-batched-l2-projection-operator.cpp
  )
  gendil_add_test(test-batched-l2-error test-batched-l2-error.cpp)
  gendil_add_test(
    test-batched-mass-inverse-operator
    test-batched-mass-inverse-operator.cpp
  )
  gendil_add_test(
    test-batched-linear-form-operator
    test-batched-linear-form-operator.cpp
  )
  gendil_add_test(
    test-batched-grad-grad-operator
    test-batched-grad-grad-operator.cpp
  )
  gendil_add_test(
    test-batched-grad-grad-write-coverage
    test-batched-grad-grad-write-coverage.cpp
  )
  gendil_add_test(
    test-batched-grad-grad-debug-variants
    test-batched-grad-grad-debug-variants.cpp
  )
  gendil_add_test(
    test-threaded-helper-undercoverage
    test-threaded-helper-undercoverage.cpp
  )
  gendil_add_test(
    test-threaded-aggregate-dimensions-diagnostic
    test-threaded-aggregate-dimensions-diagnostic.cpp
  )
  set_tests_properties(
    test-batched-mass-operator
    test-batched-speed-of-light-operator
    test-batched-global-face-speed-of-light-operator
    test-batched-global-face-advection-operator
    test-batched-mixed-mass-operator
    test-batched-l2-projection-operator
    test-batched-l2-error
    test-batched-mass-inverse-operator
    test-batched-linear-form-operator
    test-batched-grad-grad-operator
    test-batched-grad-grad-write-coverage
    test-batched-grad-grad-debug-variants
    test-threaded-helper-undercoverage
    test-threaded-aggregate-dimensions-diagnostic
    PROPERTIES
    TIMEOUT 30
  )
endif()
gendil_add_non_nvcc_test(test-coefficient-field-inputs-cell test-coefficient-field-inputs-cell.cpp)
gendil_add_non_nvcc_test(test-boundary-coefficient-field-inputs test-boundary-coefficient-field-inputs.cpp)
gendil_add_non_nvcc_test(test-interior-coefficient-field-inputs test-interior-coefficient-field-inputs.cpp)
gendil_add_non_nvcc_test(test-vector-coefficient-field-inputs test-vector-coefficient-field-inputs.cpp)
gendil_add_non_nvcc_test(test-upwind-vector-coefficient-field-inputs test-upwind-vector-coefficient-field-inputs.cpp)
gendil_add_non_nvcc_test(test-sipdg-coefficient-field-inputs test-sipdg-coefficient-field-inputs.cpp)
gendil_add_non_nvcc_test(test-anisotropic-diffusion-coefficient test-anisotropic-diffusion-coefficient.cpp)
gendil_add_non_nvcc_test(test-vector-mass-cell-3d test-vector-mass-cell-3d.cpp)
gendil_add_non_nvcc_test(test-vector-mass-bsr-3d test-vector-mass-bsr-3d.cpp)
gendil_add_non_nvcc_test(test-vector-boundary-mass-bsr-2d test-vector-boundary-mass-bsr-2d.cpp)
gendil_add_non_nvcc_test(test-vector-interior-jump-bsr-2d test-vector-interior-jump-bsr-2d.cpp)
gendil_add_test(test-rectangular-bsr-assembly test-rectangular-bsr-assembly.cpp)
gendil_add_non_nvcc_test(test-vector-gradgrad-cell-3d test-vector-gradgrad-cell-3d.cpp)
# gendil_add_test(test-vector-jump-penalty-interior-3d test-vector-jump-penalty-interior-3d.cpp)
gendil_add_non_nvcc_test(test-advection-1d-generic test-advection-1d-generic.cpp)
gendil_add_non_nvcc_test(test-advection-3d-generic test-advection-3d-generic.cpp)
gendil_add_non_nvcc_test(test-upwind-3d-generic test-upwind-3d-generic.cpp)
gendil_add_non_nvcc_test(test-diffusion-3d-generic test-diffusion-3d-generic.cpp)
gendil_add_non_nvcc_test(test-pullback-runtime-validation test-pullback-runtime-validation.cpp)
gendil_add_non_nvcc_test(test-diffusion-volume-only test-diffusion-volume-only.cpp)
gendil_add_non_nvcc_test(test-diffusion-consistency-only test-diffusion-consistency-only.cpp)
gendil_add_non_nvcc_test(test-diffusion-symmetry-only test-diffusion-symmetry-only.cpp)
gendil_add_non_nvcc_test(test-diffusion-symmetry-vs-legacy test-diffusion-symmetry-vs-legacy.cpp)
gendil_add_non_nvcc_test(test-diffusion-symmetry-1d-minimal test-diffusion-symmetry-1d-minimal.cpp)
gendil_add_non_nvcc_test(test-diffusion-vary-sigma test-diffusion-vary-sigma.cpp)
gendil_add_test(test-face-advection test-face-advection.cpp)
gendil_add_test(test-advection-p-adaptivity test-advection-p-adaptivity.cpp)
gendil_add_test(test-advection-h-adaptivity-2d test-advection-h-adaptivity-2d.cpp)
gendil_add_test(test-advection-hp-adaptivity-3d test-advection-hp-adaptivity-3d.cpp)
gendil_add_test(test-face-advection-split-mesh test-face-advection-split-mesh.cpp)
gendil_add_test(test-face-advection-split-mesh-2d-ascii test-face-advection-split-mesh-2d-ascii.cpp)
gendil_add_non_nvcc_test(test-diffusion-symmetry-face-analysis test-diffusion-symmetry-face-analysis.cpp)
gendil_add_non_nvcc_test(test-diffusion-symmetry-per-face test-diffusion-symmetry-per-face.cpp)
gendil_add_non_nvcc_test(test-diffusion-symmetry-plus-minus-faces test-diffusion-symmetry-plus-minus-faces.cpp)
gendil_add_non_nvcc_test(test-boundary-mass-2d-generic test-boundary-mass-2d-generic.cpp)
gendil_add_non_nvcc_test(test-boundary-sipdg-2d-generic test-boundary-sipdg-2d-generic.cpp)
gendil_add_non_nvcc_test(test-generic-global-face-domains test-generic-global-face-domains.cpp)
gendil_add_non_nvcc_test(test-unstructured-global-face-connectivity test-unstructured-global-face-connectivity.cpp)
gendil_add_non_nvcc_test(test-nonconforming-global-facet-geometry test-nonconforming-global-facet-geometry.cpp)
gendil_add_non_nvcc_test(test-h-adaptive-face-geometry-map test-h-adaptive-face-geometry-map.cpp)
gendil_add_non_nvcc_test(test-generic-hp-adaptive-upwind test-generic-hp-adaptive-upwind.cpp)
gendil_add_non_nvcc_test(test-mfem-point-matrix-decoder test-mfem-point-matrix-decoder.cpp)
gendil_add_non_nvcc_test(test-mixed-integration-domain test-mixed-integration-domain.cpp)
gendil_add_non_nvcc_test(test-partition-mixed-fes test-partition-mixed-fes.cpp)
gendil_add_non_nvcc_test(test-tensor-product-facet-dof-io test-tensor-product-facet-dof-io.cpp)
gendil_add_non_nvcc_test(test-cartesian-product-partition-structure test-cartesian-product-partition-structure.cpp)
gendil_add_non_nvcc_test(test-cartesian-product-partition-operators test-cartesian-product-partition-operators.cpp)
gendil_add_non_nvcc_test(test-cartesian-product-partition-nonconforming test-cartesian-product-partition-nonconforming.cpp)
gendil_add_test(test-cartesian-product-partition-device-smoke test-cartesian-product-partition-device-smoke.cpp)

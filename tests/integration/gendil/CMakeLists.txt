
add_non_cuda_test(test-mass-3d-generic test-mass-3d-generic.cpp)
if(USE_CUDA OR USE_HIP)
  add_my_test(test-batched-mass-operator test-batched-mass-operator.cpp)
  add_my_test(
    test-batched-speed-of-light-operator
    test-batched-speed-of-light-operator.cpp
  )
  add_my_test(
    test-batched-global-face-speed-of-light-operator
    test-batched-global-face-speed-of-light-operator.cpp
  )
  add_my_test(
    test-batched-mixed-mass-operator
    test-batched-mixed-mass-operator.cpp
  )
  add_my_test(
    test-batched-l2-projection-operator
    test-batched-l2-projection-operator.cpp
  )
  add_my_test(test-batched-l2-error test-batched-l2-error.cpp)
  add_my_test(
    test-batched-mass-inverse-operator
    test-batched-mass-inverse-operator.cpp
  )
  add_my_test(
    test-batched-linear-form-operator
    test-batched-linear-form-operator.cpp
  )
  add_my_test(
    test-batched-grad-grad-operator
    test-batched-grad-grad-operator.cpp
  )
  add_my_test(
    test-batched-grad-grad-write-coverage
    test-batched-grad-grad-write-coverage.cpp
  )
  add_my_test(
    test-batched-grad-grad-debug-variants
    test-batched-grad-grad-debug-variants.cpp
  )
  add_my_test(
    test-threaded-helper-undercoverage
    test-threaded-helper-undercoverage.cpp
  )
  add_my_test(
    test-threaded-aggregate-dimensions-diagnostic
    test-threaded-aggregate-dimensions-diagnostic.cpp
  )
  set_tests_properties(
    test-batched-mass-operator
    test-batched-speed-of-light-operator
    test-batched-global-face-speed-of-light-operator
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
add_non_cuda_test(test-coefficient-field-inputs-cell test-coefficient-field-inputs-cell.cpp)
add_non_cuda_test(test-boundary-coefficient-field-inputs test-boundary-coefficient-field-inputs.cpp)
add_non_cuda_test(test-interior-coefficient-field-inputs test-interior-coefficient-field-inputs.cpp)
add_non_cuda_test(test-vector-coefficient-field-inputs test-vector-coefficient-field-inputs.cpp)
add_non_cuda_test(test-upwind-vector-coefficient-field-inputs test-upwind-vector-coefficient-field-inputs.cpp)
add_non_cuda_test(test-sipdg-coefficient-field-inputs test-sipdg-coefficient-field-inputs.cpp)
add_non_cuda_test(test-anisotropic-diffusion-coefficient test-anisotropic-diffusion-coefficient.cpp)
add_non_cuda_test(test-vector-mass-cell-3d test-vector-mass-cell-3d.cpp)
add_non_cuda_test(test-vector-mass-bsr-3d test-vector-mass-bsr-3d.cpp)
add_non_cuda_test(test-vector-boundary-mass-bsr-2d test-vector-boundary-mass-bsr-2d.cpp)
add_non_cuda_test(test-vector-interior-jump-bsr-2d test-vector-interior-jump-bsr-2d.cpp)
add_non_cuda_test(test-vector-gradgrad-cell-3d test-vector-gradgrad-cell-3d.cpp)
# add_my_test(test-vector-jump-penalty-interior-3d test-vector-jump-penalty-interior-3d.cpp)
add_non_cuda_test(test-advection-1d-generic test-advection-1d-generic.cpp)
add_non_cuda_test(test-advection-3d-generic test-advection-3d-generic.cpp)
add_non_cuda_test(test-upwind-3d-generic test-upwind-3d-generic.cpp)
add_non_cuda_test(test-diffusion-3d-generic test-diffusion-3d-generic.cpp)
add_non_cuda_test(test-pullback-runtime-validation test-pullback-runtime-validation.cpp)
add_non_cuda_test(test-diffusion-volume-only test-diffusion-volume-only.cpp)
add_non_cuda_test(test-diffusion-consistency-only test-diffusion-consistency-only.cpp)
add_non_cuda_test(test-diffusion-symmetry-only test-diffusion-symmetry-only.cpp)
add_non_cuda_test(test-diffusion-symmetry-vs-legacy test-diffusion-symmetry-vs-legacy.cpp)
add_non_cuda_test(test-diffusion-symmetry-1d-minimal test-diffusion-symmetry-1d-minimal.cpp)
add_non_cuda_test(test-diffusion-vary-sigma test-diffusion-vary-sigma.cpp)
add_my_test(test-face-advection test-face-advection.cpp)
add_my_test(test-advection-p-adaptivity test-advection-p-adaptivity.cpp)
add_my_test(test-advection-h-adaptivity-2d test-advection-h-adaptivity-2d.cpp)
add_my_test(test-advection-hp-adaptivity-3d test-advection-hp-adaptivity-3d.cpp)
add_my_test(test-face-advection-split-mesh test-face-advection-split-mesh.cpp)
add_my_test(test-face-advection-split-mesh-2d-ascii test-face-advection-split-mesh-2d-ascii.cpp)
add_non_cuda_test(test-diffusion-symmetry-face-analysis test-diffusion-symmetry-face-analysis.cpp)
add_non_cuda_test(test-diffusion-symmetry-per-face test-diffusion-symmetry-per-face.cpp)
add_non_cuda_test(test-diffusion-symmetry-plus-minus-faces test-diffusion-symmetry-plus-minus-faces.cpp)
add_non_cuda_test(test-boundary-mass-2d-generic test-boundary-mass-2d-generic.cpp)
add_non_cuda_test(test-boundary-sipdg-2d-generic test-boundary-sipdg-2d-generic.cpp)

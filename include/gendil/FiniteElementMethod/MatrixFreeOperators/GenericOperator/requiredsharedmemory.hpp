// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <tuple>
#include <type_traits>

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/doflayout.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/facereaddofspolicy.hpp"
#include "gendil/Utilities/KernelContext/isthreadeddim.hpp"
#include "gendil/Utilities/MathHelperFunctions/max.hpp"

namespace gendil {

namespace detail {

template < typename T, typename = void >
struct is_tuple_like : std::false_type
{
};

template < typename T >
struct is_tuple_like<
   T,
   std::void_t<
      decltype(std::tuple_size<std::remove_cvref_t<T>>::value)> >
   : std::true_type
{
};

template < typename T >
inline constexpr bool is_tuple_like_v =
   is_tuple_like<T>::value;

template < typename DofShapes, size_t... I >
constexpr size_t MaxComponentDofCount(std::index_sequence<I...>)
{
   size_t value = 0;
   (
      (
         value =
            Max(
               value,
               static_cast<size_t>(
                  Product(std::tuple_element_t<I, DofShapes>{})) )
      ),
      ... );
   return value;
}

template < typename ShapeFunctions >
struct FaceReadDofScratchCount
{
   static constexpr size_t value =
      static_cast<size_t>(
         Product(finite_element_dof_shape_t<ShapeFunctions>{}));
};

template < typename... ScalarShapeFunctions >
struct FaceReadDofScratchCount<
   VectorShapeFunctions<ScalarShapeFunctions...> >
{
   using ShapeFunctions =
      VectorShapeFunctions<ScalarShapeFunctions...>;

   static constexpr size_t value =
      MaxComponentDofCount<typename ShapeFunctions::dof_shape>(
         std::make_index_sequence<ShapeFunctions::vector_dim>{});
};

} // namespace detail

template <
   typename KernelPolicy,
   typename IntegrationRule >
struct GenericOperatorIntegrandRequiredSharedMemory
{
   static constexpr size_t value =
      required_shared_memory_v<KernelPolicy, IntegrationRule>;
};

template <
   typename KernelPolicy,
   typename IntegrationRule >
inline constexpr size_t generic_operator_integrand_required_shared_memory_v =
   GenericOperatorIntegrandRequiredSharedMemory<
      KernelPolicy,
      IntegrationRule>::value;

/**
 * @brief Shared scratch needed by one global-face ReadDofs call.
 *
 * @details Non-threaded face reads use local stack/register storage. Direct
 * threaded global-face reads also return local/register objects without shared
 * staging. The current non-direct threaded face read path allocates one
 * temporary staging buffer, copies into the returned local/register object,
 * synchronizes, and resets the shared arena before returning. The returned
 * minus/plus DoF objects do not alias that temporary staging, so mirrored
 * global interior evaluation can reuse this one read buffer for the second
 * side.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename DofsInView = void >
struct GenericOperatorFaceReadScratchRequirement
{
private:
   using Space = std::remove_cvref_t<FiniteElementSpace>;
   using ShapeFunctions =
      typename Space::finite_element_type::shape_functions;
   using FaceReadPolicy = face_read_dofs_policy_t<KernelPolicy>;

   static constexpr bool threaded = is_threaded_v<KernelPolicy>;
   static constexpr bool direct_global_read =
      std::is_same_v<
         FaceReadPolicy,
         DirectGlobalFaceReadDofsPolicy>;
   static constexpr bool vector_tuple_view =
      is_vector_shape_functions_v<ShapeFunctions> &&
      detail::is_tuple_like_v<DofsInView>;

   static constexpr size_t staged_dofs =
      vector_tuple_view
         ? detail::FaceReadDofScratchCount<ShapeFunctions>::value
         : Space::finite_element_type::GetNumDofs();

public:
   static constexpr size_t value =
      !threaded || direct_global_read ? 0 : staged_dofs;
};

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename DofsInView = void >
inline constexpr size_t generic_operator_face_read_scratch_requirement_v =
   GenericOperatorFaceReadScratchRequirement<
      KernelPolicy,
      FiniteElementSpace,
      DofsInView>::value;

/**
 * @brief Shared scratch needed by one global-face WriteAddDofs call.
 *
 * @details The threaded non-direct face write path stages one face's local
 * DoFs, performs the oriented/additive global write, then resets the shared
 * arena before returning. In the generic global-face kernels the write happens
 * after the local facet integrand returns, so write scratch is not live with
 * read scratch or quadrature/test scratch. Scalar direct global writes bypass
 * the staging buffer; vector face writes currently use the staged path.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename DofsOutView = void >
struct GenericOperatorFaceWriteScratchRequirement
{
private:
   using Space = std::remove_cvref_t<FiniteElementSpace>;
   using ShapeFunctions =
      typename Space::finite_element_type::shape_functions;
   using FaceWritePolicy = face_write_dofs_policy_t<KernelPolicy>;

   static constexpr bool threaded = is_threaded_v<KernelPolicy>;
   static constexpr bool scalar_dofs =
      !is_vector_shape_functions_v<ShapeFunctions>;
   static constexpr bool direct_global_write =
      scalar_dofs &&
      std::is_same_v<
         FaceWritePolicy,
         DirectGlobalFaceWriteDofsPolicy>;

public:
   static constexpr size_t value =
      !threaded || direct_global_write
         ? 0
         : Space::finite_element_type::GetNumDofs();
};

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename DofsOutView = void >
inline constexpr size_t generic_operator_face_write_scratch_requirement_v =
   GenericOperatorFaceWriteScratchRequirement<
      KernelPolicy,
      FiniteElementSpace,
      DofsOutView>::value;

/**
 * @brief Local cell-owned generic operator scratch policy.
 *
 * @details Local interior and boundary facet evaluation is nested inside the
 * existing cell-owned traversal and shares the same quadrature/test scratch
 * model as the cell phase. The local facet reads/writes do not introduce an
 * additional independent global-face staging phase in this traversal.
 */
template <
   typename KernelPolicy,
   typename IntegrationRule,
   typename FiniteElementSpace = void,
   typename WeakForm = void,
   typename DofsInView = void,
   typename DofsOutView = void >
struct LocalGenericCellRequiredSharedMemory
{
   static constexpr size_t value =
      generic_operator_integrand_required_shared_memory_v<
         KernelPolicy,
         IntegrationRule>;
};

template <
   typename KernelPolicy,
   typename IntegrationRule,
   typename FiniteElementSpace = void,
   typename WeakForm = void,
   typename DofsInView = void,
   typename DofsOutView = void >
inline constexpr size_t local_generic_cell_required_shared_memory_v =
   LocalGenericCellRequiredSharedMemory<
      KernelPolicy,
      IntegrationRule,
      FiniteElementSpace,
      WeakForm,
      DofsInView,
      DofsOutView>::value;

template <
   typename KernelPolicy,
   typename IntegrationRule,
   typename FiniteElementSpace = void,
   typename WeakForm = void,
   typename DofsInView = void,
   typename DofsOutView = void >
struct LocalGenericInteriorFacetRequiredSharedMemory
   : LocalGenericCellRequiredSharedMemory<
        KernelPolicy,
        IntegrationRule,
        FiniteElementSpace,
        WeakForm,
        DofsInView,
        DofsOutView>
{
};

template <
   typename KernelPolicy,
   typename IntegrationRule,
   typename FiniteElementSpace = void,
   typename WeakForm = void,
   typename DofsInView = void,
   typename DofsOutView = void >
inline constexpr size_t
local_generic_interior_facet_required_shared_memory_v =
   LocalGenericInteriorFacetRequiredSharedMemory<
      KernelPolicy,
      IntegrationRule,
      FiniteElementSpace,
      WeakForm,
      DofsInView,
      DofsOutView>::value;

template <
   typename KernelPolicy,
   typename IntegrationRule,
   typename FiniteElementSpace = void,
   typename WeakForm = void,
   typename DofsInView = void,
   typename DofsOutView = void >
struct LocalGenericBoundaryFacetRequiredSharedMemory
   : LocalGenericCellRequiredSharedMemory<
        KernelPolicy,
        IntegrationRule,
        FiniteElementSpace,
        WeakForm,
        DofsInView,
        DofsOutView>
{
};

template <
   typename KernelPolicy,
   typename IntegrationRule,
   typename FiniteElementSpace = void,
   typename WeakForm = void,
   typename DofsInView = void,
   typename DofsOutView = void >
inline constexpr size_t
local_generic_boundary_facet_required_shared_memory_v =
   LocalGenericBoundaryFacetRequiredSharedMemory<
      KernelPolicy,
      IntegrationRule,
      FiniteElementSpace,
      WeakForm,
      DofsInView,
      DofsOutView>::value;

/**
 * @brief Global interior face generic operator scratch policy.
 *
 * @details The mirrored global prototype reads the canonical minus side, then
 * the canonical plus side, evaluates one local/current-row facet integrand,
 * writes that row, swaps sides, evaluates the other row, and writes again.
 * Each ReadDofs/WriteAddDofs call resets the shared arena before returning,
 * and the integrand/quadrature/test scratch is used between read and write
 * phases. The required arena is therefore the maximum live requirement among
 * one face read, one facet integrand, and one face write.
 */
template <
   typename KernelPolicy,
   typename IntegrationRule,
   typename FiniteElementSpace,
   typename WeakForm = void,
   typename DofsInView = void,
   typename DofsOutView = void >
struct GlobalGenericInteriorFacetRequiredSharedMemory
{
   static constexpr size_t value =
      Max(
         generic_operator_integrand_required_shared_memory_v<
            KernelPolicy,
            IntegrationRule>,
         generic_operator_face_read_scratch_requirement_v<
            KernelPolicy,
            FiniteElementSpace,
            DofsInView>,
         generic_operator_face_write_scratch_requirement_v<
            KernelPolicy,
            FiniteElementSpace,
            DofsOutView>);
};

template <
   typename KernelPolicy,
   typename IntegrationRule,
   typename FiniteElementSpace,
   typename WeakForm = void,
   typename DofsInView = void,
   typename DofsOutView = void >
inline constexpr size_t
global_generic_interior_facet_required_shared_memory_v =
   GlobalGenericInteriorFacetRequiredSharedMemory<
      KernelPolicy,
      IntegrationRule,
      FiniteElementSpace,
      WeakForm,
      DofsInView,
      DofsOutView>::value;

/**
 * @brief Global boundary face generic operator scratch policy.
 *
 * @details Boundary global faces are one-sided, but the scratch lifetime is
 * otherwise the same sequence as the interior prototype: one face read,
 * local/current-row boundary integrand evaluation, then one additive face
 * write. These phases reuse the shared arena sequentially, so they combine by
 * Max rather than by addition.
 */
template <
   typename KernelPolicy,
   typename IntegrationRule,
   typename FiniteElementSpace,
   typename WeakForm = void,
   typename DofsInView = void,
   typename DofsOutView = void >
struct GlobalGenericBoundaryFacetRequiredSharedMemory
{
   static constexpr size_t value =
      Max(
         generic_operator_integrand_required_shared_memory_v<
            KernelPolicy,
            IntegrationRule>,
         generic_operator_face_read_scratch_requirement_v<
            KernelPolicy,
            FiniteElementSpace,
            DofsInView>,
         generic_operator_face_write_scratch_requirement_v<
            KernelPolicy,
            FiniteElementSpace,
            DofsOutView>);
};

template <
   typename KernelPolicy,
   typename IntegrationRule,
   typename FiniteElementSpace,
   typename WeakForm = void,
   typename DofsInView = void,
   typename DofsOutView = void >
inline constexpr size_t
global_generic_boundary_facet_required_shared_memory_v =
   GlobalGenericBoundaryFacetRequiredSharedMemory<
      KernelPolicy,
      IntegrationRule,
      FiniteElementSpace,
      WeakForm,
      DofsInView,
      DofsOutView>::value;

} // namespace gendil

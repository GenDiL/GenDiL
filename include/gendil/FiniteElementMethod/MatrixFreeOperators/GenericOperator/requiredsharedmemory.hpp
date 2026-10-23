// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <tuple>
#include <type_traits>

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/facedofspolicies.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TensorContraction/contractionhelper.hpp"
#include "gendil/Utilities/KernelContext/isthreadeddim.hpp"
#include "gendil/Utilities/MathHelperFunctions/max.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/executioncontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontextcommon.hpp"

namespace gendil {

template<class KernelPolicy, class FiniteElementSpace, class DofsInView>
struct GenericOperatorFaceReadScratchRequirement;
template<class KernelPolicy, class FiniteElementSpace, class DofsOutView>
struct GenericOperatorFaceWriteScratchRequirement;

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

namespace detail {
// Full parent-element gathers/scatters use local/register rows. Unknown
// endpoint types must declare their staging contract instead of assuming zero.
template<class Endpoint, class Policy> struct FieldEndpointScratch;
template<class Space, class Storage, class Policy>
struct FieldEndpointScratch<FiniteElementFieldView<Space, Storage>, Policy>
   : std::integral_constant<size_t, 0> {};

template<class Policy, class QData> struct DofToQuadScratch;
template<class Policy>
struct DofToQuadScratch<Policy, Empty> : std::integral_constant<size_t, 0> {};
template<class Policy, class... Entries>
struct DofToQuadScratch<Policy, TensorProductData<Entries...>>
{
   using QD = TensorProductData<Entries...>;
   using Shape = max_sequence_t<make_contraction_input_shape<QD>, make_contraction_output_shape<QD>>;
   static constexpr size_t value = is_threaded_v<Policy> ? 2 * shared_block_size_v<Policy, Shape> : 0;
};
template<class Policy, class... Components>
struct DofToQuadScratch<Policy, std::tuple<Components...>>
{
   static constexpr size_t value = [] { size_t result = 0;
      ((result = Max(result, DofToQuadScratch<Policy, Components>::value)), ...);
      return result; }();
};
template<class Policy, class... Components>
struct DofToQuadScratch<Policy, VectorDofToQuad<Components...>>
   : DofToQuadScratch<Policy, std::tuple<Components...>> {};

template<class Policy, class Signature, class OperatorContext, SemanticSide Side = SemanticSide::Unqualified, class... Fields>
consteval size_t FieldContractionScratch(type_list<Fields...>)
{
   if constexpr (!is_threaded_v<Policy>)
      return 0;
   else
   {
      size_t result = 0;
      if constexpr (is_boundary_facets_domain<typename Signature::domain_type>::value ||
                    is_interior_facets_domain<typename Signature::domain_type>::value)
      {
         if constexpr (Side == SemanticSide::Plus)
            ((result = Max(result, DofToQuadScratch<Policy, std::remove_cvref_t<decltype(
               std::declval<const OperatorContext&>().template GetFiniteElementFacetQuadData<Fields::name>().PlusSide())>>::value)), ...);
         else
            ((result = Max(result, DofToQuadScratch<Policy, std::remove_cvref_t<decltype(
               std::declval<const OperatorContext&>().template GetFiniteElementFacetQuadData<Fields::name>().MinusSide())>>::value)), ...);
      }
      else
         ((result = Max(result, DofToQuadScratch<Policy, std::remove_cvref_t<decltype(
            std::declval<const OperatorContext&>().template GetFiniteElementQuadData<Fields::name>())>>::value)), ...);
      return result;
   }
}
template<class Policy, class Execution, class... Fields>
consteval size_t ReadEndpointScratch(type_list<Fields...>)
{
   size_t result = 0;
   ((result = Max(result, FieldEndpointScratch<std::remove_cvref_t<decltype(
      std::declval<const Execution&>().template GetReadField<Fields::name>())>, Policy>::value)), ...);
   return result;
}
template<class Policy, class Execution, class... Fields>
consteval size_t WriteEndpointScratch(type_list<Fields...>)
{
   size_t result = 0;
   ((result = Max(result, FieldEndpointScratch<std::remove_cvref_t<decltype(
      std::declval<Execution&>().template GetWriteField<Fields::name>())>, Policy>::value)), ...);
   return result;
}
template<bool Plus, bool Write, class Policy, class Endpoint, bool ParentCell = false>
consteval size_t FacetEndpointScratch()
{
   if constexpr (Endpoint::is_local) return 0;
   else if constexpr (ParentCell && !is_face_field_binding_v<decltype(std::declval<Endpoint>().space)>)
      return FieldEndpointScratch<Endpoint, Policy>::value;
   else
   {
      using Space = std::remove_cvref_t<decltype([](const Endpoint& endpoint) -> decltype(auto) {
         if constexpr (Plus) return GetPlusFacetFieldVolumeSpace(endpoint.space);
         else return GetMinusFacetFieldVolumeSpace(endpoint.space);
      }(std::declval<const Endpoint&>()))>;
      using View = std::remove_cvref_t<decltype([](const Endpoint& endpoint) -> decltype(auto) {
         if constexpr (is_face_field_binding_v<decltype(endpoint.space)> &&
                       !execution_context_detail::IsPreboundElementStorage<
                          std::remove_cvref_t<decltype(endpoint.dofs)>>())
         {
            if constexpr (Plus)
               return MakeElementTensorView(GetPlusFacetFieldVolumeSpace(endpoint.space), endpoint.dofs);
            else return MakeElementTensorView(GetMinusFacetFieldVolumeSpace(endpoint.space), endpoint.dofs);
         }
         else return (endpoint.dofs);
      }(std::declval<const Endpoint&>()))>;
      if constexpr (Write)
         return GenericOperatorFaceWriteScratchRequirement<Policy, Space, View>::value;
      else return GenericOperatorFaceReadScratchRequirement<Policy, Space, View>::value;
   }
}

template<bool Plus, bool Write, class Policy, class Context, bool ParentCell = false, class... Fields>
consteval size_t FacetEndpointsScratch(type_list<Fields...>)
{
   size_t result = 0;
   if constexpr (Write)
      ((result = Max(result, FacetEndpointScratch<Plus, true, Policy, std::remove_cvref_t<decltype(
         std::declval<Context&>().template GetWriteField<Fields::name>())>, ParentCell>())), ...);
   else
      ((result = Max(result, FacetEndpointScratch<Plus, false, Policy, std::remove_cvref_t<decltype(
         std::declval<const Context&>().template GetReadField<Fields::name>())>, ParentCell>())), ...);
   return result;
}

template<class Policy, class Signature, class... Fields>
consteval size_t OutputContractionMinimum(type_list<Fields...>)
{
   // The cell gradient-only two-stage transpose reserves one scalar even for
   // a serial kernel. Boundary transpose does not use that helper.
   return !is_threaded_v<Policy> && is_cells_domain<typename Signature::domain_type>::value &&
      ((need_gradients(Fields::mask) && !need_values(Fields::mask)) || ...);
}
} // namespace detail

template<class KernelPolicy, class Integrand, class Execution, class OperatorContext,
         class PlusContext = Execution>
struct GenericOperatorIntegrandRequiredSharedMemory
{
   using Signature = typename OperatorContext::template signature_for<Integrand>;
   using Reads = concat_t<typename Signature::active_inputs, typename Signature::bound_inputs>;
   using Outputs = typename Signature::test_outputs;
   static constexpr size_t read_staging_slots = [] {
      if constexpr ((Signature::entity == SemanticEntity::Cell))
         return detail::ReadEndpointScratch<KernelPolicy, Execution>(Reads{});
      else if constexpr ((Signature::entity == SemanticEntity::BoundaryFacet))
         return detail::FacetEndpointsScratch<false, false, KernelPolicy, Execution, true>(Reads{});
      else
      {
         using MinusReads = concat_t<typename Signature::template active_inputs_on<SemanticSide::Minus>,
            typename Signature::template bound_inputs_on<SemanticSide::Minus>>;
         using PlusReads = concat_t<typename Signature::template active_inputs_on<SemanticSide::Plus>,
            typename Signature::template bound_inputs_on<SemanticSide::Plus>>;
         return Max(detail::FacetEndpointsScratch<false, false, KernelPolicy, Execution>(MinusReads{}),
            detail::FacetEndpointsScratch<true, false, KernelPolicy, PlusContext>(PlusReads{}));
      }
   }();
   static constexpr bool canonical = [] {
      if constexpr (!Signature::is_zero && (Signature::entity == SemanticEntity::InteriorFacet))
      {
         using Output = weak_form_traits::only_field_t<Outputs>;
         return is_face_field_binding_v<decltype(std::declval<const Execution&>().template GetSpace<Output::name>())>;
      }
      else return false;
   }();
   using MinusOutputs = std::conditional_t<canonical,
      typename Signature::template test_outputs_on<SemanticSide::Minus>, Outputs>;
   using PlusOutputs = std::conditional_t<canonical,
      typename Signature::template test_outputs_on<SemanticSide::Plus>, type_list<>>;
   static constexpr size_t write_staging_slots = [] {
      if constexpr ((Signature::entity == SemanticEntity::Cell))
         return detail::WriteEndpointScratch<KernelPolicy, Execution>(Outputs{});
      else return Max(detail::FacetEndpointsScratch<false, true, KernelPolicy, Execution,
         (Signature::entity == SemanticEntity::BoundaryFacet)>(MinusOutputs{}),
         detail::FacetEndpointsScratch<true, true, KernelPolicy, PlusContext>(PlusOutputs{}));
   }();
   // Gather, contraction and scatter phases release their arena sequentially.
   // Retained interpolants, channels and rows own local/register storage.
   static constexpr size_t value = [] {
      size_t contractions = 0;
      if constexpr ((Signature::entity == SemanticEntity::InteriorFacet))
      {
         using MinusReads = concat_t<typename Signature::template active_inputs_on<SemanticSide::Minus>,
            typename Signature::template bound_inputs_on<SemanticSide::Minus>>;
         using PlusReads = concat_t<typename Signature::template active_inputs_on<SemanticSide::Plus>,
            typename Signature::template bound_inputs_on<SemanticSide::Plus>>;
         contractions = Max(
            detail::FieldContractionScratch<KernelPolicy, Signature, OperatorContext, SemanticSide::Minus>(MinusReads{}),
            detail::FieldContractionScratch<KernelPolicy, Signature, OperatorContext, SemanticSide::Plus>(PlusReads{}),
            detail::FieldContractionScratch<KernelPolicy, Signature, OperatorContext, SemanticSide::Minus>(MinusOutputs{}),
            detail::FieldContractionScratch<KernelPolicy, Signature, OperatorContext, SemanticSide::Plus>(PlusOutputs{}));
      }
      else contractions = Max(detail::FieldContractionScratch<KernelPolicy, Signature, OperatorContext>(Reads{}),
         detail::FieldContractionScratch<KernelPolicy, Signature, OperatorContext>(Outputs{}));
      return Max(read_staging_slots, write_staging_slots, contractions,
         detail::OutputContractionMinimum<KernelPolicy, Signature>(Outputs{}));
   }();
};

template<class KernelPolicy, class Integrand, class Execution, class OperatorContext,
         class PlusContext = Execution>
inline constexpr size_t generic_operator_integrand_required_shared_memory_v =
   GenericOperatorIntegrandRequiredSharedMemory<KernelPolicy, Integrand,
      Execution, OperatorContext, PlusContext>::value;

/**
 * @brief Shared scratch needed by one global-face ReadDofs call.
 *
 * @details Non-threaded face reads use local stack/register storage. Direct
 * threaded global-face reads also return local/register objects without shared
 * staging. The current non-direct threaded face read path allocates one
 * temporary staging buffer, copies into the returned local/register object,
 * synchronizes, and resets the shared arena before returning. The returned
 * global-face DoF objects do not alias that temporary staging, so later phases
 * can reuse the arena.
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

namespace detail
{

// Sequential integrands reuse scratch; only their maximum is live at once.
template<class Policy, class Execution, class OperatorContext, class PlusContext = Execution, class... Terms>
consteval size_t TermSharedSlots(std::tuple<Terms...>*)
{
   size_t slots = 0;
   ((slots = Max(slots, GenericOperatorIntegrandRequiredSharedMemory<Policy,
      std::remove_cvref_t<Terms>, Execution, OperatorContext, PlusContext>::value)), ...);
   return slots;
}

} // namespace detail

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file
 * @brief Interpolate the named fields read by one integrand at quadrature points.
 *
 * InterpolateFields connects invocation storage to weak-form evaluation. For each
 * requested field, it reads the element DOFs and applies that field's basis to
 * compute the requested values, reference-coordinate gradients, or both. Active
 * inputs and persistent coefficient fields use the same named read accessors.
 *
 * A cell integrand obtains its fields with:
 * @code
 * auto fields = InterpolateFields(
 *    kernel_context, execution_context, operator_context,
 *    element_context, integrand);
 * @endcode
 * The returned StaticMap contains InterpolatedField entries keyed by NameTag.
 * Boundary interpolation adds a face argument and also returns one map. Interior
 * interpolation returns FaceFields with separate minus and plus maps; their side
 * ordering follows the supplied face, whether current/neighbor or canonical.
 *
 * The operator context supplies reusable basis data and the integrand's signature,
 * already resolved against the complete form. Only the named reads and operators
 * required by that signature are interpolated. Geometry is supplied separately
 * by quadrature-point contexts, which also provide the mapping used to convert
 * reference gradients to physical gradients during expression evaluation.
 *
 * Each field's gather and interpolation finish before the next field begins.
 * Returned fields own their quadrature tensors; temporary gather and contraction
 * storage may reuse the kernel's shared scratch. Callers must provide a validated
 * execution context, matching operator context, and sufficient kernel scratch.
 */

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/executioncontext.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/Utilities/staticmap.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globalfacefieldbinding.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"

namespace gendil
{

/**
 * @brief Named quadrature fields on the two sides of an interior facet.
 *
 * Local traversal supplies current/neighbor fields; canonical traversal supplies
 * fixed minus/plus fields. The two maps may contain different names and operators.
 */
template < typename MinusFields, typename PlusFields >
struct FaceFields
{
   MinusFields minus_fields;
   PlusFields plus_fields;

   /// Unqualified access uses the current (minus) side after facet validation.
   template < typename Key >
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) Get() const
   {
      return minus_fields.template Get<Key>();
   }
};

namespace details
{

// Unrequested operators produce Empty without instantiating their contractions.
template <
   bool Need,
   typename KernelContext,
   typename QuadData,
   typename ElementDofsIn >
GENDIL_HOST_DEVICE
auto InterpolateValuesIfNeeded(
   KernelContext & kernel_context,
   const QuadData & finite_element_quad_data,
   const ElementDofsIn & element_dofs )
{
   if constexpr (Need)
   {
      return InterpolateValues(kernel_context, finite_element_quad_data, element_dofs);
   }
   else
   {
      return Empty{};
   }
}

template <
   bool Need,
   typename KernelContext,
   typename QuadData,
   typename ElementDofsIn >
GENDIL_HOST_DEVICE
auto InterpolateGradientIfNeeded(
   KernelContext & kernel_context,
   const QuadData & finite_element_quad_data,
   const ElementDofsIn & element_dofs )
{
   if constexpr (Need)
   {
      return InterpolateGradient(kernel_context, finite_element_quad_data, element_dofs);
   }
   else
   {
      return Empty{};
   }
}

template <
   bool Need,
   typename KernelContext,
   typename FaceSide,
   typename FaceQuadData,
   typename ElementDofsIn >
GENDIL_HOST_DEVICE
auto InterpolateFacetValuesIfNeeded(
   KernelContext & kernel_context,
   const FaceSide & side,
   const FaceQuadData & finite_element_quad_data,
   const ElementDofsIn & element_dofs )
{
   if constexpr (Need)
   {
      return InterpolateValues(
         kernel_context,
         side,
         finite_element_quad_data,
         element_dofs);
   }
   else
   {
      return Empty{};
   }
}

template <
   bool Need,
   typename KernelContext,
   typename FaceSide,
   typename FaceQuadData,
   typename ElementDofsIn >
GENDIL_HOST_DEVICE
auto InterpolateFacetGradientIfNeeded(
   KernelContext & kernel_context,
   const FaceSide & side,
   const FaceQuadData & finite_element_quad_data,
   const ElementDofsIn & element_dofs )
{
   if constexpr (Need)
   {
      return InterpolateGradient(
         kernel_context,
         side,
         finite_element_quad_data,
         element_dofs);
   }
   else
   {
      return Empty{};
   }
}

} // namespace details

/// Apply the requested cell contractions and package one named quadrature field.
template <
   StaticString Name,
   OperatorMask Mask,
   typename KernelContext,
   typename QuadData,
   typename ElementDofsIn >
GENDIL_HOST_DEVICE
auto MakeCellInterpolatedEntry(
   KernelContext & kernel_context,
   const QuadData & finite_element_quad_data,
   const ElementDofsIn & element_dofs )
{
   constexpr bool needs_values = need_values(Mask);
   constexpr bool needs_gradients = need_gradients(Mask);

   using Values = std::remove_cvref_t<decltype(
      details::InterpolateValuesIfNeeded<needs_values>(
         kernel_context, finite_element_quad_data, element_dofs))>;
   using Gradients = std::remove_cvref_t<decltype(
      details::InterpolateGradientIfNeeded<needs_gradients>(
         kernel_context, finite_element_quad_data, element_dofs))>;
   using Field = InterpolatedField<Values, Gradients>;

   return Entry<NameTag<Name>, Field>{
      Field{
         details::InterpolateValuesIfNeeded<needs_values>(
            kernel_context, finite_element_quad_data, element_dofs),
         details::InterpolateGradientIfNeeded<needs_gradients>(
            kernel_context, finite_element_quad_data, element_dofs)
      }
   };
}

// Read one named input, whether active or persistent, through its bound endpoint.
template <
   typename Requirement,
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename Entity >
GENDIL_HOST_DEVICE
auto InterpolateBoundField(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const Entity & entity )
{
   const auto & source = execution_context.template GetReadField<Requirement::name>();
   const auto & finite_element_quad_data =
      operator_context.template GetFiniteElementQuadData<Requirement::name>();

   // Preserve the borrowed row when the source is a local element overlay.
   decltype(auto) element_dofs = ReadFieldDofs(kernel_context, source, entity);

   return MakeCellInterpolatedEntry<Requirement::name, Requirement::mask>(
      kernel_context,
      finite_element_quad_data,
      element_dofs);
}

template <
   typename... Fields,
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename Entity >
GENDIL_HOST_DEVICE
auto InterpolateBoundFields(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const Entity & entity,
   type_list<Fields...> )
{
   // Braced initialization completes each field before gathering the next one.
   // An empty requirement list naturally produces an empty map.
   return StaticMap<decltype(InterpolateBoundField<Fields>(
      kernel_context, execution_context, operator_context, entity))...>{
      {
         InterpolateBoundField<Fields>(
            kernel_context, execution_context, operator_context, entity)...
      }
   };
}

/**
 * @brief Interpolate the named reads of a cell integrand on the current element.
 *
 * The stored signature merges active and persistent reads by name and operator
 * mask. Each field uses its own space and basis; a term without field reads
 * returns an empty map. The integrand is inspected only through its type here.
 */
template <
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename Entity,
   typename Integrand >
   requires CellIntegrand<Integrand>
GENDIL_HOST_DEVICE
auto InterpolateFields(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const Entity & entity,
   const Integrand & )
{
   using Signature = typename OperatorContext::template signature_for<Integrand>;
   using Reads = merge_named_field_requirement_list_t<concat_t<
      typename Signature::active_inputs,
      typename Signature::bound_inputs>>;

   return InterpolateBoundFields(
      kernel_context,
      execution_context,
      operator_context,
      entity,
      Reads{});
}

/// Apply the requested facet contractions using the selected side's basis data.
// The input row is already oriented; interpolation selects conforming or
// nonconforming basis data.
template <
   StaticString Name,
   OperatorMask Mask,
   typename KernelContext,
   typename FaceSide,
   typename FaceQuadData,
   typename ElementDofsIn >
GENDIL_HOST_DEVICE
auto MakeFacetInterpolatedEntry(
   KernelContext & kernel_context,
   const FaceSide & side,
   const FaceQuadData & finite_element_quad_data,
   const ElementDofsIn & element_dofs )
{
   constexpr bool needs_values = need_values(Mask);
   constexpr bool needs_gradients = need_gradients(Mask);

   using Values = std::remove_cvref_t<decltype(
      details::InterpolateFacetValuesIfNeeded<needs_values>(
         kernel_context, side, finite_element_quad_data, element_dofs))>;
   using Gradients = std::remove_cvref_t<decltype(
      details::InterpolateFacetGradientIfNeeded<needs_gradients>(
         kernel_context, side, finite_element_quad_data, element_dofs))>;
   using Field = InterpolatedField<Values, Gradients>;

   return Entry<NameTag<Name>, Field>{
      Field{
         details::InterpolateFacetValuesIfNeeded<needs_values>(
            kernel_context, side, finite_element_quad_data, element_dofs),
         details::InterpolateFacetGradientIfNeeded<needs_gradients>(
            kernel_context, side, finite_element_quad_data, element_dofs)
      }
   };
}

/**
 * @brief Select a field's volume space and quadrature data on one facet side.
 *
 * The side descriptor is copied; the space and quadrature data are borrowed and
 * must outlive this binding. No field storage is acquired or copied here.
 */
template < typename FaceSide, typename VolumeSpace, typename FaceQuadData >
struct FacetFieldSideBinding
{
   FaceSide side;
   const VolumeSpace & volume_space;
   const FaceQuadData & qdata;
};

/// Select the minus side (the adjacent cell for a physical boundary).
template < typename FaceContext, typename FieldSpace, typename QDataWrapper >
GENDIL_HOST_DEVICE
constexpr auto MakeMinusFacetFieldBinding(
   const FaceContext & face_info,
   const FieldSpace & field_space,
   const QDataWrapper & qdata_wrapper )
{
   decltype(auto) volume_space = GetMinusFacetFieldVolumeSpace(field_space);
   decltype(auto) qdata = qdata_wrapper.MinusSide();

   using FaceSide = std::remove_cvref_t<decltype(face_info.MinusSide())>;
   using VolumeSpace = std::remove_cvref_t<decltype(volume_space)>;
   using FaceQuadData = std::remove_cvref_t<decltype(qdata)>;

   return FacetFieldSideBinding<FaceSide, VolumeSpace, FaceQuadData>{
      face_info.MinusSide(),
      volume_space,
      qdata
   };
}

/// Select the plus side using the same ordering as the supplied interior face.
template < typename FaceContext, typename FieldSpace, typename QDataWrapper >
GENDIL_HOST_DEVICE
constexpr auto MakePlusFacetFieldBinding(
   const FaceContext & face_info,
   const FieldSpace & field_space,
   const QDataWrapper & qdata_wrapper )
{
   decltype(auto) volume_space = GetPlusFacetFieldVolumeSpace(field_space);
   decltype(auto) qdata = qdata_wrapper.PlusSide();

   using FaceSide = std::remove_cvref_t<decltype(face_info.PlusSide())>;
   using VolumeSpace = std::remove_cvref_t<decltype(volume_space)>;
   using FaceQuadData = std::remove_cvref_t<decltype(qdata)>;

   return FacetFieldSideBinding<FaceSide, VolumeSpace, FaceQuadData>{
      face_info.PlusSide(),
      volume_space,
      qdata
   };
}

/**
 * @brief Read a named field using the restriction of the selected facet side.
 *
 * Restricted storage uses the existing orientation-aware gather. Local overlays
 * instead return a const reference to their original row: it already has the
 * numerical routine's expected layout. Keep decltype(auto) at the call site to
 * preserve that reference rather than copy the tensor.
 */
template <
   typename KernelContext,
   typename Source,
   typename Side,
   typename Space,
   typename QData >
GENDIL_HOST_DEVICE
decltype(auto) ReadFieldDofs(
   KernelContext & kernel_context,
   const Source & source,
   const FacetFieldSideBinding<Side, Space, QData> & binding )
{
   if constexpr (Source::is_local)
   {
      return static_cast<const std::remove_reference_t<decltype(source.dofs)> &>(source.dofs);
   }
   else if constexpr (is_face_field_binding_v<decltype(source.space)> &&
                      !execution_context_detail::IsPreboundElementStorage<
                         std::remove_cvref_t<decltype(source.dofs)>>())
   {
      auto view = MakeElementTensorView(binding.volume_space, source.dofs);
      return ReadDofs(kernel_context, binding.volume_space, binding.side, view);
   }
   else
   {
      return ReadDofs(kernel_context, binding.volume_space, binding.side, source.dofs);
   }
}

// Select the side after resolving the field by name; storage stays name-only.
template <
   bool Plus,
   typename Requirement,
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename Face >
GENDIL_HOST_DEVICE
auto InterpolateFacetBoundField(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const Face & face )
{
   const auto & source = execution_context.template GetReadField<Requirement::name>();
   const auto & finite_element_quad_data =
      operator_context.template GetFiniteElementFacetQuadData<Requirement::name>();

   const auto binding = [&]()
   {
      if constexpr (Plus)
      {
         return MakePlusFacetFieldBinding(face, source.space, finite_element_quad_data);
      }
      else
      {
         return MakeMinusFacetFieldBinding(face, source.space, finite_element_quad_data);
      }
   }();

   decltype(auto) element_dofs = ReadFieldDofs(kernel_context, source, binding);

   return MakeFacetInterpolatedEntry<Requirement::name, Requirement::mask>(
      kernel_context,
      binding.side,
      binding.qdata,
      element_dofs);
}

template <
   bool Plus,
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename Face,
   typename... Fields >
GENDIL_HOST_DEVICE
auto InterpolateFacetBoundFields(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const Face & face,
   type_list<Fields...> )
{
   // Complete each gather/interpolation before beginning the next field, and
   // construct the final named map directly without intermediate tuples.
   return StaticMap<decltype(InterpolateFacetBoundField<Plus, Fields>(
      kernel_context, execution_context, operator_context, face))...>{
      {
         InterpolateFacetBoundField<Plus, Fields>(
            kernel_context, execution_context, operator_context, face)...
      }
   };
}

/**
 * @brief Interpolate an interior integrand's reads separately on each side.
 *
 * Each map uses its side's required names and masks. Distinct contexts allow
 * cached current rows with restricted neighbor reads, or two local assembly
 * buffers sharing the same field names. Minus interpolation finishes before
 * plus interpolation begins, so their temporary staging can reuse scratch.
 */
template <
   typename KernelContext,
   typename MinusContext,
   typename PlusContext,
   typename OperatorContext,
   typename Face,
   typename Integrand >
   requires InteriorFacetIntegrand<Integrand>
GENDIL_HOST_DEVICE
auto InterpolateFields(
   KernelContext & kernel_context,
   const MinusContext & minus_execution_context,
   const PlusContext & plus_execution_context,
   const OperatorContext & operator_context,
   const Face & face,
   const Integrand & )
{
   using Signature = typename OperatorContext::template signature_for<Integrand>;
   using MinusReads = merge_named_field_requirement_list_t<concat_t<
      typename Signature::template active_inputs_on<SemanticSide::Minus>,
      typename Signature::template bound_inputs_on<SemanticSide::Minus>>>;
   using PlusReads = merge_named_field_requirement_list_t<concat_t<
      typename Signature::template active_inputs_on<SemanticSide::Plus>,
      typename Signature::template bound_inputs_on<SemanticSide::Plus>>>;

   auto minus_fields = InterpolateFacetBoundFields<false>(
      kernel_context,
      minus_execution_context,
      operator_context,
      face,
      MinusReads{});
   auto plus_fields = InterpolateFacetBoundFields<true>(
      kernel_context,
      plus_execution_context,
      operator_context,
      face,
      PlusReads{});

   return FaceFields{std::move(minus_fields), std::move(plus_fields)};
}

/// Read both interior sides from one invocation context using their restrictions.
template <
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename Face,
   typename Integrand >
   requires InteriorFacetIntegrand<Integrand>
GENDIL_HOST_DEVICE
auto InterpolateFields(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const Face & face,
   const Integrand & integrand )
{
   return InterpolateFields(
      kernel_context,
      execution_context,
      execution_context,
      operator_context,
      face,
      integrand);
}

// Boundary face bindings use their adjacent side; volume fields use the parent
// element. Both are interpolated with that field's boundary-face basis data.
template <
   typename Requirement,
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename Entity,
   typename Face >
GENDIL_HOST_DEVICE
auto InterpolateBoundaryBoundField(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const Entity & entity,
   const Face & face )
{
   const auto & source = execution_context.template GetReadField<Requirement::name>();
   const auto & finite_element_quad_data =
      operator_context.template GetFiniteElementFacetQuadData<Requirement::name>();
   const auto binding = MakeMinusFacetFieldBinding(face, source.space, finite_element_quad_data);

   decltype(auto) element_dofs = [&]() -> decltype(auto)
   {
      if constexpr (is_face_field_binding_v<decltype(source.space)>)
      {
         return ReadFieldDofs(kernel_context, source, binding);
      }
      else
      {
         return ReadFieldDofs(kernel_context, source, entity);
      }
   }();

   return MakeFacetInterpolatedEntry<Requirement::name, Requirement::mask>(
      kernel_context,
      binding.side,
      binding.qdata,
      element_dofs);
}

template <
   typename... Fields,
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename Entity,
   typename Face >
GENDIL_HOST_DEVICE
auto InterpolateBoundaryBoundFields(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const Entity & entity,
   const Face & face,
   type_list<Fields...> )
{
   return StaticMap<decltype(InterpolateBoundaryBoundField<Fields>(
      kernel_context, execution_context, operator_context, entity, face))...>{
      {
         InterpolateBoundaryBoundField<Fields>(
            kernel_context, execution_context, operator_context, entity, face)...
      }
   };
}

/**
 * @brief Interpolate the named reads at a physical boundary's quadrature points.
 *
 * The result is a single map for the adjacent cell, with no plus-side reads.
 * Entity identifies the parent element for volume-field gathers; face selects
 * the adjacent side and boundary interpolation data. Geometry and boundary trace
 * identities are handled by the quadrature context and weak-form evaluation.
 */
template <
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename Entity,
   typename Face,
   typename Integrand >
   requires BoundaryFacetIntegrand<Integrand>
GENDIL_HOST_DEVICE
auto InterpolateFields(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const Entity & entity,
   const Face & face,
   const Integrand & )
{
   using Signature = typename OperatorContext::template signature_for<Integrand>;
   using Reads = merge_named_field_requirement_list_t<concat_t<
      typename Signature::active_inputs,
      typename Signature::bound_inputs>>;

   return InterpolateBoundaryBoundFields(
      kernel_context,
      execution_context,
      operator_context,
      entity,
      face,
      Reads{});
}

} // namespace gendil

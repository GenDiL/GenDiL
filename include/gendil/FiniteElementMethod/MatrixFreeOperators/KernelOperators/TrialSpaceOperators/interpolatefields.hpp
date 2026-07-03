// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/Utilities/staticmap.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/globalfacefieldbinding.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"

namespace gendil
{

/**
 * @file interpolatefields.hpp
 * @brief Utilities to interpolate trial and named finite-element fields
 *        at quadrature points for cell and facet integrals.
 *
 * This file provides the interpolation layer that transforms:
 *
 * - element-local trial degrees of freedom, and
 * - named finite-element fields stored in the weak-form context,
 *
 * into map-like structures containing interpolated values and/or gradients at
 * quadrature points.
 *
 * Two integration situations are handled:
 *
 * - **cell integration**
 *   returns one flat map containing the trial field and all referenced
 *   named finite-element fields required by explicit field expressions or
 *   coefficient input tags;
 *
 * - **interior-facet integration**
 *   returns a `FaceFields` object containing side-specific named-field maps for
 *   the minus and plus sides.
 *
 * The facet path intentionally preserves the current face-aware interpolation
 * model:
 *
 * - the face interpolation call site selects one coherent side binding:
 *   FaceView side, side volume finite element space, and side qdata,
 * - boundary facets use the minus-side binding by convention,
 * - selection of the local-face entry and conforming / non-conforming handling
 *   are delegated to the existing overloads of `InterpolateValues` and
 *   `InterpolateGradient`.
 *
 * Only named finite-element data backed by the active trial field or by fields
 * in the weak-form context is interpolated through this layer. Other coefficient
 * inputs are read from the quadrature-point context.
 *
 * @note This file assumes the existence of:
 * - `requirements<Integrand>`
 * - `interpolation_named_field_requirements_t<Expr>`
 * - `type_list<...>`
 * - `Entry<Key, Value>`
 * - `NameTag<Name>`
 * - `make_map(...)`
 * - `ReadDofs(...)`
 * - `InterpolateValues(...)`
 * - `InterpolateGradient(...)`
 * - `Empty`
 */


/* -------------------------------------------------------------------------- */
/*                              Field containers                              */
/* -------------------------------------------------------------------------- */

// /**
//  * @brief Container holding interpolated data associated with one field.
//  *
//  * @tparam ValuesType    Type storing interpolated field values.
//  * @tparam GradientType  Type storing interpolated field gradients.
//  *
//  * Either member may be `Empty` when the corresponding quantity is not needed.
//  */
// template<typename ValuesType, typename GradientType>
// struct InterpolatedField
// {
//    /// Interpolated field values, or `Empty`.
//    ValuesType values;

//    /// Interpolated field gradients, or `Empty`.
//    GradientType gradients;
// };

/**
 * @brief Container grouping the fields required on a facet.
 *
 * @tparam MinusFields  Map-like type storing minus-side named fields.
 * @tparam PlusFields   Map-like type storing plus-side named fields.
 */
template<class MinusFields, class PlusFields>
struct FaceFields
{
   /// Minus-side named fields.
   MinusFields minus_fields;

   /// Plus-side named fields.
   PlusFields plus_fields;

   /// Current-side access for forms that have passed interior-facet validation.
   template<class Key>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) get() const
   {
      return minus_fields.template get<Key>();
   }
};


/* -------------------------------------------------------------------------- */
/*                          Internal lazy helper API                          */
/* -------------------------------------------------------------------------- */

namespace details
{

template<bool Need, class KernelContext, class QuadData, class ElementDofsIn>
GENDIL_HOST_DEVICE
auto InterpolateValuesIfNeeded(
   KernelContext& k,
   const QuadData& qd,
   const ElementDofsIn& dofs_in)
{
   if constexpr (Need) return InterpolateValues(k, qd, dofs_in);
   else                return Empty{};
}

template<bool Need, class KernelContext, class QuadData, class ElementDofsIn>
GENDIL_HOST_DEVICE
auto InterpolateGradientIfNeeded(
   KernelContext& k,
   const QuadData& qd,
   const ElementDofsIn& dofs_in)
{
   if constexpr (Need) return InterpolateGradient(k, qd, dofs_in);
   else                return Empty{};
}

template<bool Need, class KernelContext, class FaceSide, class FaceQuadData, class ElementDofsIn>
GENDIL_HOST_DEVICE
auto InterpolateFacetValuesIfNeeded(
   KernelContext& k,
   const FaceSide& side,
   const FaceQuadData& face_qd,
   const ElementDofsIn& dofs_in)
{
   if constexpr (Need) return InterpolateValues(k, side, face_qd, dofs_in);
   else                return Empty{};
}

template<bool Need, class KernelContext, class FaceSide, class FaceQuadData, class ElementDofsIn>
GENDIL_HOST_DEVICE
auto InterpolateFacetGradientIfNeeded(
   KernelContext& k,
   const FaceSide& side,
   const FaceQuadData& face_qd,
   const ElementDofsIn& dofs_in)
{
   if constexpr (Need) return InterpolateGradient(k, side, face_qd, dofs_in);
   else                return Empty{};
}

} // namespace details


/* -------------------------------------------------------------------------- */
/*                    Low-level cell interpolation helpers                     */
/* -------------------------------------------------------------------------- */

/**
 * @brief Build the interpolated entry associated with the trial field for a
 *        cell integration context.
 *
 * This helper constructs
 * @code
 * Entry<NameTag<TrialName>, InterpolatedField<ValuesT, GradsT>>
 * @endcode
 * where values and/or gradients are computed according to `Mask`.
 *
 * @tparam TrialName      Compile-time trial-field name.
 * @tparam Mask           Operator mask describing which quantities are needed.
 * @tparam KernelContext  Kernel execution-context type.
 * @tparam QuadData       Cell quadrature-data type.
 * @tparam ElementDofsIn  Element-local trial-DOF container type.
 *
 * @param[in] k        Kernel execution context.
 * @param[in] qd       Cell quadrature data for the trial field.
 * @param[in] dofs_in  Element-local trial degrees of freedom.
 *
 * @return An `Entry<NameTag<TrialName>, InterpolatedField<...>>`.
 */
template<
   StaticString TrialName,
   OperatorMask Mask,
   class KernelContext,
   class QuadData,
   class ElementDofsIn>
GENDIL_HOST_DEVICE
auto MakeTrialInterpolatedEntryFromQD(
   KernelContext& k,
   const QuadData& qd,
   const ElementDofsIn& dofs_in)
{
   constexpr bool need_vals  = need_values(Mask);
   constexpr bool need_grads = need_gradients(Mask);

   using ValuesT = std::remove_cvref_t<
      decltype(details::InterpolateValuesIfNeeded<need_vals>(k, qd, dofs_in))>;

   using GradsT = std::remove_cvref_t<
      decltype(details::InterpolateGradientIfNeeded<need_grads>(k, qd, dofs_in))>;

   using IF = InterpolatedField<ValuesT, GradsT>;

   return Entry<NameTag<TrialName>, IF>{
      IF{
         details::InterpolateValuesIfNeeded<need_vals>(k, qd, dofs_in),
         details::InterpolateGradientIfNeeded<need_grads>(k, qd, dofs_in)
      }
   };
}

/**
 * @brief Build the interpolated entry associated with a supplied named
 *        finite-element field for a cell integration context.
 *
 * The field is read from the weak-form context on the specified element and
 * interpolated according to the requested value/gradient mask.
 *
 * @tparam Name            Compile-time named-field name.
 * @tparam KernelContext   Kernel execution-context type.
 * @tparam WeakFormContext Weak-form-context type.
 * @tparam QuadData        Cell quadrature-data type.
 *
 * @param[in] k              Kernel execution context.
 * @param[in] wf             Weak-form context containing the named field.
 * @param[in] qd             Cell quadrature data for the named field.
 * @param[in] element_index  Element index from which coefficient DOFs are read.
 *
 * @return An `Entry<NameTag<Name>, InterpolatedField<ValuesT, Empty>>`.
 */
template<
   StaticString Name,
   OperatorMask Mask,
   class KernelContext,
   class WeakFormContext,
   class QuadData>
GENDIL_HOST_DEVICE
auto MakeCoeffInterpolatedEntryFromQD(
   KernelContext& k,
   const WeakFormContext& wf,
   const QuadData& qd,
   const GlobalIndex element_index)
{
   constexpr bool need_vals  = need_values(Mask);
   constexpr bool need_grads = need_gradients(Mask);

   const auto& fev = wf.template fe_field<Name>();
   auto elem_dofs  = ReadDofs(k, fev.space, element_index, fev.dofs);

   using ValuesT = std::remove_cvref_t<
      decltype(details::InterpolateValuesIfNeeded<need_vals>(k, qd, elem_dofs))>;

   using GradsT = std::remove_cvref_t<
      decltype(details::InterpolateGradientIfNeeded<need_grads>(k, qd, elem_dofs))>;

   using IF      = InterpolatedField<ValuesT, GradsT>;

   return Entry<NameTag<Name>, IF>{
      IF{
         details::InterpolateValuesIfNeeded<need_vals>(k, qd, elem_dofs),
         details::InterpolateGradientIfNeeded<need_grads>(k, qd, elem_dofs)
      }
   };
}


/* -------------------------------------------------------------------------- */
/*                    Low-level facet interpolation helpers                    */
/* -------------------------------------------------------------------------- */

/**
 * @brief Build the interpolated entry associated with the trial field for one
 *        side of a facet.
 *
 * This helper forwards the facet-side object and the full facet quadrature-data
 * container to the existing face-aware interpolation routines:
 * @code
 * InterpolateValues(k, side, face_qd, dofs_in)
 * InterpolateGradient(k, side, face_qd, dofs_in)
 * @endcode
 *
 * This preserves local-face selection and conforming / non-conforming handling
 * already implemented elsewhere.
 *
 * @tparam TrialName      Compile-time trial-field name.
 * @tparam Mask           Operator mask describing which quantities are needed.
 * @tparam KernelContext  Kernel execution-context type.
 * @tparam FaceSide       Facet-side type.
 * @tparam FaceQuadData   Facet quadrature-data container type.
 * @tparam ElementDofsIn  Element-local trial-DOF container type.
 *
 * @param[in] k        Kernel execution context.
 * @param[in] side     Facet side.
 * @param[in] face_qd  Full facet quadrature-data container.
 * @param[in] dofs_in  Trial degrees of freedom on this side.
 *
 * @return An `Entry<NameTag<TrialName>, InterpolatedField<...>>`.
 */
template<
   StaticString TrialName,
   OperatorMask Mask,
   class KernelContext,
   class FaceSide,
   class FaceQuadData,
   class ElementDofsIn>
GENDIL_HOST_DEVICE
auto MakeFacetTrialInterpolatedEntry(
   KernelContext& k,
   const FaceSide& side,
   const FaceQuadData& face_qd,
   const ElementDofsIn& dofs_in)
{
   constexpr bool need_vals  = need_values(Mask);
   constexpr bool need_grads = need_gradients(Mask);

   using ValuesT = std::remove_cvref_t<
      decltype(details::InterpolateFacetValuesIfNeeded<need_vals>(k, side, face_qd, dofs_in))>;

   using GradsT = std::remove_cvref_t<
      decltype(details::InterpolateFacetGradientIfNeeded<need_grads>(k, side, face_qd, dofs_in))>;

   using IF = InterpolatedField<ValuesT, GradsT>;

   return Entry<NameTag<TrialName>, IF>{
      IF{
         details::InterpolateFacetValuesIfNeeded<need_vals>(k, side, face_qd, dofs_in),
         details::InterpolateFacetGradientIfNeeded<need_grads>(k, side, face_qd, dofs_in)
      }
   };
}

template<class FaceSide, class VolumeSpace, class FaceQuadData>
struct FacetFieldSideBinding
{
   FaceSide side;
   const VolumeSpace& volume_space;
   const FaceQuadData& qdata;
};

template<class Space>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetMinusFacetFieldVolumeSpace(const Space& space)
{
   using SpaceType = std::remove_cvref_t<Space>;

   if constexpr (is_face_field_binding_v<SpaceType>)
   {
      return space.GetMinusFiniteElementSpace();
   }
   else
   {
      return (space);
   }
}

template<class Space>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetPlusFacetFieldVolumeSpace(const Space& space)
{
   using SpaceType = std::remove_cvref_t<Space>;
   static_assert(
      !is_boundary_face_field_binding_v<SpaceType>,
      "Plus-side finite element field interpolation is not defined for "
      "boundary face field bindings.");

   if constexpr (is_face_field_binding_v<SpaceType>)
   {
      static_assert(
         is_interior_face_field_binding_v<SpaceType>,
         "Plus-side finite element field interpolation requires an interior "
         "face field binding or a homogeneous volume finite element space.");
      return space.GetPlusFiniteElementSpace();
   }
   else
   {
      return (space);
   }
}

template<class FaceContext, class FieldSpace, class QDataWrapper>
GENDIL_HOST_DEVICE
constexpr auto MakeMinusFacetFieldBinding(
   const FaceContext& face_info,
   const FieldSpace& field_space,
   const QDataWrapper& qdata_wrapper)
{
   decltype(auto) volume_space =
      GetMinusFacetFieldVolumeSpace(field_space);
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

template<class FaceContext, class FieldSpace, class QDataWrapper>
GENDIL_HOST_DEVICE
constexpr auto MakePlusFacetFieldBinding(
   const FaceContext& face_info,
   const FieldSpace& field_space,
   const QDataWrapper& qdata_wrapper)
{
   decltype(auto) volume_space =
      GetPlusFacetFieldVolumeSpace(field_space);
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
 * @brief Build a masked interpolated entry associated with a supplied named
 *        finite-element field on one side of a facet.
 *
 * This is used by boundary facets and by the side-specific interior-facet
 * maps. Coefficient input tags and explicit finite-element field expressions
 * share the same named-field requirement model as cell integration.
 */
template<
   StaticString Name,
   OperatorMask Mask,
   class KernelContext,
   class WeakFormContext,
   class Binding>
GENDIL_HOST_DEVICE
auto MakeFacetCoeffInterpolatedEntryWithMask(
   KernelContext& k,
   const WeakFormContext& wf,
   const Binding& binding)
{
   constexpr bool need_vals  = need_values(Mask);
   constexpr bool need_grads = need_gradients(Mask);

   const auto& fev = wf.template fe_field<Name>();
   auto elem_dofs  =
      ReadDofs(k, binding.volume_space, binding.side.GetCellIndex(), fev.dofs);

   using ValuesT = std::remove_cvref_t<
      decltype(details::InterpolateFacetValuesIfNeeded<need_vals>(
         k,
         binding.side,
         binding.qdata,
         elem_dofs))>;

   using GradsT = std::remove_cvref_t<
      decltype(details::InterpolateFacetGradientIfNeeded<need_grads>(
         k,
         binding.side,
         binding.qdata,
         elem_dofs))>;

   using IF = InterpolatedField<ValuesT, GradsT>;

   return Entry<NameTag<Name>, IF>{
      IF{
         details::InterpolateFacetValuesIfNeeded<need_vals>(
            k,
            binding.side,
            binding.qdata,
            elem_dofs),
         details::InterpolateFacetGradientIfNeeded<need_grads>(
            k,
            binding.side,
            binding.qdata,
            elem_dofs)
      }
   };
}

/**
 * @brief Build one named-field entry on one side of a facet.
 *
 * Active-trial requirements are interpolated from the side-specific trial DOF
 * container. Supplied finite-element fields are read from the weak-form context
 * on the side's cell and interpolated with the same value/gradient mask.
 */
template<
   StaticString TrialName,
   class Req,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceContext,
   class ElementDofsIn>
GENDIL_HOST_DEVICE
auto MakeMinusFacetNamedFieldInterpolatedEntry(
   KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const FaceContext&     face_info,
   const ElementDofsIn&   dofs_in)
{
   constexpr auto Name = Req::name;
   constexpr auto Mask = Req::mask;

   if constexpr (Name == TrialName)
   {
      static_assert(
         !has_provenance(Req::provenance, NamedFieldProvenance::FiniteElementExpression),
         "InterpolateFields(facet): active trial field name conflicts with a "
         "FiniteElementField expression name.");

      const auto& fev = wf.template fe_field<TrialName>();
      const auto& qd_wrapper =
         op.template finite_element_facet_quad_data<TrialName>();
      const auto binding =
         MakeMinusFacetFieldBinding(face_info, fev.space, qd_wrapper);

      return MakeFacetTrialInterpolatedEntry<TrialName, Mask>(
         k,
         binding.side,
         binding.qdata,
         dofs_in);
   }
   else
   {
      const auto& fev = wf.template fe_field<Name>();
      const auto& qd_wrapper =
         op.template finite_element_facet_quad_data<Name>();
      const auto binding =
         MakeMinusFacetFieldBinding(face_info, fev.space, qd_wrapper);

      return MakeFacetCoeffInterpolatedEntryWithMask<Name, Mask>(
         k,
         wf,
         binding);
   }
}

template<
   StaticString TrialName,
   class Req,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceContext,
   class ElementDofsIn>
GENDIL_HOST_DEVICE
auto MakePlusFacetNamedFieldInterpolatedEntry(
   KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const FaceContext&     face_info,
   const ElementDofsIn&   dofs_in)
{
   constexpr auto Name = Req::name;
   constexpr auto Mask = Req::mask;

   if constexpr (Name == TrialName)
   {
      static_assert(
         !has_provenance(Req::provenance, NamedFieldProvenance::FiniteElementExpression),
         "InterpolateFields(facet): active trial field name conflicts with a "
         "FiniteElementField expression name.");

      const auto& fev = wf.template fe_field<TrialName>();
      const auto& qd_wrapper =
         op.template finite_element_facet_quad_data<TrialName>();
      const auto binding =
         MakePlusFacetFieldBinding(face_info, fev.space, qd_wrapper);

      return MakeFacetTrialInterpolatedEntry<TrialName, Mask>(
         k,
         binding.side,
         binding.qdata,
         dofs_in);
   }
   else
   {
      const auto& fev = wf.template fe_field<Name>();
      const auto& qd_wrapper =
         op.template finite_element_facet_quad_data<Name>();
      const auto binding =
         MakePlusFacetFieldBinding(face_info, fev.space, qd_wrapper);

      return MakeFacetCoeffInterpolatedEntryWithMask<Name, Mask>(
         k,
         wf,
         binding);
   }
}


/* -------------------------------------------------------------------------- */
/*                         Internal facet-map builders                         */
/* -------------------------------------------------------------------------- */

/**
 * @brief Build the one-entry map containing the trial field on one side of a
 *        facet.
 *
 * @tparam KernelContext  Kernel execution-context type.
 * @tparam Integrand      Integrand type.
 * @tparam FaceSide       Facet-side type.
 * @tparam FaceQuadData   Facet quadrature-data container type.
 * @tparam ElementDofsIn  Trial-DOF container type on this side.
 *
 * @param[in] k         Kernel execution context.
 * @param[in] side      Facet side.
 * @param[in] face_qd   Full facet quadrature-data container for the trial field.
 * @param[in] integrand Integrand instance; only its type is inspected.
 * @param[in] dofs_in   Trial degrees of freedom on this side.
 *
 * @return A map-like object containing exactly one entry for the trial field.
 */
template<
   class KernelContext,
   class Integrand,
   class FaceSide,
   class FaceQuadData,
   class ElementDofsIn>
GENDIL_HOST_DEVICE
auto MakeFacetTrialFieldsMap(
   KernelContext& k,
   const FaceSide&      side,
   const FaceQuadData&  face_qd,
   const Integrand&     /*integrand*/,
   const ElementDofsIn& dofs_in)
{
   using I = std::remove_cvref_t<Integrand>;

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TrialMask = requirements<I>::trial_mask;

   static_assert(TrialName != StaticString("Error"),
                 "MakeFacetTrialFieldsMap: trial_name == \"Error\". "
                 "Integrand must contain a TrialSpace.");

   return make_map(
      MakeFacetTrialInterpolatedEntry<TrialName, TrialMask>(k, side, face_qd, dofs_in)
   );
}

/* -------------------------------------------------------------------------- */
/*                            Public cell overload                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief Helper for InterpolateFields (cell) that is templated on TrialName/TrialMask.
 *
 * This helper avoids using TrialName as a captured variable in a generic lambda,
 * which causes template argument deduction issues when TrialName appears in
 * NameTag<TrialName>.
 */
template<
   StaticString TrialName,
   class Req,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class ElementContext,
   class ElementDofsIn>
GENDIL_HOST_DEVICE
auto MakeCellNamedFieldInterpolatedEntry(
   KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const ElementContext&  ec,
   const ElementDofsIn&   dofs_in)
{
   constexpr auto Name = Req::name;
   constexpr auto Mask = Req::mask;

   if constexpr (Name == TrialName)
   {
      static_assert(
         !has_provenance(Req::provenance, NamedFieldProvenance::FiniteElementExpression),
         "InterpolateFields(cell): active trial field name conflicts with a "
         "FiniteElementField expression name.");

      return MakeTrialInterpolatedEntryFromQD<TrialName, Mask>(
         k,
         op.template finite_element_quad_data<TrialName>(),
         dofs_in);
   }
   else
   {
      return MakeCoeffInterpolatedEntryFromQD<Name, Mask>(
         k,
         wf,
         op.template finite_element_quad_data<Name>(),
         ec.element_index);
   }
}

template<
   StaticString TrialName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class ElementContext,
   class ElementDofsIn,
   class... Reqs>
GENDIL_HOST_DEVICE
auto InterpolateFieldsCellImpl(
   KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const ElementContext&  ec,
   const ElementDofsIn&   dofs_in,
   type_list<Reqs...>)
{
   return make_map(
      MakeCellNamedFieldInterpolatedEntry<TrialName, Reqs>(
         k, wf, op, ec, dofs_in)...
   );
}

template<
   StaticString TrialName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceContext,
   class ElementDofsIn,
   class... Reqs>
GENDIL_HOST_DEVICE
auto MakeMinusFacetNamedFieldsMap(
   KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const FaceContext&     face_info,
   const ElementDofsIn&   dofs_in,
   type_list<Reqs...>)
{
   return make_map(
      MakeMinusFacetNamedFieldInterpolatedEntry<TrialName, Reqs>(
         k, wf, op, face_info, dofs_in)...
   );
}

template<
   StaticString TrialName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceContext,
   class ElementDofsIn,
   class... Reqs>
GENDIL_HOST_DEVICE
auto MakePlusFacetNamedFieldsMap(
   KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const FaceContext&     face_info,
   const ElementDofsIn&   dofs_in,
   type_list<Reqs...>)
{
   return make_map(
      MakePlusFacetNamedFieldInterpolatedEntry<TrialName, Reqs>(
         k, wf, op, face_info, dofs_in)...
   );
}

template<
   StaticString TrialName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceContext,
   class ElementDofsInMinus,
   class ElementDofsInPlus,
   class... Reqs>
GENDIL_HOST_DEVICE
auto InterpolateFieldsInteriorFacetImpl(
   KernelContext&      k,
   const WeakFormContext&    wf,
   const OperatorContext&    op,
   const FaceContext&        face_info,
   const ElementDofsInMinus& dofs_in_minus,
   const ElementDofsInPlus&  dofs_in_plus,
   type_list<Reqs...> deps)
{
   auto minus_fields =
      MakeMinusFacetNamedFieldsMap<TrialName>(
         k, wf, op, face_info, dofs_in_minus, deps);

   auto plus_fields =
      MakePlusFacetNamedFieldsMap<TrialName>(
         k, wf, op, face_info, dofs_in_plus, deps);

   return FaceFields<
      std::remove_cvref_t<decltype(minus_fields)>,
      std::remove_cvref_t<decltype(plus_fields)>>{
         minus_fields,
         plus_fields
      };
}

/**
 * @brief Interpolate all fields required by a cell integrand.
 *
 * This overload is intended for cell / element integration. It returns one flat
 * map containing:
 *
 * - the trial field required by the integrand,
 * - all named finite-element fields required by explicit field expressions or
 *   coefficient input tags.
 *
 * The interpolation mask is supplied by
 * `interpolation_named_field_requirements_t<Integrand>`, which unions active
 * trial, explicit field-expression, and coefficient-input requirements by
 * field name.
 *
 * @tparam KernelContext    Kernel execution-context type.
 * @tparam WeakFormContext  Weak-form-context type.
 * @tparam OperatorContext  Operator-context type.
 * @tparam ElementContext   Element-context type. Must provide `element_index`.
 * @tparam Integrand        Integrand type.
 * @tparam ElementDofsIn    Element-local trial-DOF container type.
 *
 * @param[in] k         Kernel execution context.
 * @param[in] wf        Weak-form context.
 * @param[in] op        Operator context.
 * @param[in] ec        Element context.
 * @param[in] integrand Integrand instance; only its type is inspected.
 * @param[in] dofs_in   Element-local trial degrees of freedom.
 *
 * @return A flat map containing the trial field and all referenced named
 * finite-element fields.
 */
template<
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class ElementContext,
   class Integrand,
   class ElementDofsIn>
   requires CellIntegrand<Integrand>
GENDIL_HOST_DEVICE
auto InterpolateFields(
   KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const ElementContext&  ec,
   const Integrand&       /*integrand*/,
   const ElementDofsIn&   dofs_in)
{
   using I = std::remove_cvref_t<Integrand>;

   static_assert(requirements<I>::trial_name != StaticString("Error"),
                 "InterpolateFields(cell): trial_name == \"Error\". "
                 "Integrand must contain a TrialSpace.");

   using Deps = interpolation_named_field_requirements_t<I>;

   return InterpolateFieldsCellImpl<
      requirements<I>::trial_name>(k, wf, op, ec, dofs_in, Deps{});
}


/* -------------------------------------------------------------------------- */
/*                           Public facet overload                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief Interpolate all fields required by a facet integrand.
 *
 * This overload is intended for facet integration. It returns a `FaceFields`
 * object separating:
 *
 * - minus-side named-field traces,
 * - plus-side named-field traces.
 *
 * More precisely:
 *
 * - `minus_fields` contains every named-field requirement interpolated with
 *   `fc.MinusSide()`;
 *
 * - `plus_fields` contains every named-field requirement interpolated with
 *   `fc.PlusSide()`.
 *
 * @tparam KernelContext      Kernel execution-context type.
 * @tparam WeakFormContext    Weak-form-context type.
 * @tparam OperatorContext    Operator-context type.
 * @tparam FaceContext        Face-context type. Must provide `MinusSide()` and
 *                            `PlusSide()`. Each side must provide
 *                            `GetCellIndex()`.
 * @tparam Integrand          Integrand type.
 * @tparam ElementDofsInMinus Type of trial DOFs on the minus side.
 * @tparam ElementDofsInPlus  Type of trial DOFs on the plus side.
 *
 * @param[in] k              Kernel execution context.
 * @param[in] wf             Weak-form context.
 * @param[in] op             Operator context.
 * @param[in] fc             Face context.
 * @param[in] integrand      Integrand instance; only its type is inspected.
 * @param[in] dofs_in_minus  Trial degrees of freedom on the minus side.
 * @param[in] dofs_in_plus   Trial degrees of freedom on the plus side.
 *
 * @return A `FaceFields<MinusFields, PlusFields>` object.
 */
template<
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceContext,
   class Integrand,
   class ElementDofsInMinus,
   class ElementDofsInPlus>
GENDIL_HOST_DEVICE
auto InterpolateFields(
   KernelContext&      k,
   const WeakFormContext&    wf,
   const OperatorContext&    op,
   const FaceContext&        fc,
   const Integrand&          /*integrand*/,
   const ElementDofsInMinus& dofs_in_minus,
   const ElementDofsInPlus&  dofs_in_plus)
{
   using I = std::remove_cvref_t<Integrand>;

   constexpr auto TrialName = requirements<I>::trial_name;

   static_assert(TrialName != StaticString("Error"),
                 "InterpolateFields(facet): trial_name == \"Error\". "
                 "Integrand must contain a TrialSpace.");

   static_assert(
      !has_invalid_unqualified_interior_side_dependencies_v<I, WeakFormContext, FaceContext>,
      "Side-dependent expression appears on an interior facet without side "
      "selection. Use minus(expr), plus(expr), average(expr), jump(expr), "
      "or a trace-aware operator such as upwind(...).");

   using Deps = interpolation_named_field_requirements_t<I>;

   return InterpolateFieldsInteriorFacetImpl<TrialName>(
      k,
      wf,
      op,
      fc,
      dofs_in_minus,
      dofs_in_plus,
      Deps{});
}

/**
 * @brief Helper for InterpolateFields (boundary facet) that is templated on
 *        TrialName and one named-field requirement.
 *
 * This helper avoids using TrialName as a captured variable in a generic lambda,
 * which causes template argument deduction issues when TrialName appears in
 * NameTag<TrialName>.
 */
template<
   StaticString TrialName,
   class Req,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceContext,
   class ElementDofsIn>
GENDIL_HOST_DEVICE
auto MakeBoundaryFacetNamedFieldInterpolatedEntry(
   KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const FaceContext&     face_info,
   const ElementDofsIn&   dofs_in)
{
   constexpr auto Name = Req::name;
   constexpr auto Mask = Req::mask;

   if constexpr (Name == TrialName)
   {
      static_assert(
         !has_provenance(Req::provenance, NamedFieldProvenance::FiniteElementExpression),
         "InterpolateFields(boundary facet): active trial field name conflicts with a "
         "FiniteElementField expression name.");

      const auto& fev = wf.template fe_field<TrialName>();
      const auto binding =
         MakeMinusFacetFieldBinding(
            face_info,
            fev.space,
            op.template finite_element_facet_quad_data<TrialName>());
      return MakeFacetTrialInterpolatedEntry<TrialName, Mask>(
         k,
         binding.side,
         binding.qdata,
         dofs_in);
   }
   else
   {
      const auto& fev = wf.template fe_field<Name>();
      const auto binding =
         MakeMinusFacetFieldBinding(
            face_info,
            fev.space,
            op.template finite_element_facet_quad_data<Name>());
      return MakeFacetCoeffInterpolatedEntryWithMask<Name, Mask>(
         k,
         wf,
         binding);
   }
}

template<
   StaticString TrialName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceContext,
   class ElementDofsIn,
   class... Reqs>
GENDIL_HOST_DEVICE
auto InterpolateFieldsBoundaryFacetImpl(
   KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const FaceContext&     face_info,
   const ElementDofsIn&   dofs_in,
   type_list<Reqs...>)
{
   return make_map(
      MakeBoundaryFacetNamedFieldInterpolatedEntry<TrialName, Reqs>(
         k, wf, op, face_info, dofs_in)...
   );
}

/**
 * @brief Interpolate all fields required by a boundary facet integrand.
 *
 * This overload is intended for boundary facet integration. It returns one flat
 * map containing:
 *
 * - the trial field required by the integrand (from interior/minus side only),
 * - all named finite-element fields required by explicit field expressions or
 *   coefficient input tags (from interior/minus side only).
 *
 * The interpolation mask is supplied by
 * `interpolation_named_field_requirements_t<Integrand>`, which unions active
 * trial, explicit field-expression, and coefficient-input requirements by
 * field name.
 *
 * @tparam KernelContext    Kernel execution-context type.
 * @tparam WeakFormContext  Weak-form-context type.
 * @tparam OperatorContext  Operator-context type.
 * @tparam FaceContext      Face-context type (GlobalFaceInfo).
 * @tparam Integrand        Integrand type.
 * @tparam ElementDofsIn    Element-local trial-DOF container type.
 *
 * @param[in] k         Kernel execution context.
 * @param[in] wf        Weak-form context.
 * @param[in] op        Operator context.
 * @param[in] fc        Face context (two-sided, but plus has boundary info).
 * @param[in] integrand Integrand instance; only its type is inspected.
 * @param[in] dofs_in   Element-local trial degrees of freedom (minus side).
 *
 * @return A flat map containing all interpolated fields.
 */
template<
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceContext,
   class Integrand,
   class ElementDofsIn>
   requires BoundaryFacetIntegrand<Integrand>
GENDIL_HOST_DEVICE
auto InterpolateFields(
   KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const FaceContext&     fc,
   const Integrand&       /*integrand*/,
   const ElementDofsIn&   dofs_in)
{
   using I = std::remove_cvref_t<Integrand>;

   static_assert(requirements<I>::trial_name != StaticString("Error"),
                 "InterpolateFields(boundary facet): trial_name == \"Error\". "
                 "Integrand must contain a TrialSpace.");

   using Deps = interpolation_named_field_requirements_t<I>;

   return InterpolateFieldsBoundaryFacetImpl<
      requirements<I>::trial_name>(k, wf, op, fc, dofs_in, Deps{});
}

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/Utilities/staticmap.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"

namespace gendil
{

/**
 * @file interpolatefields.hpp
 * @brief Utilities to interpolate trial and finite-element coefficient fields
 *        at quadrature points for cell and facet integrals.
 *
 * This file provides the interpolation layer that transforms:
 *
 * - element-local trial degrees of freedom, and
 * - finite-element coefficient fields stored in the weak-form context,
 *
 * into map-like structures containing interpolated values and/or gradients at
 * quadrature points.
 *
 * Two integration situations are handled:
 *
 * - **cell integration**
 *   returns one flat map containing the trial field and all referenced
 *   finite-element coefficient fields;
 *
 * - **facet integration**
 *   returns a `FaceFields` object containing:
 *   - minus-side trial traces,
 *   - plus-side trial traces,
 *   - minus-side finite-element coefficient fields.
 *
 * The facet path intentionally preserves the current face-aware interpolation
 * model:
 *
 * - the operator context supplies the **full per-face quadrature-data
 *   container**,
 * - the facet-side object (`fc.MinusSide()` / `fc.PlusSide()`) is passed into
 *   the interpolation routines,
 * - selection of the local-face entry and conforming / non-conforming handling
 *   are delegated to the existing overloads of `InterpolateValues` and
 *   `InterpolateGradient`.
 *
 * Only finite-element coefficient fields are interpolated through this layer.
 * Non-finite-element coefficients must be handled elsewhere.
 *
 * @note This file assumes the existence of:
 * - `requirements<Integrand>`
 * - `fe_field_deps_t<Expr>`
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
 * @tparam MinusFields  Map-like type storing minus-side trial/test fields.
 * @tparam PlusFields   Map-like type storing plus-side trial/test fields.
 * @tparam CoeffFields  Map-like type storing minus-side finite-element
 *                      coefficient fields.
 *
 * The asymmetry is intentional:
 *
 * - trial/test traces live on both sides of the facet,
 * - finite-element coefficient fields are evaluated only on the minus side.
 */
template<class MinusFields, class PlusFields, class CoeffFields>
struct FaceFields
{
   /// Minus-side trial/test fields.
   MinusFields minus_fields;

   /// Plus-side trial/test fields.
   PlusFields plus_fields;

   /// Minus-side finite-element coefficient fields.
   CoeffFields coeff_fields;
};


/* -------------------------------------------------------------------------- */
/*                          Internal lazy helper API                          */
/* -------------------------------------------------------------------------- */

namespace details
{

template<bool Need, class KernelContext, class QuadData, class ElementDofsIn>
GENDIL_HOST_DEVICE
auto InterpolateValuesIfNeeded(
   const KernelContext& k,
   const QuadData& qd,
   const ElementDofsIn& dofs_in)
{
   if constexpr (Need) return InterpolateValues(k, qd, dofs_in);
   else                return Empty{};
}

template<bool Need, class KernelContext, class QuadData, class ElementDofsIn>
GENDIL_HOST_DEVICE
auto InterpolateGradientIfNeeded(
   const KernelContext& k,
   const QuadData& qd,
   const ElementDofsIn& dofs_in)
{
   if constexpr (Need) return InterpolateGradient(k, qd, dofs_in);
   else                return Empty{};
}

template<bool Need, class KernelContext, class FaceSide, class FaceQuadData, class ElementDofsIn>
GENDIL_HOST_DEVICE
auto InterpolateFacetValuesIfNeeded(
   const KernelContext& k,
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
   const KernelContext& k,
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
   const KernelContext& k,
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
 * @brief Build the interpolated entry associated with a finite-element
 *        coefficient field for a cell integration context.
 *
 * The coefficient field is read from the weak-form context on the specified
 * element and interpolated in value form only.
 *
 * @tparam Name            Compile-time coefficient-field name.
 * @tparam KernelContext   Kernel execution-context type.
 * @tparam WeakFormContext Weak-form-context type.
 * @tparam QuadData        Cell quadrature-data type.
 *
 * @param[in] k              Kernel execution context.
 * @param[in] wf             Weak-form context containing the coefficient field.
 * @param[in] qd             Cell quadrature data for the coefficient field.
 * @param[in] element_index  Element index from which coefficient DOFs are read.
 *
 * @return An `Entry<NameTag<Name>, InterpolatedField<ValuesT, Empty>>`.
 */
template<
   StaticString Name,
   class KernelContext,
   class WeakFormContext,
   class QuadData>
GENDIL_HOST_DEVICE
auto MakeCoeffInterpolatedEntryFromQD(
   const KernelContext& k,
   const WeakFormContext& wf,
   const QuadData& qd,
   const GlobalIndex element_index)
{
   const auto& fev = wf.template fe_field<Name>();
   auto elem_dofs  = ReadDofs(k, fev.space, element_index, fev.dofs);

   using ValuesT = std::remove_cvref_t<decltype(InterpolateValues(k, qd, elem_dofs))>;
   using IF      = InterpolatedField<ValuesT, Empty>;

   return Entry<NameTag<Name>, IF>{
      IF{
         InterpolateValues(k, qd, elem_dofs),
         Empty{}
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
   const KernelContext& k,
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

/**
 * @brief Build the interpolated entry associated with a finite-element
 *        coefficient field on one side of a facet.
 *
 * The coefficient field is read from the cell identified by
 * `side.GetCellIndex()` and interpolated in value form only using the existing
 * face-aware interpolation routine.
 *
 * @tparam Name            Compile-time coefficient-field name.
 * @tparam KernelContext   Kernel execution-context type.
 * @tparam WeakFormContext Weak-form-context type.
 * @tparam FaceSide        Facet-side type.
 * @tparam FaceQuadData    Facet quadrature-data container type.
 *
 * @param[in] k        Kernel execution context.
 * @param[in] wf       Weak-form context containing the coefficient field.
 * @param[in] side     Facet side used both to select the local face and to
 *                     determine the cell index.
 * @param[in] face_qd  Full facet quadrature-data container.
 *
 * @return An `Entry<NameTag<Name>, InterpolatedField<ValuesT, Empty>>`.
 */
template<
   StaticString Name,
   class KernelContext,
   class WeakFormContext,
   class FaceSide,
   class FaceQuadData>
GENDIL_HOST_DEVICE
auto MakeFacetCoeffInterpolatedEntry(
   const KernelContext& k,
   const WeakFormContext& wf,
   const FaceSide& side,
   const FaceQuadData& face_qd)
{
   const auto& fev = wf.template fe_field<Name>();
   auto elem_dofs  = ReadDofs(k, fev.space, side.GetCellIndex(), fev.dofs);

   using ValuesT = std::remove_cvref_t<decltype(InterpolateValues(k, side, face_qd, elem_dofs))>;
   using IF      = InterpolatedField<ValuesT, Empty>;

   return Entry<NameTag<Name>, IF>{
      IF{
         InterpolateValues(k, side, face_qd, elem_dofs),
         Empty{}
      }
   };
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
   const KernelContext& k,
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

/**
 * @brief Helper for MakeFacetCoeffFieldsMap that is templated on TrialName.
 *
 * This helper avoids using TrialName as a captured variable in a generic lambda,
 * which causes template argument deduction issues when TrialName appears in
 * NameTag<TrialName>.
 */
template<
   StaticString TrialName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceSide,
   class... Tags>
GENDIL_HOST_DEVICE
auto MakeFacetCoeffFieldsMapImpl(
   const KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const FaceSide&        side,
   type_list<Tags...>)
{
   static_assert((!std::is_same_v<NameTag<TrialName>, Tags> && ...),
                 "MakeFacetCoeffFieldsMap: trial field name conflicts with a "
                 "FiniteElementField name.");

   return make_map(
      MakeFacetCoeffInterpolatedEntry<Tags::name>(
         k,
         wf,
         side,
         op.template finite_element_facet_quad_data<Tags::name>())...
   );
}

/**
 * @brief Build the map containing all finite-element coefficient fields
 *        referenced by a facet integrand expression, evaluated on one side.
 *
 * This helper is intended for the asymmetric facet representation used here:
 * coefficient fields are interpolated only on the minus side and stored in
 * `FaceFields::coeff_fields`.
 *
 * @tparam KernelContext    Kernel execution-context type.
 * @tparam WeakFormContext  Weak-form-context type.
 * @tparam OperatorContext  Operator-context type.
 * @tparam Integrand        Integrand type.
 * @tparam FaceSide         Facet-side type.
 *
 * @param[in] k         Kernel execution context.
 * @param[in] wf        Weak-form context.
 * @param[in] op        Operator context.
 * @param[in] side      Facet side from which coefficient fields are
 *                      interpolated.
 * @param[in] integrand Integrand instance; only its type is inspected.
 *
 * @return A map-like object containing one entry per FE coefficient field
 * referenced by the integrand expression. This may be `StaticMap<>` when the
 * dependency list is empty.
 */
template<
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class Integrand,
   class FaceSide>
GENDIL_HOST_DEVICE
auto MakeFacetCoeffFieldsMap(
   const KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const FaceSide&        side,
   const Integrand&       /*integrand*/)
{
   using I = std::remove_cvref_t<Integrand>;

   using Expr = typename I::expression_type;
   using Deps = fe_field_deps_t<Expr>;

   return MakeFacetCoeffFieldsMapImpl<requirements<I>::trial_name>(
      k, wf, op, side, Deps{});
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
   OperatorMask TrialMask,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class ElementContext,
   class ElementDofsIn,
   class... Tags>
GENDIL_HOST_DEVICE
auto InterpolateFieldsCellImpl(
   const KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const ElementContext&  ec,
   const ElementDofsIn&   dofs_in,
   type_list<Tags...>)
{
   static_assert((!std::is_same_v<NameTag<TrialName>, Tags> && ...),
                 "InterpolateFields(cell): trial field name conflicts with a "
                 "FiniteElementField name.");

   return make_map(
      MakeTrialInterpolatedEntryFromQD<TrialName, TrialMask>(
         k,
         op.template finite_element_quad_data<TrialName>(),
         dofs_in),

      MakeCoeffInterpolatedEntryFromQD<Tags::name>(
         k,
         wf,
         op.template finite_element_quad_data<Tags::name>(),
         ec.element_index)...
   );
}

/**
 * @brief Interpolate all fields required by a cell integrand.
 *
 * This overload is intended for cell / element integration. It returns one flat
 * map containing:
 *
 * - the trial field required by the integrand,
 * - all finite-element coefficient fields referenced by the integrand
 *   expression.
 *
 * The trial field interpolation is controlled by
 * `requirements<Integrand>::trial_mask`. Finite-element coefficient fields are
 * interpolated in value form only.
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
 * @return A flat map containing the trial field and all referenced finite-
 * element coefficient fields.
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
   const KernelContext&   k,
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

   using Expr = typename I::expression_type;
   using Deps = fe_field_deps_t<Expr>;

   return InterpolateFieldsCellImpl<
      requirements<I>::trial_name,
      requirements<I>::trial_mask>(k, wf, op, ec, dofs_in, Deps{});
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
 * - minus-side trial/test traces,
 * - plus-side trial/test traces,
 * - minus-side finite-element coefficient fields.
 *
 * More precisely:
 *
 * - `minus_fields` contains the trial field interpolated with
 *   `fc.MinusSide()` and the full facet FE quadrature-data container for the
 *   trial field;
 *
 * - `plus_fields` contains the trial field interpolated with
 *   `fc.PlusSide()` and the full facet FE quadrature-data container for the
 *   trial field;
 *
 * - `coeff_fields` contains FE coefficient fields interpolated on the minus
 *   side only.
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
 * @return A `FaceFields<MinusFields, PlusFields, CoeffFields>` object.
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
   const KernelContext&      k,
   const WeakFormContext&    wf,
   const OperatorContext&    op,
   const FaceContext&        fc,
   const Integrand&          integrand,
   const ElementDofsInMinus& dofs_in_minus,
   const ElementDofsInPlus&  dofs_in_plus)
{
   using I = std::remove_cvref_t<Integrand>;

   constexpr auto TrialName = requirements<I>::trial_name;

   static_assert(TrialName != StaticString("Error"),
                 "InterpolateFields(facet): trial_name == \"Error\". "
                 "Integrand must contain a TrialSpace.");

   // Store the side objects by value so this remains correct whether
   // MinusSide()/PlusSide() return references or proxy temporaries.
   const auto minus_side = fc.MinusSide();
   const auto plus_side  = fc.PlusSide();

   auto minus_fields =
      MakeFacetTrialFieldsMap(
         k,
         minus_side,
         op.template finite_element_facet_quad_data<TrialName>(),
         integrand,
         dofs_in_minus);

   auto plus_fields =
      MakeFacetTrialFieldsMap(
         k,
         plus_side,
         op.template finite_element_facet_quad_data<TrialName>(),
         integrand,
         dofs_in_plus);

   auto coeff_fields =
      MakeFacetCoeffFieldsMap(
         k,
         wf,
         op,
         minus_side,
         integrand);

   return FaceFields<
      std::remove_cvref_t<decltype(minus_fields)>,
      std::remove_cvref_t<decltype(plus_fields)>,
      std::remove_cvref_t<decltype(coeff_fields)>>{
         minus_fields,
         plus_fields,
         coeff_fields
      };
}

/**
 * @brief Helper for InterpolateFields (boundary facet) that is templated on TrialName/TrialMask.
 *
 * This helper avoids using TrialName as a captured variable in a generic lambda,
 * which causes template argument deduction issues when TrialName appears in
 * NameTag<TrialName>.
 */
template<
   StaticString TrialName,
   OperatorMask TrialMask,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceSide,
   class ElementDofsIn,
   class... Tags>
GENDIL_HOST_DEVICE
auto InterpolateFieldsBoundaryFacetImpl(
   const KernelContext&   k,
   const WeakFormContext& wf,
   const OperatorContext& op,
   const FaceSide&        minus_side,
   const ElementDofsIn&   dofs_in,
   type_list<Tags...>)
{
   static_assert((!std::is_same_v<NameTag<TrialName>, Tags> && ...),
                 "InterpolateFields(boundary facet): trial field name conflicts with a "
                 "FiniteElementField name.");

   return make_map(
      MakeFacetTrialInterpolatedEntry<TrialName, TrialMask>(
         k,
         minus_side,
         op.template finite_element_facet_quad_data<TrialName>(),
         dofs_in),

      MakeFacetCoeffInterpolatedEntry<Tags::name>(
         k,
         wf,
         minus_side,
         op.template finite_element_facet_quad_data<Tags::name>())...
   );
}

/**
 * @brief Interpolate all fields required by a boundary facet integrand.
 *
 * This overload is intended for boundary facet integration. It returns one flat
 * map containing:
 *
 * - the trial field required by the integrand (from interior/minus side only),
 * - all finite-element coefficient fields referenced by the integrand
 *   expression (from interior/minus side only).
 *
 * The trial field interpolation is controlled by
 * `requirements<Integrand>::trial_mask`. Finite-element coefficient fields are
 * interpolated in value form only.
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
   const KernelContext&   k,
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

   using Expr = typename I::expression_type;
   using Deps = fe_field_deps_t<Expr>;

   // Boundary facets are one-sided: use minus side only
   const auto minus_side = fc.MinusSide();

   return InterpolateFieldsBoundaryFacetImpl<
      requirements<I>::trial_name,
      requirements<I>::trial_mask>(k, wf, op, minus_side, dofs_in, Deps{});
}

} // namespace gendil

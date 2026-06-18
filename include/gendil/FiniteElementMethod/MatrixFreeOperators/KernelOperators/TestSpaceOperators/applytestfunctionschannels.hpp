// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applyvaluesandgradienttestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/operatorcontext.hpp"

namespace gendil {

template <
   typename KernelContext,
   typename OperatorContext,
   typename Integrand,
   typename QuadraturePointData,
   typename DofsOut >
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(
   KernelContext& kernel_context,
   const OperatorContext& operator_context,
   const Integrand& integrand,
   const QuadraturePointData& Du,
   DofsOut& dofs_out)
{
   // CRITICAL: Do NOT use ApplyAddTestFunctions(..., integrand, Du, dofs_out)
   // That would reintroduce requirements<Integrand>::test_mask limitation
   constexpr auto TestName = requirements<Integrand>::test_name;

   // Get test quad data from operator context (TestName already extracted above)
   const auto& test_qd =
      operator_context.template finite_element_quad_data<TestName>();

   // Channel presence (compile-time)
   constexpr bool has_value_channel = need_values(requirements<Integrand>::test_mask);
   constexpr bool has_gradient_channel = need_gradients(requirements<Integrand>::test_mask);

   // Explicit dispatch based on channels (NOT requirements<Integrand>::test_mask)
   if constexpr (has_value_channel && has_gradient_channel)
   {
      // Mixed: both value and gradient channels
      // Signature: ApplyValuesAndGradientTestFunctions<Add>(KernelContext, ElementDofToQuad, ValuesInput, GradientsInput, Output)
      ApplyValuesAndGradientTestFunctions<true>(
         kernel_context,
         test_qd,
         Du.values,
         Du.gradients,
         dofs_out);
   }
   else if constexpr (has_value_channel)
   {
      // Value channel only
      // Signature: ApplyAddTestFunctions(KernelContext, ElementDofToQuad, Values, Output)
      ApplyAddTestFunctions(
         kernel_context,
         test_qd,
         Du.values,
         dofs_out);
   }
   else if constexpr (has_gradient_channel)
   {
      // Gradient channel only (two-step: G^T * Du.gradients, then B^T * tmp_q)
      // Pattern from applytestfunctionsgeneric.hpp gradient-only case
      auto tmp_q = ApplyGradientTestFunctionsAtQPoints(
         kernel_context,
         test_qd,
         Du.gradients);
      ApplyAddTestFunctions(
         kernel_context,
         test_qd,
         tmp_q,
         dofs_out);
   }
}


template <
   typename KernelContext,
   typename FaceSide,
   typename FacetTestQData,
   typename Integrand,
   typename QuadraturePointData,
   typename DofsOut >
GENDIL_HOST_DEVICE
void ApplyAddFacetTestFunctions(
   KernelContext& kernel_context,
   const FaceSide& side,
   const FacetTestQData& test_qd,
   const Integrand&,
   const QuadraturePointData& Du,
   DofsOut& dofs_out)
{
   constexpr bool has_value_channel = need_test_values_v<Integrand>;
   constexpr bool has_gradient_channel = need_test_grads_v<Integrand>;

   static_assert(has_value_channel || has_gradient_channel,
      "At least one channel (Value or Gradient) must be present.");

   if constexpr (has_value_channel && has_gradient_channel)
   {
      ApplyValuesAndGradientTestFunctions<true>(
         kernel_context,
         side,
         test_qd,
         Du.values,
         Du.gradients,
         dofs_out);
   }
   else if constexpr (has_value_channel)
   {
      ApplyAddTestFunctions(
         kernel_context,
         side,
         test_qd,
         Du.values,
         dofs_out);
   }
   else if constexpr (has_gradient_channel)
   {
      ApplyGradientTestFunctions<true>(
         kernel_context,
         side,
         test_qd,
         Du.gradients,
         dofs_out);
   }
}

template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename FaceContext,
   typename Integrand,
   typename QuadraturePointData,
   typename DofsOut >
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const FaceContext& face_info,
   const Integrand& integrand,
   const QuadraturePointData& Du,
   DofsOut& dofs_out)
{
   // That would reintroduce requirements<Integrand>::test_mask limitation
   constexpr auto TestName = requirements<Integrand>::test_name;

   // Get test quad data from operator context (TestName already extracted above)
   const auto& test_qd_wrapper =
      operator_context.template finite_element_facet_quad_data<TestName>();

   ApplyAddFacetTestFunctions(
      kernel_context,
      face_info.MinusSide(),
      test_qd_wrapper.MinusSide(),
      integrand,
      Du,
      dofs_out);
}

} // namespace gendil

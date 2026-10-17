// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/executioncontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/weakformrequirements.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/elementcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/writechannelcontributions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatefields.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionschannels.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"

namespace gendil {

template <
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename ElementContext,
   typename Integrand >
   requires CellIntegrand<Integrand>
GENDIL_HOST_DEVICE
void GenericCellIntegrandOperator(
   KernelContext & kernel_context,
   ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const ElementContext & element_context,
   const Integrand & integrand )
{
   auto fields = InterpolateFields(
      kernel_context,
      execution_context,
      operator_context,
      element_context,
      integrand);

   auto contributions = MakeQuadraturePointContainerFromIntegrand(
      kernel_context,
      execution_context,
      operator_context,
      integrand);

   const auto pullback = MakeCellPullback(integrand);
   const auto& integration_rule = operator_context.GetIntegrationRule();

   QuadraturePointLoop(
      kernel_context,
      integration_rule,
      [&] (const auto& quad_index)
      {
         auto quad_pt_context = MakeQuadraturePointContext(
            kernel_context,
            execution_context,
            operator_context,
            element_context,
            integrand,
            quad_index);

         ComputeContributions(
            kernel_context,
            execution_context,
            operator_context,
            element_context,
            quad_pt_context,
            fields,
            pullback,
            quad_index,
            contributions);
      });

   ApplyAddTestFunctions(
      kernel_context,
      execution_context,
      operator_context,
      element_context,
      integrand,
      contributions);
}

} // namespace gendil

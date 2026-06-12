// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/elementcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/quadraturepointcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatefields.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctions.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"

namespace gendil {

template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename Integrand,
   typename ElementDofsIn,
   typename ElementDofsOut >
GENDIL_HOST_DEVICE
void GenericCellIntegrandOperator(
   KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const ElementContext & element_context,
   const Integrand & integrand,
   const ElementDofsIn & dofs_in,
   ElementDofsOut & dofs_out )
{
   if constexpr ( CellIntegrand<Integrand> )
   {
      auto fields = InterpolateFields(
         kernel_context,
         weak_form_context,
         operator_context,
         element_context,
         integrand,
         dofs_in);

      const auto& integration_rule = operator_context.integration_rule();

      auto Du = MakeQuadraturePointContainerFromIntegrand(
         kernel_context,
         weak_form_context,
         integrand,
         integration_rule);

      QuadraturePointLoop(
         kernel_context,
         integration_rule,
         [&] (const auto& quad_index)
         {
            auto quad_pt_context = MakeQuadraturePointContext(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               integrand,
               quad_index);

            ComputeContributions(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               quad_pt_context,
               fields,
               integrand,
               quad_index,
               Du);
         });

      ApplyAddTestFunctions(
         kernel_context,
         operator_context,
         integrand,
         Du,
         dofs_out);
   }
}

template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename Map,
   typename ElementDofsIn,
   typename ElementDofsOut >
GENDIL_HOST_DEVICE
void GenericCellIntegrandOperator(
   KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const ElementContext & element_context,
   const SumFormExpr<Map> & sum_integrand,
   const ElementDofsIn & dofs_in,
   ElementDofsOut & dofs_out )
{
   std::apply(
      [&] (auto const & ... entries)
      {
         (
            GenericCellIntegrandOperator(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               entries.value,
               dofs_in,
               dofs_out
            ),
            ...
         );
      },
      sum_integrand.map.entries
   );
}

} // namespace gendil

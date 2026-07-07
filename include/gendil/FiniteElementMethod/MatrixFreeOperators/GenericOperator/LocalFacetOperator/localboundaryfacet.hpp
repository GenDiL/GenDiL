// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/elementcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/localoperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/faceloop.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/writechannelcontributions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionschannels.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatefields.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"

namespace gendil {

// Evaluates a local/cell-owned boundary facet contribution for one adjacent
// Cell. There is no plus side; face_info.MinusSide() is the active/current
// Cell side and only that row contribution is accumulated.
template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename FaceInfo,
   typename Integrand,
   typename ElementDofsIn,
   typename ElementDofsOut>
GENDIL_HOST_DEVICE
void LocalBoundaryFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const FaceInfo& face_info,
   const Integrand& integrand,
   const ElementDofsIn& dofs_in,
   ElementDofsOut& dofs_out)
{
   if constexpr (BoundaryFacetIntegrand<Integrand>)
   {
      auto fields = InterpolateFields(
         kernel_context,
         weak_form_context,
         operator_context,
         face_info,
         integrand,
         dofs_in);

      const auto minus_side = face_info.MinusSide();
      auto face_integration_rule =
         GetFaceIntegrationRule(
            minus_side,
            operator_context.facet_integration_rules());

      auto Du = MakeQuadraturePointContainerFromIntegrand(
         kernel_context,
         weak_form_context,
         integrand,
         face_integration_rule);

      QuadraturePointLoop(
         kernel_context,
         face_integration_rule,
         [&] (const auto& quad_index)
         {
            auto facet_quad_pt_context = MakeFacetQuadraturePointContext(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_info,
               integrand,
               quad_index);

            ComputeContributions(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               facet_quad_pt_context,
               fields,
               integrand,
               quad_index,
               Du);
         });

      ApplyAddTestFunctions(
         kernel_context,
         weak_form_context,
         operator_context,
         face_info,
         integrand,
         Du,
         dofs_out);
   }
}

template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename FaceInfo,
   typename Map,
   typename ElementDofsIn,
   typename ElementDofsOut>
GENDIL_HOST_DEVICE
void LocalBoundaryFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const FaceInfo& face_info,
   const SumFormExpr<Map>& sum_integrand,
   const ElementDofsIn& dofs_in,
   ElementDofsOut& dofs_out)
{
   std::apply(
      [&] (auto const & ... entries)
      {
         (
            LocalBoundaryFacetIntegrandOperator(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_info,
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

template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename TrialSpace,
   typename Integrand,
   typename ElementDofsIn,
   typename ElementDofsOut>
GENDIL_HOST_DEVICE
void LocalBoundaryFacetOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const TrialSpace& trial_space,
   const Integrand& integrand,
   const ElementDofsIn& dofs_in,
   ElementDofsOut& dofs_out)
{
   if constexpr (has_boundary_facet_contributions_v<Integrand>)
   {
      BoundaryFaceLoop(
         trial_space,
         element_context.element_index,
         [&] (auto const & face_info)
         {
            LocalBoundaryFacetIntegrandOperator(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_info,
               integrand,
               dofs_in,
               dofs_out);
         }
      );
   }
}

} // namespace gendil

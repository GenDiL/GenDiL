// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/elementcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/facetcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/localoperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/quadraturepointcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/faceloop.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/writechannelcontributions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionschannels.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatefields.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"

namespace gendil {

// Evaluates a local/cell-owned interior facet contribution for one row side.
// For this adapter, face_info.MinusSide() is the active/current Cell side,
// face_info.PlusSide() is the neighbor, and only the current row contribution
// is accumulated into minus_dofs_out. Global interior execution uses the
// canonical side-qualified channel path instead of this local/current-row
// semantics.
template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename GlobalFaceInfo,
   typename Integrand,
   typename MinusElementDofsIn,
   typename PlusElementDofsIn,
   typename MinusElementDofsOut>
GENDIL_HOST_DEVICE
void LocalInteriorFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const GlobalFaceInfo& face_info,
   const Integrand& integrand,
   const MinusElementDofsIn& minus_dofs_in,
   const PlusElementDofsIn& plus_dofs_in,
   MinusElementDofsOut& minus_dofs_out)
{
   if constexpr (InteriorFacetIntegrand<Integrand>)
   {
      auto fields = InterpolateFields(
         kernel_context,
         weak_form_context,
         operator_context,
         face_info,
         integrand,
         minus_dofs_in,
         plus_dofs_in);

      auto face_context =
         MakeLocalInteriorFacetContext(
            weak_form_context,
            integrand,
            face_info);

      const auto minus_side = face_info.MinusSide();
      auto face_integration_rule =
         GetFaceIntegrationRule(minus_side, operator_context.facet_integration_rules());

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
            auto facet_quad_pt_context =
               MakeLocalInteriorFacetQuadraturePointContext(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_context,
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
         }
      );

      ApplyAddTestFunctions(
         kernel_context,
         weak_form_context,
         operator_context,
         face_info,
         integrand,
         Du,
         minus_dofs_out);
   }
}

template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename GlobalFaceInfo,
   typename Map,
   typename MinusElementDofsIn,
   typename PlusElementDofsIn,
   typename MinusElementDofsOut>
GENDIL_HOST_DEVICE
void LocalInteriorFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const GlobalFaceInfo& face_info,
   const SumFormExpr<Map>& sum_integrand,
   const MinusElementDofsIn& minus_dofs_in,
   const PlusElementDofsIn& plus_dofs_in,
   MinusElementDofsOut& minus_dofs_out)
{
   std::apply(
      [&] (auto const & ... entries)
      {
         (
            LocalInteriorFacetIntegrandOperator(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_info,
               entries.value,
               minus_dofs_in,
               plus_dofs_in,
               minus_dofs_out
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
   typename DofsInView,
   typename MinusElementDofsIn,
   typename MinusElementDofsOut>
GENDIL_HOST_DEVICE
void LocalInteriorFacetOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const TrialSpace& trial_space,
   const Integrand& integrand,
   const DofsInView& dofs_in,
   const MinusElementDofsIn& minus_dofs_in,
   MinusElementDofsOut& minus_dofs_out)
{
   if constexpr (has_interior_facet_contributions_v<Integrand>)
   {
      InteriorFaceLoop(
         trial_space,
         element_context.element_index,
         [&] (auto const & face_info)
         {
            auto plus_dofs_in = ReadDofs(
               kernel_context,
               trial_space,
               face_info.PlusSide(),
               dofs_in);

            LocalInteriorFacetIntegrandOperator(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_info,
               integrand,
               minus_dofs_in,
               plus_dofs_in,
               minus_dofs_out);
         }
      );
   }
}

} // namespace gendil

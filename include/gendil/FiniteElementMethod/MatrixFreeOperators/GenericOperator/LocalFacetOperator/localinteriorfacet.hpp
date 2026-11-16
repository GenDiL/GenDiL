// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/elementcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/facetcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/localoperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/faceloop.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/writechannelcontributions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionschannels.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatefields.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/weakformtraits.hpp"

namespace gendil {

// Local traversal owns the current row; neighbor reads may use a separate
// parent context when the current row has been cached or seeded by assembly.
template<class Kernel, class CurrentContext, class NeighborContext, class OperatorContext,
         class Entity, class Face, class Integrand>
   requires InteriorFacetIntegrand<Integrand>
GENDIL_HOST_DEVICE
void LocalInteriorFacetIntegrandOperator(Kernel& kernel, CurrentContext& execution_context,
   const NeighborContext& neighbor_execution_context, const OperatorContext& operator_context,
   const Entity& entity, const Face& face, const Integrand& integrand)
{
   WithLocalFacetNormalOrientation(face, [&](const auto& static_face) {
      constexpr int Sign = static_face_normal_orientation_v<decltype(static_face)>;
      const auto pullback = MakeLocalInteriorFacetPullback<Sign>(integrand);
      using Pullback = std::remove_cvref_t<decltype(pullback)>;
      if constexpr (Pullback::template Contains<ValueChannel>() ||
                    Pullback::template Contains<GradientChannel>())
      {
         auto local_fields = InterpolateLocalFacetFields(kernel, execution_context,
            neighbor_execution_context, operator_context, static_face, integrand);
         const auto fields = [&] {
            if constexpr (Sign == 1)
               return FacetTraceFields<const decltype(local_fields.current_fields)&,
                                       const decltype(local_fields.neighbor_fields)&>{
                  local_fields.current_fields, local_fields.neighbor_fields};
            else
               return FacetTraceFields<const decltype(local_fields.neighbor_fields)&,
                                       const decltype(local_fields.current_fields)&>{
                  local_fields.neighbor_fields, local_fields.current_fields};
         }();
         auto face_context = MakeLocalInteriorFacetContext(execution_context, integrand, static_face);
         const auto& current_face = static_face.CurrentSide();
         auto rule = GetFaceIntegrationRule(current_face, operator_context.GetFacetIntegrationRules());
         const auto& output_space = GetOutputSpace(integrand, execution_context);
         auto contributions = MakeQuadraturePointContainerForSpace(kernel, rule, output_space, pullback);
         QuadraturePointLoop(kernel, rule, [&](const auto& qi)
         {
            auto qcontext = MakeLocalInteriorFacetQuadraturePointContext(kernel,
               execution_context, operator_context, entity, face_context, integrand, qi);
            ComputeContributions(kernel, execution_context, operator_context, face_context,
               qcontext, fields, pullback, qi, contributions);
         });
         ApplyAddTestFunctions(kernel, execution_context, operator_context, entity, static_face, integrand, contributions);
      }
   });
}

template<class Kernel, class Context, class OperatorContext, class Entity, class Face, class Integrand>
   requires InteriorFacetIntegrand<Integrand>
GENDIL_HOST_DEVICE
void LocalInteriorFacetIntegrandOperator(Kernel& kernel, Context& execution_context,
   const OperatorContext& operator_context, const Entity& entity, const Face& face, const Integrand& integrand)
{
   LocalInteriorFacetIntegrandOperator(kernel, execution_context, execution_context,
      operator_context, entity, face, integrand);
}


} // namespace gendil

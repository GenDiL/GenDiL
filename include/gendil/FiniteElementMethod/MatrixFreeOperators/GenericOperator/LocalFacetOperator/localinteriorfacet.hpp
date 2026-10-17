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
   auto fields = InterpolateFields(kernel, execution_context, neighbor_execution_context,
      operator_context, face, integrand);
   auto face_context = MakeLocalInteriorFacetContext(execution_context, integrand, face);
   auto rule = GetFaceIntegrationRule(face.MinusSide(), operator_context.GetFacetIntegrationRules());
   const auto& output_space = GetOutputSpace(integrand, execution_context);
   const auto pullback = MakeLocalInteriorFacetPullback(integrand);
   auto contributions = MakeQuadraturePointContainerForSpace(kernel, rule, output_space, pullback);
   QuadraturePointLoop(kernel, rule, [&](const auto& qi)
   {
      auto qcontext = MakeLocalInteriorFacetQuadraturePointContext(kernel,
         execution_context, operator_context, entity, face_context, integrand, qi);
      ComputeContributions(kernel, execution_context, operator_context, entity,
         qcontext, fields, pullback, qi, contributions);
   });
   ApplyAddTestFunctions(kernel, execution_context, operator_context, entity, face, integrand, contributions);
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

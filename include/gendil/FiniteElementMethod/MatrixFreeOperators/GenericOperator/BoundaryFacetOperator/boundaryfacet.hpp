// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/executioncontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/weakformrequirements.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/elementcontext.hpp"
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

template<class Kernel, class Context, class OperatorContext, class Entity, class Face, class Integrand>
   requires BoundaryFacetIntegrand<Integrand>
GENDIL_HOST_DEVICE
void BoundaryFacetIntegrandOperator(Kernel& kernel, Context& context,
   const OperatorContext& operator_context, const Entity& entity, const Face& face, const Integrand& integrand)
{
   auto fields = InterpolateFields(kernel, context, operator_context, entity, face, integrand);
   auto rule = GetFaceIntegrationRule(face.MinusSide(), operator_context.GetFacetIntegrationRules());
   const auto& output_space = GetOutputSpace(integrand, context);
   const auto pullback = MakeBoundaryFacetPullback(integrand);
   auto contributions = MakeQuadraturePointContainerForSpace(kernel, rule, output_space, pullback);
   QuadraturePointLoop(kernel, rule, [&] (const auto& qi)
   {
      auto qcontext = MakeFacetQuadraturePointContext(kernel, context, operator_context, entity, face, integrand, qi);
      ComputeContributions(kernel, context, operator_context, entity,
         qcontext, fields, pullback, qi, contributions);
   });
   ApplyAddTestFunctions(kernel, context, operator_context, entity, face, integrand, contributions);
}

} // namespace gendil

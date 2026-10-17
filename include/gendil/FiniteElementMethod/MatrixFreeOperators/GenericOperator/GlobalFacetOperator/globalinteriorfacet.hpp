// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/elementcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globaloperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/restrictedweakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/writechannelcontributions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionschannels.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"

namespace gendil {

template <
   typename KernelContext,
   typename MinusExecutionContext,
   typename PlusExecutionContext,
   typename OperatorContext,
   typename FaceDomain,
   typename FaceInfo,
   typename Integrand >
   requires InteriorFacetIntegrand<Integrand>
GENDIL_HOST_DEVICE
void GlobalInteriorFacetIntegrandOperator(
   KernelContext & kernel_context,
   MinusExecutionContext & minus_execution_context,
   PlusExecutionContext & plus_execution_context,
   const OperatorContext & operator_context,
   const FaceDomain & face_domain,
   const FaceInfo & face_info,
   const Integrand & integrand )
{
   auto fields = InterpolateFields(
      kernel_context,
      minus_execution_context,
      plus_execution_context,
      operator_context,
      face_info,
      integrand);

   auto pullback = MakeGlobalInteriorFacetPullback(integrand);

   auto integration_rule = GetFaceIntegrationRule(
      face_info.MinusSide(),
      operator_context.GetFacetIntegrationRules());

   const auto& output_space = GetOutputSpace(integrand, minus_execution_context);

   auto minus_contributions = MakeQuadraturePointContainerForSpace<SemanticSide::Minus>(
      kernel_context,
      integration_rule,
      output_space,
      pullback);

   auto plus_contributions = MakeQuadraturePointContainerForSpace<SemanticSide::Plus>(
      kernel_context,
      integration_rule,
      output_space,
      pullback);

   ElementContext element_context{
      face_info.MinusSide().GetCellIndex(),
      face_domain.GetMinusCellMesh().GetCell(face_info.MinusSide().GetCellIndex())};

   auto face_context = MakeGlobalInteriorFacetContext(
      face_domain,
      integrand,
      pullback,
      face_info);

   QuadraturePointLoop(
      kernel_context,
      integration_rule,
      [&] (const auto& quad_index)
      {
         auto quad_pt_context = MakeGlobalInteriorFacetQuadraturePointContext(
            kernel_context,
            minus_execution_context,
            operator_context,
            element_context,
            face_context,
            integrand,
            pullback,
            quad_index);

         ComputeContributions(
            kernel_context,
            minus_execution_context,
            operator_context,
            element_context,
            quad_pt_context,
            fields,
            pullback,
            quad_index,
            minus_contributions,
            plus_contributions);
      });

   ApplyAddTestFunctions<SemanticSide::Minus>(
      kernel_context,
      minus_execution_context,
      operator_context,
      face_info,
      integrand,
      minus_contributions);

   ApplyAddTestFunctions<SemanticSide::Plus>(
      kernel_context,
      plus_execution_context,
      operator_context,
      face_info,
      integrand,
      plus_contributions);
}

template <
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename FaceDomain,
   typename FaceInfo,
   typename Integrand >
   requires InteriorFacetIntegrand<Integrand>
GENDIL_HOST_DEVICE
void GlobalInteriorFacetIntegrandOperator(
   KernelContext & kernel_context,
   ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const FaceDomain & face_domain,
   const FaceInfo & face_info,
   const Integrand & integrand )
{
   GlobalInteriorFacetIntegrandOperator(
      kernel_context,
      execution_context,
      execution_context,
      operator_context,
      face_domain,
      face_info,
      integrand);
}

} // namespace gendil

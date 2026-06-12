// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/elementcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/facetcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/quadraturepointcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatefields.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"
#include "gendil/FiniteElementMethod/WeakForm/quadraturemeasure.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/writechannelcontributions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applyvaluesandgradienttestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionschannels.hpp"

#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/cellintegrand.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/localinteriorfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/localboundaryfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/localfacetsoperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/globalfacetsoperator.hpp"

namespace gendil {

// ============================================================================
// Pullback GenericExplicitOperator
// ============================================================================

/**
 * @brief Pullback-based explicit operator strategy dispatcher.
 *
 */
template<
  class KernelPolicy,
  class WeakForm,
  class WeakFormContext,
  class OperatorContext,
  class DofsInView,
  class DofsOutView>
void GenericExplicitOperator(
  const WeakForm& weak_form,
  const WeakFormContext& wf_ctx,
  const OperatorContext& op_ctx,
  const DofsInView& dofs_in,
  DofsOutView& dofs_out)
{
   if constexpr (use_global_facets_operator_v<WeakFormContext>)
   {
      GlobalFacetsGenericExplicitOperator<KernelPolicy>(
         weak_form,
         wf_ctx,
         op_ctx,
         dofs_in,
         dofs_out);
   }
   else
   {
      LocalFacetsGenericExplicitOperator<KernelPolicy>(
         weak_form,
         wf_ctx,
         op_ctx,
         dofs_in,
         dofs_out);
   }
}

// ============================================================================
// Pullback GenericOperator class
// ============================================================================

/**
 * @brief Pullback-based generic operator.
 *
 * High-level operator wrapper that uses pullback channels for cell integrands.
 */
template<
   class KernelPolicy,
   class IntegrationRule,
   class WeakForm,
   class WeakFormContext>
class PullbackGenericOperator
{
   WeakForm weak_form;
   WeakFormContext wf_ctx;
   IntegrationRule ir;

   using OperatorContext = decltype(MakeOperatorContext(wf_ctx, ir));
   OperatorContext op_ctx;

public:
   PullbackGenericOperator(WeakForm wf,
                           WeakFormContext ctx,
                           IntegrationRule ir_)
      : weak_form(std::move(wf))
      , wf_ctx(std::move(ctx))
      , ir(ir_)
      , op_ctx(MakeOperatorContext(wf_ctx, ir))
   {}

   template<class input, class output>
   void operator()(const input& dofs_vector_in, output& dofs_vector_out) const
   {
      constexpr auto TrialName = requirements<WeakForm>::trial_name;
      constexpr auto TestName  = requirements<WeakForm>::test_name;
      const auto & trial_space = wf_ctx.template fe_field<TrialName>().space;
      const auto & test_space = wf_ctx.template fe_field<TestName>().space;
      using TestSpace = std::remove_cvref_t< decltype(test_space) >;
      using TestRestriction = typename TestSpace::restriction_type;

      auto dofs_in = MakeReadOnlyElementTensorView< KernelPolicy >( trial_space, dofs_vector_in );

      if constexpr ( std::is_same_v< TestRestriction, L2Restriction > )
      {
         auto dofs_out = MakeWriteOnlyElementTensorView< KernelPolicy >( test_space, dofs_vector_out );
         GenericExplicitOperator<KernelPolicy>(
            weak_form, wf_ctx, op_ctx, dofs_in, dofs_out
         );
      }
      else if constexpr ( is_h1_restriction_v< TestRestriction > )
      {
         dofs_vector_out = 0.0;
         auto dofs_out = MakeReadWriteElementTensorView< KernelPolicy >( test_space, dofs_vector_out );
         GenericExplicitOperator<KernelPolicy>(
            weak_form, wf_ctx, op_ctx, dofs_in, dofs_out
         );
      }
      else
      {
         static_assert(
            is_h1_restriction_v< TestRestriction >,
            "PullbackGenericOperator supports only L2Restriction, H1Restriction, and VectorH1Restriction outputs." );
      }
   }
};

/**
 * @brief Factory for pullback-based GenericOperator
 *
 * **Usage:**
 * ```cpp
 * auto pb_op = MakePullbackGenericOperator<KernelPolicy>(
 *    weak_form, wf_ctx, integration_rule);
 * pb_op(input_vector, output_vector);
 * ```
 */
template<
  class KernelPolicy,
  class IntegrationRule,
  class WeakFormTuple,
  class WeakFormContext>
auto MakePullbackGenericOperator(
   WeakFormTuple weak_form,
   WeakFormContext wf_ctx,
   IntegrationRule ir)
{
   return PullbackGenericOperator<KernelPolicy, IntegrationRule, WeakFormTuple, WeakFormContext>(
      std::move(weak_form),
      std::move(wf_ctx),
      ir
   );
}

template<
  class KernelPolicy,
  class IntegrationRule,
  class WeakFormTuple,
  class WeakFormContext>
auto MakeGenericOperator(
   WeakFormTuple weak_form,
   WeakFormContext wf_ctx,
   IntegrationRule ir)
{
   return PullbackGenericOperator<KernelPolicy, IntegrationRule, WeakFormTuple, WeakFormContext>(
      std::move(weak_form),
      std::move(wf_ctx),
      ir
   );
}

template<
  class KernelPolicy,
  class IntegrationRule,
  class WeakFormTuple,
  class WeakFormContext>
auto MakeWeakFormResidualOperator(
   WeakFormTuple weak_form,
   WeakFormContext wf_ctx,
   IntegrationRule ir)
{
   return PullbackGenericOperator<KernelPolicy, IntegrationRule, WeakFormTuple, WeakFormContext>(
      std::move(weak_form),
      std::move(wf_ctx),
      ir
   );
}

} // namespace gendil

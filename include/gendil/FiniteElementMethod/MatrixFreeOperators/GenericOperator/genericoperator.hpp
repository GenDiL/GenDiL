// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/LocalFacetOperator/localfacetsoperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/GlobalFacetOperator/globalfacetsoperator.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"

namespace gendil {

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericExplicitOperator(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   using I = std::remove_cvref_t<WeakForm>;

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TestName  = requirements<I>::test_name;

   static_assert(
      TrialName != StaticString{"Error"},
      "GenericExplicitOperator: missing TrialSpace in weak_form.");
   static_assert(
      TestName != StaticString{"Error"},
      "GenericExplicitOperator: missing TestSpace in weak_form.");

   if constexpr (use_global_facets_operator_v<WeakFormContext>)
   {
      GenericGlobalDomainOperator<TrialName, TestName, KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         dofs_vector_in,
         dofs_vector_out);
   }
   else
   {
      GenericLocalDomainOperator<TrialName, TestName, KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         dofs_vector_in,
         dofs_vector_out);
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

public:
   PullbackGenericOperator(WeakForm wf,
                           WeakFormContext ctx,
                           IntegrationRule ir_)
      : weak_form(std::move(wf))
      , wf_ctx(std::move(ctx))
      , ir(ir_)
   {}

   template<class input, class output>
   void operator()(const input& dofs_vector_in, output& dofs_vector_out) const
   {
      // Prepare the output exactly once. Recursive term and domain-batch
      // dispatch contributes additively into this vector.
      dofs_vector_out = 0.0;

      GenericExplicitOperator<KernelPolicy>(
         weak_form,
         wf_ctx,
         ir,
         dofs_vector_in,
         dofs_vector_out);
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

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/FiniteElementMethod/OperatorValidation/localfacetquadraturevalidation.hpp"

namespace gendil
{

template<class KernelPolicy, class Rule, class Form, class Context>
class PullbackGenericOperator;

/**
 * Select optional operator checks at compile time. Derive a policy from this
 * type and override the flags to enable checks; the default enables none.
 */
struct OperatorChecks
{
   static constexpr bool local_facet_quadrature = false;
};

/**
 * Check a form before direct evaluation or sparse assembly. For example:
 *
 *     struct Checks : OperatorChecks
 *     {
 *        static constexpr bool local_facet_quadrature = true;
 *     };
 *     ValidateOperator(form, context, rule, Checks{});
 *
 * Checks run on the host, including for GPU execution policies. Connectivity
 * must be host-readable when a check needs to inspect it. Failure aborts with
 * a diagnostic in both Debug and Release builds; success returns nothing.
 * Disabled checks are not instantiated. No validation state is stored: repeat
 * validation after changing the topology, connectivity, or quadrature.
 */
template<class Form, class Context, class Rule, class Checks>
void ValidateOperator(const Form& form, const Context& context,
                      const Rule& rule, const Checks&)
{
   if constexpr (Checks::local_facet_quadrature)
      local_facet_quadrature_detail::ValidateLocalFacetQuadrature(form, context, rule);
}

/** Run selected checks on an operator made by any generic operator factory. */
template<class KernelPolicy, class Rule, class Form, class Context, class Checks>
void ValidateOperator(
   const PullbackGenericOperator<KernelPolicy, Rule, Form, Context>& op,
   const Checks& checks)
{
   ValidateOperator(op.weak_form, op.wf_ctx, op.ir, checks);
}

} // namespace gendil

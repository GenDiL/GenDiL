// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/globalfacetassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/localfacetassembly.hpp"

namespace gendil {

// Format factories have already checked admission before allocating a target.
// Direct target entry points perform the same checks before reaching this body.
template<class KernelPolicy, SparseAssemblyMode Mode = SparseAssemblyMode::Full,
         class Form, class Context, class Rule, class Target>
void AssembleSparseTarget(const Form& form, const Context& context, const Rule& rule, Target& target)
{
   if constexpr (weak_form_uses_partition_integration_domain_v<Form, Context>)
      details::AssemblePartitionSparseTarget<KernelPolicy, Mode>(form, context, rule, target);
   else
      details::AssembleLocalSparseTarget<KernelPolicy, Mode>(form, context, rule, target);
}

template<class KernelPolicy, class WeakForm, class WeakFormContext, class IntegrationRule, class SparseMatrixType>
requires std::is_copy_constructible_v<SparseMatrixType>
void GenericAssembly(const WeakForm& form, const WeakFormContext& context,
   const IntegrationRule& rule, SparseMatrixType& target)
{
   if constexpr (weak_form_uses_partition_integration_domain_v<WeakForm, WeakFormContext>)
      details::ValidateSparseAssemblyExecutionInputs<KernelPolicy>(form, context);
   else
      details::ValidateSparseAssemblyExecutionInputs<KernelPolicy>(form, context, target);
   AssembleSparseTarget<KernelPolicy>(form, context, rule, target);
}
} // namespace gendil

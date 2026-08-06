// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/sparseassemblyvalidation.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrationdomain.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class SparseMatrixType>
void GenericGlobalFacetAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   SparseMatrixType& sparse_matrix)
{
   (void)weak_form;
   (void)wf_ctx;
   (void)integration_rule;
   (void)sparse_matrix;
   static_assert(
      dependent_false_v<KernelPolicy, WeakForm>,
      "GenericGlobalFacetAssembly: PartitionIntegrationDomain is unsupported "
      "by sparse assembly; use a MeshIntegrationDomain or matrix-free "
      "GenericOperator.");
}

} // namespace gendil

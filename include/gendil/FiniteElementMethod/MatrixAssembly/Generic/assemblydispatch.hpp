// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/globalfacetassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/localfacetassembly.hpp"

#include <type_traits>

namespace gendil {

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class SparseMatrixType>
requires std::is_copy_constructible_v< SparseMatrixType >
void GenericAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   SparseMatrixType& sparse_matrix)
{
   if constexpr (
      weak_form_uses_partition_integration_domain_v<
         WeakForm,
         WeakFormContext>)
   {
      GenericGlobalFacetAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         sparse_matrix);
   }
   else
   {
      GenericLocalFacetAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         sparse_matrix);
   }
}

} // namespace gendil

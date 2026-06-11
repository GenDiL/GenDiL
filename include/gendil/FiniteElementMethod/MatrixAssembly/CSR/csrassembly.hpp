// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/csrmatrix.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcooassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/CSR/csrfinalization.hpp"

namespace gendil {

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericCSRAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   Backend backend )
{
   auto raw_coo =
      GenericRawCOOAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule );
   auto csr =
      FinalizeRawCOOToCSR(
         raw_coo,
         HostSortReduceRawCOOToCSRPolicy{},
         backend );
   FreeRawCOOTripletBuffer( raw_coo );
   return csr;
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericCSRAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule )
{
   return GenericCSRAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      DefaultCSRBackend{} );
}

} // namespace gendil

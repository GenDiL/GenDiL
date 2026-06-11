// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_HYPRE

#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcooassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/HypreCSR/hyprecsrfinalization.hpp"

namespace gendil
{

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericHypreCSRAssembly(
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
   auto hypre_csr =
      FinalizeRawCOOToHypreCSR(
         raw_coo,
         HostSortReduceRawCOOToHypreCSRPolicy{},
         backend );
   FreeRawCOOTripletBuffer( raw_coo );
   return hypre_csr;
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericHypreCSRAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule )
{
   return GenericHypreCSRAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      typename HypreCSRMatrix<>::backend_type{} );
}

} // namespace gendil

#endif // GENDIL_USE_HYPRE

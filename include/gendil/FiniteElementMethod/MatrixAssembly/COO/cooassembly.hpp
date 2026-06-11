// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Canonical COO storage and RawCOO host sort/reduce finalization.
#include "gendil/Algebra/SparseMatrixTypes/coomatrix.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/coofinalization.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/localinsertion.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcooassembly.hpp"

namespace gendil {

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericCOOAssembly(
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
   auto coo =
      FinalizeRawCOOToCOO(
         raw_coo,
         HostSortReduceRawCOOPolicy{},
         backend );
   FreeRawCOOTripletBuffer( raw_coo );
   return coo;
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericCOOAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule )
{
   return GenericCOOAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      DefaultCOOBackend{} );
}

} // namespace gendil

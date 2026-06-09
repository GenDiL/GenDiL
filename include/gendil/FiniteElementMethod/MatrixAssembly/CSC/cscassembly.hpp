// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/cscmatrix.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcooassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/CSC/cscfinalization.hpp"

namespace gendil {

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericCSCAssembly(
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
   auto csc =
      FinalizeRawCOOToCSC(
         raw_coo,
         HostSortReduceRawCOOToCSCPolicy{},
         backend );
   FreeRawCOOTripletBuffer( raw_coo );
   return csc;
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericCSCAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule )
{
   return GenericCSCAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      DefaultCSCBackend{} );
}

} // namespace gendil

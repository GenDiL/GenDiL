// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrpattern.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/localinsertion.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/weakformtraversal.hpp"
#include "gendil/Utilities/KernelContext/kernelcontexttraits.hpp"

#include <type_traits>

namespace gendil {

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class SparseMatrixType >
void GenericBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   SparseMatrixType & sparse_matrix)
{
   GENDIL_REQUIRE_UNBATCHED_OPERATOR( KernelPolicy );

   auto op_ctx = MakeOperatorContext(wf_ctx, integration_rule);

   using I = std::remove_cvref_t<WeakForm>;
   ValidateSparseLinearAssemblyCoefficientInputs<I>();

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TestName  = requirements<I>::test_name;

   static_assert(TrialName != StaticString{"Error"}, "GenericExplicitOperator: missing TrialSpace in integrand.");
   static_assert(TestName  != StaticString{"Error"}, "GenericExplicitOperator: missing TestSpace in integrand.");

   // FE spaces come from wf_ctx via MakeTrialField/MakeTestField
   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;

   // Shared memory requirement: for now, bind to the integration rule used by this operator
   constexpr size_t required_shared_mem = required_shared_memory_v<KernelPolicy, IntegrationRule>;

   mesh::CellIterator<KernelPolicy>(
      trial_space,
      [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
      {
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];
         KernelContext<KernelPolicy, required_shared_mem> kernel_ctx(_shared_mem);

         AssembleElementSparseMatrix(
            kernel_ctx,
            wf_ctx,
            op_ctx,
            element_index,
            weak_form,
            sparse_matrix
         );
      }
   );
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   constexpr auto TrialName = requirements<WeakForm>::trial_name;
   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   auto bsr_matrix = MakeBlockDiagonalDGBSRPattern( trial_space );

   GenericBlockDiagonalAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      bsr_matrix
   );

   return bsr_matrix;
}

template < typename KernelPolicy, typename BSRMatrixType >
void SyncAssembledBSRValues(
   BSRMatrixType & bsr_matrix )
{
#if defined(GENDIL_USE_DEVICE)
   const GlobalIndex value_count =
      static_cast< GlobalIndex >( bsr_matrix.num_blocks ) *
      static_cast< GlobalIndex >( bsr_matrix.block_rows ) *
      static_cast< GlobalIndex >( bsr_matrix.block_cols );

   if constexpr ( is_host_configuration_v< KernelPolicy > )
   {
      ToDevice( value_count, bsr_matrix.values );
   }
   else
   {
      GENDIL_DEVICE_SYNC;
      ToHost( value_count, bsr_matrix.values );
   }
#else
   (void) bsr_matrix;
#endif
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericBSRAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   Backend backend)
{
   constexpr auto TrialName = requirements<WeakForm>::trial_name;
   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   auto bsr_matrix = MakeDGBSRPattern( trial_space, backend );

   GenericAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      bsr_matrix
   );

   SyncAssembledBSRValues< KernelPolicy >( bsr_matrix );

   return bsr_matrix;
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericBSRAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   return GenericBSRAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      DefaultBSRBackend{} );
}

} // namespace gendil

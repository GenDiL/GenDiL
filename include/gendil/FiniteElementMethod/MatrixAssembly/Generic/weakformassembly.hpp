// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/cooassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/CSC/cscassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/CSR/csrassembly.hpp"
#ifdef GENDIL_USE_HYPRE
#include "gendil/FiniteElementMethod/MatrixAssembly/HypreCSR/hyprecsrassembly.hpp"
#endif
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/assemblydispatch.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/defaultbackend.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/SGBSR/sgbsrassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/matrixassemblytype.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

namespace gendil {

template<
   MatrixAssemblyType Type,
   class WeakForm,
   class WeakFormContext >
auto MakeDefaultBackendFor(
   const WeakForm & weak_form,
   const WeakFormContext & wf_ctx )
{
   if constexpr (
      Type == MatrixAssemblyType::BSR ||
      Type == MatrixAssemblyType::SGBSR )
   {
      return MakeDefaultBSRBackend( weak_form, wf_ctx );
   }
   else
   {
      (void) weak_form;
      (void) wf_ctx;
      return DefaultBackendFor_t< Type >{};
   }
}

template<
   MatrixAssemblyType Type,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   Backend backend)
{
   if constexpr ( Type == MatrixAssemblyType::BSR )
   {
      return GenericBSRAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         std::move( backend ) );
   }
   else if constexpr ( Type == MatrixAssemblyType::SGBSR )
   {
      return GenericSGBSRAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         std::move( backend ) );
   }
   else if constexpr ( Type == MatrixAssemblyType::RawCOO )
   {
      (void) backend;
      return GenericRawCOOAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule );
   }
   else if constexpr ( Type == MatrixAssemblyType::COO )
   {
      return GenericCOOAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         std::move( backend ) );
   }
   else if constexpr ( Type == MatrixAssemblyType::CSR )
   {
      return GenericCSRAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         std::move( backend ) );
   }
#ifdef GENDIL_USE_HYPRE
   else if constexpr ( Type == MatrixAssemblyType::HypreCSR )
   {
      return GenericHypreCSRAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         std::move( backend ) );
   }
#endif
   else if constexpr ( Type == MatrixAssemblyType::CSC )
   {
      return GenericCSCAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         std::move( backend ) );
   }
   else
   {
      static_assert(
         dependent_false_value_v< Type >,
         "GenericAssembly: requested matrix assembly type is not implemented yet." );
   }
}

template<
   MatrixAssemblyType Type,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   return GenericAssembly<Type, KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      MakeDefaultBackendFor< Type >( weak_form, wf_ctx ) );
}

template<
   MatrixAssemblyType Type,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericElementBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   Backend backend)
{
   if constexpr (Type == MatrixAssemblyType::BSR)
   {
      return GenericBSRElementBlockDiagonalAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         std::move( backend ));
   }
   else if constexpr (Type == MatrixAssemblyType::SGBSR)
   {
      return GenericSGBSRElementBlockDiagonalAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         std::move( backend ));
   }
   else
   {
      auto raw_coo =
         GenericRawCOOElementBlockDiagonalAssembly<KernelPolicy>(
            weak_form,
            wf_ctx,
            integration_rule);

      if constexpr (Type == MatrixAssemblyType::RawCOO)
      {
         (void)backend;
         return raw_coo;
      }
      else if constexpr (Type == MatrixAssemblyType::COO)
      {
         auto matrix =
            FinalizeRawCOOToCOO< KernelPolicy >(
               raw_coo,
               std::move( backend ));
         return matrix;
      }
      else if constexpr (Type == MatrixAssemblyType::CSR)
      {
         auto matrix =
            FinalizeRawCOOToCSR< KernelPolicy >(
               raw_coo,
               std::move( backend ));
         return matrix;
      }
#ifdef GENDIL_USE_HYPRE
      else if constexpr (Type == MatrixAssemblyType::HypreCSR)
      {
         auto matrix =
            FinalizeRawCOOToHypreCSR< KernelPolicy >(
               raw_coo,
               std::move( backend ));
         return matrix;
      }
#endif
      else if constexpr (Type == MatrixAssemblyType::CSC)
      {
         auto matrix =
            FinalizeRawCOOToCSC< KernelPolicy >(
               raw_coo,
               std::move( backend ));
         return matrix;
      }
      else
      {
         static_assert(
            dependent_false_value_v<Type>,
            "GenericElementBlockDiagonalAssembly: requested matrix "
            "assembly type is not implemented yet.");
      }
   }
}

template<
   MatrixAssemblyType Type,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericElementBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   return GenericElementBlockDiagonalAssembly<Type, KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      MakeDefaultBackendFor< Type >( weak_form, wf_ctx ) );
}

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <iostream>
#include <type_traits>

using namespace gendil;

namespace
{

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

#if defined(GENDIL_USE_DEVICE)
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::BSR >,
      NativeDeviceBSRBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::SGBSR >,
      NativeDeviceBSRBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::COO >,
      NativeDeviceCOOBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::CSR >,
      NativeDeviceCSRBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::CSC >,
      NativeDeviceCSCBackend<> > );
#else
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::BSR >,
      HostBSRBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::SGBSR >,
      HostBSRBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::COO >,
      HostCOOBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::CSR >,
      HostCSRBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::CSC >,
      HostCSCBackend<> > );
#endif
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::RawCOO >,
      Empty > );
#ifdef GENDIL_USE_HYPRE
#ifdef GENDIL_USE_HYPRE_DEVICE
using ExpectedHypreCSRBackend = HypreCSRDeviceBackend;
#else
using ExpectedHypreCSRBackend = HypreCSRHostBackend;
#endif
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::HypreCSR >,
      ExpectedHypreCSRBackend > );
#endif

bool TestTypedGenericAssemblyDefaults()
{
   Cartesian1DMesh mesh( 1.0, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   Cells< "mesh" > domain;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto weak_form = integrate( domain, u * v );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto csc =
      GenericAssembly< MatrixAssemblyType::CSC, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   using BSRType = std::remove_cvref_t< decltype( bsr ) >;
   using COOType = std::remove_cvref_t< decltype( coo ) >;
   using RawCOOType = std::remove_cvref_t< decltype( raw_coo ) >;
   using CSRType = std::remove_cvref_t< decltype( csr ) >;
   using CSCType = std::remove_cvref_t< decltype( csc ) >;

   static_assert(
      std::is_same_v<
         typename BSRType::backend_type,
         DefaultBackendFor_t< MatrixAssemblyType::BSR > > );
   static_assert(
      std::is_same_v<
         typename COOType::backend_type,
         DefaultBackendFor_t< MatrixAssemblyType::COO > > );
   static_assert(
      std::is_same_v<
         RawCOOType,
         RawCOOTripletBuffer< Real, GlobalIndex > > );
   static_assert(
      std::is_same_v<
         typename CSRType::backend_type,
         DefaultBackendFor_t< MatrixAssemblyType::CSR > > );
   static_assert(
      std::is_same_v<
         typename CSCType::backend_type,
         DefaultBackendFor_t< MatrixAssemblyType::CSC > > );

   bool success = true;
   success = Check(
      bsr.num_row_blocks == 1 && bsr.num_col_blocks == 1,
      "Typed BSR GenericAssembly returned the wrong matrix dimensions." ) && success;
   success = Check(
      coo.num_rows == 1 && coo.num_cols == 1 && coo.nnz == 1,
      "Typed COO GenericAssembly returned the wrong matrix dimensions." ) && success;
   success = Check(
      raw_coo.num_rows == 1 &&
      raw_coo.num_cols == 1 &&
      raw_coo.nnz_raw == 1,
      "Typed RawCOO GenericAssembly returned the wrong triplet dimensions." ) && success;
   success = Check(
      csr.num_rows == 1 && csr.num_cols == 1 && csr.nnz == 1,
      "Typed CSR GenericAssembly returned the wrong matrix dimensions." ) && success;
   success = Check(
      csc.num_rows == 1 && csc.num_cols == 1 && csc.nnz == 1,
      "Typed CSC GenericAssembly returned the wrong matrix dimensions." ) && success;

   FreeCSCMatrix( csc );
   FreeCSRMatrix( csr );
   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw_coo );
   return success;
}

} // namespace

int main()
{
   return TestTypedGenericAssemblyDefaults() ? 0 : 1;
}

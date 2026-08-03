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
      VendorDeviceBSRBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::SGBSR >,
      VendorDeviceBSRBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::COO >,
      VendorDeviceCOOBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::CSR >,
      VendorDeviceCSRBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::CSC >,
      VendorDeviceCSCBackend<> > );
static_assert(
   std::is_same_v< DefaultBSRBackend, VendorDeviceBSRBackend<> > );
static_assert(
   std::is_same_v< DefaultCOOBackend, VendorDeviceCOOBackend<> > );
static_assert(
   std::is_same_v< DefaultCSRBackend, VendorDeviceCSRBackend<> > );
static_assert(
   std::is_same_v< DefaultCSCBackend, VendorDeviceCSCBackend<> > );
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
static_assert(
   std::is_same_v< DefaultBSRBackend, HostBSRBackend<> > );
static_assert(
   std::is_same_v< DefaultCOOBackend, HostCOOBackend<> > );
static_assert(
   std::is_same_v< DefaultCSRBackend, HostCSRBackend<> > );
static_assert(
   std::is_same_v< DefaultCSCBackend, HostCSCBackend<> > );
#endif
static_assert(
   std::is_same_v<
      typename BSRMatrix<>::backend_type,
      DefaultBSRBackend > );
static_assert(
   std::is_same_v<
      typename COOMatrix<>::backend_type,
      DefaultCOOBackend > );
static_assert(
   std::is_same_v<
      typename CSRMatrix<>::backend_type,
      DefaultCSRBackend > );
static_assert(
   std::is_same_v<
      typename CSCMatrix<>::backend_type,
      DefaultCSCBackend > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::RawCOO >,
      Empty > );

#if defined(GENDIL_USE_DEVICE)
#if \
   defined(GENDIL_CUSPARSE_HAS_GENERIC_BSR) || \
   defined(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
using ExpectedDefaultAssembledBSRBackend = VendorDeviceBSRBackend<>;
#else
using ExpectedDefaultAssembledBSRBackend = NativeDeviceBSRBackend<>;
#endif
#else
using ExpectedDefaultAssembledBSRBackend = HostBSRBackend<>;
#endif

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
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
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
   using SGBSRType = std::remove_cvref_t< decltype( sgbsr ) >;
   using COOType = std::remove_cvref_t< decltype( coo ) >;
   using RawCOOType = std::remove_cvref_t< decltype( raw_coo ) >;
   using CSRType = std::remove_cvref_t< decltype( csr ) >;
   using CSCType = std::remove_cvref_t< decltype( csc ) >;

   static_assert(
      std::is_same_v<
         typename BSRType::backend_type,
         ExpectedDefaultAssembledBSRBackend > );
   static_assert(
      std::is_same_v<
         typename SGBSRType::backend_type,
         ExpectedDefaultAssembledBSRBackend > );
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
      sgbsr.TrialBsrSize() == 1 && sgbsr.TestBsrSize() == 1,
      "Typed SGBSR GenericAssembly returned the wrong matrix dimensions." ) &&
      success;
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

   return success;
}

} // namespace

int main()
{
   return TestTypedGenericAssemblyDefaults() ? 0 : 1;
}

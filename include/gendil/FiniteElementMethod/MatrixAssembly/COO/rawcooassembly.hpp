// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcoo.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/localinsertion.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoolayout.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/assemblydispatch.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/sparseassemblyvalidation.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"
#include "gendil/Utilities/KernelContext/kernelcontexttraits.hpp"
#include "gendil/Utilities/Loop/kernelloop.hpp"

#include <type_traits>

namespace gendil {

namespace details
{

template <
   bool OnDevice,
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
RawCOOTripletBuffer< ValueType, IndexType >
MakeAssemblyRawCOOTripletBuffer(
   const IndexType num_rows,
   const IndexType num_cols,
   const IndexType nnz_raw )
{
   auto buffer =
      AllocateRawCOOTripletBuffer< ValueType, IndexType >(
         num_rows,
         num_cols,
         nnz_raw );
   auto view = GetKernelWriteView< OnDevice >( buffer );
   KernelLoop< OnDevice >(
      nnz_raw,
      [=] GENDIL_HOST_DEVICE ( const IndexType i )
      {
         view.rows[i] = IndexType( 0 );
         view.cols[i] = IndexType( 0 );
         view.values[i] = ValueType( 0 );
      } );
   return buffer;
}

} // namespace details

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericRawCOOAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule )
{
   const auto can_instantiate =
      details::ValidateSparseAssemblyInputs<
         MatrixAssemblyType::RawCOO,
         details::SparseAssemblyMode::Full,
         KernelPolicy>(weak_form, wf_ctx);

   if constexpr (can_instantiate)
   {
      constexpr auto TrialName = requirements<WeakForm>::trial_name;
      constexpr auto TestName  = requirements<WeakForm>::test_name;

      const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
      const auto& test_space  = wf_ctx.template fe_field<TestName>().space;

      using TrialSpace = std::remove_cvref_t<decltype(trial_space)>;
      using TestSpace = std::remove_cvref_t<decltype(test_space)>;
      using TrialShapeFunctions =
         typename TrialSpace::finite_element_type::shape_functions;
      using TestShapeFunctions =
         typename TestSpace::finite_element_type::shape_functions;

      using OffsetType = RawCOOAssemblyLayout::offset_type;
      constexpr OffsetType ntrial =
         static_cast<OffsetType>(LocalDofCount<TrialShapeFunctions>());
      constexpr OffsetType ntest =
         static_cast<OffsetType>(LocalDofCount<TestShapeFunctions>());
      const OffsetType block_entry_count =
         CheckedRawCOOMultiply(
            ntest,
            ntrial,
            "RawCOO local trial/test block size overflow.");
      const auto& domain_mesh =
         GetCellIntegrationDomainMesh(weak_form, wf_ctx);

      auto layout =
         MakeRawCOOAssemblyLayout<
            has_cell_contributions_v< WeakForm >,
            has_boundary_facet_contributions_v< WeakForm >,
            has_interior_facet_contributions_v< WeakForm >>(
               domain_mesh,
               block_entry_count );

      constexpr bool on_device =
         is_device_configuration_v< KernelPolicy >;
      auto coo_buffer =
         details::MakeAssemblyRawCOOTripletBuffer<
            on_device,
            Real,
            GlobalIndex >(
               static_cast< GlobalIndex >(
                  test_space.GetNumberOfFiniteElementDofs() ),
               static_cast< GlobalIndex >(
                  trial_space.GetNumberOfFiniteElementDofs() ),
               layout.nnz_raw );
      auto coo_target =
         MakeRawCOOAssemblyTarget< on_device >( coo_buffer, layout );

      GenericAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         coo_target );

      if constexpr ( !is_host_configuration_v< KernelPolicy > )
      {
         GENDIL_DEVICE_SYNC;
      }

      return coo_buffer;
   }
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericRawCOOElementBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule )
{
   const auto can_instantiate =
      details::ValidateSparseAssemblyInputs<
         MatrixAssemblyType::RawCOO,
         details::SparseAssemblyMode::ElementBlockDiagonal,
         KernelPolicy>(weak_form, wf_ctx);

   if constexpr (can_instantiate)
   {
      constexpr auto TrialName = requirements<WeakForm>::trial_name;
      constexpr auto TestName = requirements<WeakForm>::test_name;

      const auto& trial_space =
         wf_ctx.template fe_field<TrialName>().space;
      const auto& test_space =
         wf_ctx.template fe_field<TestName>().space;

      using TrialSpace = std::remove_cvref_t<decltype(trial_space)>;
      using TestSpace = std::remove_cvref_t<decltype(test_space)>;
      using TrialShapeFunctions =
         typename TrialSpace::finite_element_type::shape_functions;
      using TestShapeFunctions =
         typename TestSpace::finite_element_type::shape_functions;

      using OffsetType = RawCOOAssemblyLayout::offset_type;
      constexpr OffsetType ntrial =
         static_cast<OffsetType>(LocalDofCount<TrialShapeFunctions>());
      constexpr OffsetType ntest =
         static_cast<OffsetType>(LocalDofCount<TestShapeFunctions>());
      const OffsetType block_entry_count =
         CheckedRawCOOMultiply(
            ntest,
            ntrial,
            "Element RawCOO local trial/test block size overflow.");
      const auto& domain_mesh =
         GetCellIntegrationDomainMesh(weak_form, wf_ctx);

      auto layout =
         MakeRawCOOElementBlockDiagonalAssemblyLayout<
            has_cell_contributions_v<WeakForm>,
            has_boundary_facet_contributions_v<WeakForm>,
            has_interior_facet_contributions_v<WeakForm>>(
               domain_mesh,
               block_entry_count);

      constexpr bool on_device =
         is_device_configuration_v< KernelPolicy >;
      auto coo_buffer =
         details::MakeAssemblyRawCOOTripletBuffer<
            on_device,
            Real,
            GlobalIndex >(
               static_cast<GlobalIndex>(
                  test_space.GetNumberOfFiniteElementDofs()),
               static_cast<GlobalIndex>(
                  trial_space.GetNumberOfFiniteElementDofs()),
               layout.nnz_raw);
      auto coo_target =
         MakeRawCOOAssemblyTarget< on_device >( coo_buffer, layout );

      AssembleElementBlockDiagonalSparseTarget<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         coo_target);

      if constexpr ( !is_host_configuration_v< KernelPolicy > )
      {
         GENDIL_DEVICE_SYNC;
      }

      return coo_buffer;
   }
}

} // namespace gendil

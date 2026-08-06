// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcoo.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/localinsertion.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoolayout.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/assemblydispatch.hpp"
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

template < typename FESpace >
struct IsRawCOOCellAssemblySpace
{
   using Space = std::remove_cvref_t< FESpace >;
   using ShapeFunctions =
      typename Space::finite_element_type::shape_functions;
   using Restriction = typename Space::restriction_type;

   static constexpr bool vector_h1_value = [] {
      if constexpr ( is_vector_h1_restriction_v< Restriction > )
      {
         static_assert(
            !is_vector_shape_functions_v< ShapeFunctions > ||
               Restriction::num_comp == ShapeFunctions::vector_dim,
            "VectorH1Restriction<NComp> must match the vector finite element component count." );

         return is_vector_shape_functions_v< ShapeFunctions >;
      }
      else
      {
         return false;
      }
   }();

   static constexpr bool value =
      restriction_traits< Restriction >::is_direct_index_map &&
      ( std::is_same_v< Restriction, L2Restriction > ||
        ( std::is_same_v< Restriction, H1Restriction > &&
          !is_vector_shape_functions_v< ShapeFunctions > ) ||
        ( is_tensor_product_restriction_v< Restriction > &&
          !is_vector_shape_functions_v< ShapeFunctions > ) ||
        vector_h1_value );
};

template < typename FESpace >
struct IsRawCOOFaceAssemblySpace
{
   using Space = std::remove_cvref_t< FESpace >;

   static constexpr bool value =
      std::is_same_v< typename Space::restriction_type, L2Restriction >;
};

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
   using I = std::remove_cvref_t<WeakForm>;
   ValidateSparseLinearAssemblyCoefficientInputs<I>();

   ValidateSparseAssemblyDomainSupport<WeakForm, WeakFormContext>();

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TestName  = requirements<I>::test_name;

   static_assert(TrialName != StaticString{"Error"}, "GenericAssembly<RawCOO>: missing TrialSpace in integrand.");
   static_assert(TestName  != StaticString{"Error"}, "GenericAssembly<RawCOO>: missing TestSpace in integrand.");
   static_assert(
      has_cell_contributions_v< I > ||
      has_boundary_facet_contributions_v< I > ||
      has_interior_facet_contributions_v< I >,
      "GenericAssembly<RawCOO> requires at least one active weak-form domain." );

   ValidateWeakFormContext(weak_form, wf_ctx);

   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   const auto& test_space  = wf_ctx.template fe_field<TestName>().space;

   using TrialSpace = std::remove_cvref_t<decltype(trial_space)>;
   using TestSpace = std::remove_cvref_t<decltype(test_space)>;
   using TrialShapeFunctions =
      typename TrialSpace::finite_element_type::shape_functions;
   using TestShapeFunctions =
      typename TestSpace::finite_element_type::shape_functions;

   constexpr bool has_face_terms =
      has_boundary_facet_contributions_v< I > ||
      has_interior_facet_contributions_v< I >;

   static_assert(
      ( !has_face_terms &&
        IsRawCOOCellAssemblySpace< TrialSpace >::value &&
        IsRawCOOCellAssemblySpace< TestSpace >::value ) ||
      ( has_face_terms &&
        IsRawCOOFaceAssemblySpace< TrialSpace >::value &&
        IsRawCOOFaceAssemblySpace< TestSpace >::value ),
      "GenericAssembly<RawCOO> supports scalar/vector L2/DG cell-only terms, "
      "scalar/vector H1/CG cell-only terms, scalar tensor-product direct-index "
      "cell-only terms, and scalar/vector L2/DG conforming face terms. H1 face "
      "terms, nonconforming faces, global face traversal, and "
      "variable-size hp emission are unsupported." );

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
         has_cell_contributions_v< I >,
         has_boundary_facet_contributions_v< I >,
         has_interior_facet_contributions_v< I >>(
            domain_mesh,
            block_entry_count );

   static_assert(
      is_host_configuration_v< KernelPolicy > !=
         is_device_configuration_v< KernelPolicy >,
      "RawCOO assembly requires a host or device kernel policy." );
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
   using I = std::remove_cvref_t<WeakForm>;
   ValidateSparseLinearAssemblyCoefficientInputs<I>();
   ValidateSparseAssemblyDomainSupport<WeakForm, WeakFormContext>();

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TestName = requirements<I>::test_name;

   static_assert(
      TrialName != StaticString{"Error"},
      "GenericRawCOOElementBlockDiagonalAssembly: missing TrialSpace in "
      "integrand.");
   static_assert(
      TestName != StaticString{"Error"},
      "GenericRawCOOElementBlockDiagonalAssembly: missing TestSpace in "
      "integrand.");
   static_assert(
      has_cell_contributions_v<I> ||
      has_boundary_facet_contributions_v<I> ||
      has_interior_facet_contributions_v<I>,
      "GenericRawCOOElementBlockDiagonalAssembly requires at least one "
      "active weak-form domain.");

   ValidateWeakFormContext(weak_form, wf_ctx);

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

   constexpr bool has_face_terms =
      has_boundary_facet_contributions_v<I> ||
      has_interior_facet_contributions_v<I>;

   static_assert(
      (!has_face_terms &&
       IsRawCOOCellAssemblySpace<TrialSpace>::value &&
       IsRawCOOCellAssemblySpace<TestSpace>::value) ||
      (has_face_terms &&
       IsRawCOOFaceAssemblySpace<TrialSpace>::value &&
       IsRawCOOFaceAssemblySpace<TestSpace>::value),
      "GenericRawCOOElementBlockDiagonalAssembly supports scalar/vector "
      "L2/DG cell-only terms, scalar/vector H1/CG cell-only terms, scalar "
      "tensor-product direct-index cell-only terms, and scalar/vector L2/DG "
      "conforming face terms. H1 face terms, nonconforming "
      "faces, global face traversal, and variable-size hp emission are "
      "unsupported.");

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
         has_cell_contributions_v<I>,
         has_boundary_facet_contributions_v<I>,
         has_interior_facet_contributions_v<I>>(
            domain_mesh,
            block_entry_count);

   static_assert(
      is_host_configuration_v< KernelPolicy > !=
         is_device_configuration_v< KernelPolicy >,
      "RawCOO assembly requires a host or device kernel policy." );
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

} // namespace gendil

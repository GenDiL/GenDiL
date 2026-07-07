// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcootripletbuffer.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/localinsertion.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/weakformtraversal.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"

#include <type_traits>

namespace gendil {

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

   static_assert(
      !weak_form_context_has_mixed_sparse_domain_v<WeakFormContext>,
      "GenericAssembly<RawCOO>: mixed sparse assembly for "
      "MakeIntegrationDomain<Name>(mixed_fes) is deferred. Homogeneous "
      "sparse assembly currently supports MakeIntegrationDomain<Name>(fe_space).");

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TestName  = requirements<I>::test_name;

   static_assert(TrialName != StaticString{"Error"}, "GenericAssembly<RawCOO>: missing TrialSpace in integrand.");
   static_assert(TestName  != StaticString{"Error"}, "GenericAssembly<RawCOO>: missing TestSpace in integrand.");
   static_assert(
      has_cell_contributions_v< I > ||
      has_boundary_facet_contributions_v< I > ||
      has_interior_facet_contributions_v< I >,
      "GenericAssembly<RawCOO> requires at least one active weak-form domain." );

   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   const auto& test_space  = wf_ctx.template fe_field<TestName>().space;

   using TrialSpace = std::remove_cvref_t<decltype(trial_space)>;
   using TestSpace = std::remove_cvref_t<decltype(test_space)>;
   using TrialShapeFunctions =
      typename TrialSpace::finite_element_type::shape_functions;
   using TestShapeFunctions =
      typename TestSpace::finite_element_type::shape_functions;

   static_assert(
      std::is_same_v< TrialSpace, TestSpace >,
      "GenericAssembly<RawCOO> requires matching trial/test FE spaces; mixed/rectangular spaces are unsupported." );

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
      "terms, mixed spaces, nonconforming faces, global face traversal, and "
      "variable-size hp emission are unsupported." );

   constexpr GlobalIndex ntrial = LocalDofCount< TrialShapeFunctions >();
   constexpr GlobalIndex ntest = LocalDofCount< TestShapeFunctions >();
   constexpr GlobalIndex block_entry_count = ntest * ntrial;

   auto layout =
      MakeRawCOOAssemblyLayout<
         has_cell_contributions_v< I >,
         has_boundary_facet_contributions_v< I >,
         has_interior_facet_contributions_v< I > >(
            trial_space,
            block_entry_count );

   auto coo_buffer =
      MakeRawCOOTripletBuffer< Real, GlobalIndex >(
         static_cast< GlobalIndex >( test_space.GetNumberOfFiniteElementDofs() ),
         static_cast< GlobalIndex >( trial_space.GetNumberOfFiniteElementDofs() ),
         layout.nnz_raw );

   RawCOOAssemblyTarget< Real, GlobalIndex > coo_target{
      coo_buffer,
      layout
   };

   GenericAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      coo_target );

   SyncRawCOOTripletBuffer< KernelPolicy >( coo_buffer );
   FreeRawCOOAssemblyLayout( layout );

   return coo_buffer;
}

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/CellOperator/celloperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/LocalFacetOperator/localfacetsoperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/domainiteration.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globaloperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/GlobalFacetOperator/globalinteriorfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/BoundaryFacetOperator/boundaryfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/restrictedweakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/readdofs.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/writedofs.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumformexpr.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <tuple>
#include <utility>

namespace gendil {

namespace global_facet_operator_detail {

// Reuse the domain already owned by the invocation bindings. The kernel does
// not need another captured copy of the selected batch or its meshes.
template < typename Batch, typename ExecutionContext >
GENDIL_HOST_DEVICE
const auto & GetSelectedFaceDomain( const ExecutionContext & execution_context )
{
   const auto & registry = execution_context.GetRegistry();
   constexpr auto DomainName = Batch::domain_name;

   if constexpr ( is_selected_interior_face_execution_domain_v< Batch > )
   {
      return registry.template GetInteriorFaceDomain< DomainName >().domain;
   }
   else
   {
      return registry.template GetBoundaryFaceDomain< DomainName >().domain;
   }
}

template <
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename FaceDomain,
   typename FaceInfo,
   typename Integrand >
GENDIL_HOST_DEVICE
void ApplyFacetIntegrand(
   KernelContext & kernel_context,
   ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const FaceDomain & face_domain,
   const FaceInfo & face_info,
   const Integrand & integrand )
{
   if constexpr ( InteriorFacetIntegrand< Integrand > )
   {
      GlobalInteriorFacetIntegrandOperator(
         kernel_context,
         execution_context,
         operator_context,
         face_domain,
         face_info,
         integrand );
   }
   else
   {
      const GlobalIndex element_index = face_info.MinusSide().GetCellIndex();
      ElementContext element_context{
         element_index,
         face_domain.GetCellMesh().GetCell( element_index ) };

      BoundaryFacetIntegrandOperator(
         kernel_context,
         execution_context,
         operator_context,
         element_context,
         face_info,
         integrand );
   }
}

} // namespace global_facet_operator_detail

// Prepare once and evaluate the selected original integrands in one face
// traversal. Only the selected terms and shallow contexts enter the kernel.
template <
   typename KernelPolicy,
   typename WeakForm,
   typename SelectedContext,
   typename Batch,
   typename IntegrationRule,
   typename ExecutionContext >
   requires ( is_selected_interior_face_execution_domain_v< Batch > ||
              is_selected_boundary_face_execution_domain_v< Batch > )
void GenericGlobalFacetDomainOperator(
   const WeakForm & weak_form,
   const SelectedContext & selected_context,
   const Batch & batch,
   const IntegrationRule & integration_rule,
   const ExecutionContext & execution_context )
{
   constexpr auto DomainName = Batch::domain_name;
   constexpr auto Entity = is_selected_interior_face_execution_domain_v< Batch >
      ? SemanticEntity::InteriorFacet : SemanticEntity::BoundaryFacet;

   auto integrands =
      weak_form_execution_detail::SelectIntegrands< DomainName, Entity >( weak_form );
   auto operator_context = MakeFacetOperatorContext(
      weak_form,
      selected_context,
      integration_rule,
      batch );

   using Integrands = decltype( integrands );
   using OperatorContext = decltype( operator_context );
   constexpr size_t required_shared_memory =
      detail::TermSharedSlots<
         KernelPolicy,
         ExecutionContext,
         OperatorContext >( static_cast< Integrands * >( nullptr ) );
   using Kernel = KernelContext< KernelPolicy, required_shared_memory >;

   mesh::GlobalFaceIterator< KernelPolicy >(
      batch.GetFaceMesh(),
      [integrands = std::move( integrands ),
       operator_context = std::move( operator_context ),
       execution_context = execution_context]
      GENDIL_HOST_DEVICE ( GlobalIndex face_index ) mutable
      {
         (void)integrands;
         (void)operator_context;
         (void)execution_context;

         GENDIL_SHARED Real shared_memory[Kernel::shared_memory_block_size];
         Kernel kernel_context( shared_memory );

         const auto & face_domain =
            global_facet_operator_detail::GetSelectedFaceDomain< Batch >( execution_context );
         const auto face_info = face_domain.GetFaceMesh().GetGlobalFaceInfo( face_index );

         std::apply(
            [&]( const auto &... integrand )
            {
               ( global_facet_operator_detail::ApplyFacetIntegrand(
                    kernel_context,
                    execution_context,
                    operator_context,
                    face_domain,
                    face_info,
                    integrand ), ... );
            },
            integrands );
      } );
}

namespace global_facet_operator_detail {

// Select requirements once per named domain, then reuse the invocation's
// vector access while selecting and binding each concrete batch.
template <
   typename KernelPolicy,
   SemanticEntity Entity,
   typename WeakForm,
   typename WeakFormContext,
   typename IntegrationRule,
   typename... Access >
void GenericGlobalFacetOperator(
   const WeakForm & weak_form,
   const WeakFormContext & weak_form_context,
   const IntegrationRule & integration_rule,
   const Access &... access )
{
   using FormSignature = form_signature_t< WeakForm >;
   using DomainNames = generic_operator_domain_names_t< WeakForm >;

   ForEachGenericOperatorDomainName(
      DomainNames{},
      [&]( auto domain_name )
      {
         constexpr auto DomainName = decltype( domain_name )::name;
         if constexpr ( has_nonzero_integrands< WeakForm, DomainName, Entity > )
         {
            using Signatures = typename weak_form_traits::domain_signatures<
               typename FormSignature::integrands, DomainName, Entity >::type;
            using Requirements = typename weak_form_execution_detail::signature_requirements< Signatures >::type;
            auto apply_batch = [&]( const auto & batch )
            {
               auto selected_context = MakeRestrictedWeakFormContext< Requirements >(
                  weak_form_context,
                  batch );
               auto execution_context = execution_context_detail::BindExecutionContext< Requirements, KernelPolicy >(
                  selected_context,
                  access... );

               if constexpr ( Entity == SemanticEntity::Cell )
               {
                  GenericCellBatchOperator< KernelPolicy, SemanticEntity::Cell >(
                     weak_form,
                     selected_context,
                     batch,
                     integration_rule,
                     execution_context );
               }
               else
               {
                  GenericGlobalFacetDomainOperator< KernelPolicy >(
                     weak_form,
                     selected_context,
                     batch,
                     integration_rule,
                     execution_context );
               }
            };

            if constexpr ( Entity == SemanticEntity::Cell )
            {
               ForEachCellExecutionDomain(
                  weak_form_context,
                  Cells< DomainName >{},
                  apply_batch );
            }
            else if constexpr ( Entity == SemanticEntity::InteriorFacet )
            {
               static_assert(
                  WeakFormContext::template HasInteriorFaceDomain< DomainName >(),
                  "InteriorFacets<Name>: PartitionIntegrationDomain has no interior face parts; "
                  "partition domains do not fall back to local-facet traversal." );

               if constexpr ( WeakFormContext::template HasInteriorFaceDomain< DomainName >() )
               {
                  ForEachInteriorFaceExecutionDomain(
                     weak_form_context,
                     InteriorFacets< DomainName >{},
                     apply_batch );
               }
            }
            else
            {
               static_assert(
                  WeakFormContext::template HasBoundaryFaceDomain< DomainName >(),
                  "BoundaryFacets<Name>: PartitionIntegrationDomain has no boundary face parts; "
                  "partition domains do not fall back to local-facet traversal." );

               if constexpr ( WeakFormContext::template HasBoundaryFaceDomain< DomainName >() )
               {
                  ForEachBoundaryFaceExecutionDomain(
                     weak_form_context,
                     BoundaryFacets< DomainName >{},
                     apply_batch );
               }
            }
         }
      } );
}

} // namespace global_facet_operator_detail

// Common vector access is acquired by the invocation and reused by all phases.
// Preserve cell/interior/boundary order and source order within each batch.
template <
   typename KernelPolicy,
   typename WeakForm,
   typename WeakFormContext,
   typename IntegrationRule,
   typename... Access >
void GenericGlobalFacetDomainOperator(
   const WeakForm & weak_form,
   const WeakFormContext & weak_form_context,
   const IntegrationRule & integration_rule,
   const Access &... access )
{
   global_facet_operator_detail::GenericGlobalFacetOperator< KernelPolicy, SemanticEntity::Cell >(
      weak_form,
      weak_form_context,
      integration_rule,
      access... );

   global_facet_operator_detail::GenericGlobalFacetOperator< KernelPolicy, SemanticEntity::InteriorFacet >(
      weak_form,
      weak_form_context,
      integration_rule,
      access... );

   global_facet_operator_detail::GenericGlobalFacetOperator< KernelPolicy, SemanticEntity::BoundaryFacet >(
      weak_form,
      weak_form_context,
      integration_rule,
      access... );
}

} // namespace gendil

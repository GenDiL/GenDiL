// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/selectedexecutiondomain.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/weakformrequirements.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrationdomain.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumformexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

// Collect the named domains used by the original integrands.
template < typename WeakForm >
struct generic_operator_domain_names
{
   using type = type_list<>;
};

template < StaticString DomainName, FieldExpr Expr >
struct generic_operator_domain_names< Integrand< Cells< DomainName >, Expr > >
{
   using type = type_list< NameTag< DomainName > >;
};

template < StaticString DomainName, FieldExpr Expr >
struct generic_operator_domain_names< Integrand< InteriorFacets< DomainName >, Expr > >
{
   using type = type_list< NameTag< DomainName > >;
};

template < StaticString DomainName, FieldExpr Expr >
struct generic_operator_domain_names< Integrand< BoundaryFacets< DomainName >, Expr > >
{
   using type = type_list< NameTag< DomainName > >;
};

template < typename... Terms >
struct generic_operator_domain_names< std::tuple< Terms... > >
{
   using type = unique_t< concat_many_t<
      typename generic_operator_domain_names< std::remove_cvref_t< Terms > >::type... > >;
};

template < typename Terms, typename Symbols >
struct generic_operator_domain_names< SumFormExpr< Terms, Symbols > >
   : generic_operator_domain_names< Terms >
{};

template < typename WeakForm >
using generic_operator_domain_names_t =
   typename generic_operator_domain_names< std::remove_cvref_t< WeakForm > >::type;

template < typename Function, typename... NameTags >
void ForEachGenericOperatorDomainName(
   type_list< NameTags... >,
   Function && function )
{
   ( function( NameTags{} ), ... );
}

namespace weak_form_execution_detail
{

// Own only the selected original terms for kernel capture. Their complete-form
// signatures remain in the operator context; selection does not reanalyze them.
template <
   StaticString DomainName,
   SemanticEntity Entity = SemanticEntity::Unspecified,
   typename WeakForm >
constexpr auto SelectIntegrands( const WeakForm & weak_form )
{
   const auto & original_terms = AsFormTerms( weak_form );
   using Terms = std::remove_cvref_t< decltype( original_terms ) >;
   constexpr auto indices = DomainTermIndices< DomainName, Terms, Entity >(
      std::make_index_sequence< std::tuple_size_v< Terms > >{} );

   return std::apply(
      [&] ( auto... term_index )
      {
         return std::tuple{
            std::get< decltype( term_index )::value >( original_terms )... };
      },
      indices );
}

} // namespace weak_form_execution_detail

// Cell traversal visits one mesh or each cell part of a partition.
template <
   StaticString DomainName,
   typename Domain,
   typename Function >
void ForEachCellExecutionDomainInDomain(
   const CellIntegrationDomain< Domain > & normalized_domain,
   Function && function )
{
   using IntegrationDomain = std::remove_cvref_t< Domain >;

   if constexpr ( is_mesh_integration_domain_v< IntegrationDomain > )
   {
      using Mesh = typename IntegrationDomain::mesh_type;
      function(
         SelectedCellExecutionDomain< DomainName, 0, Mesh, false >{
            Mesh{ normalized_domain.domain.mesh } } );
   }
   else if constexpr ( is_partition_integration_domain_v< IntegrationDomain > )
   {
      const auto & cell_parts = normalized_domain.domain.partition.CellParts();
      using CellParts = std::remove_cvref_t< decltype( cell_parts ) >;
      constexpr size_t NumCellParts = std::tuple_size_v< CellParts >;

      ConstexprLoop< NumCellParts >(
         [&] ( auto index )
         {
            constexpr size_t CellPartIndex = decltype( index )::value;
            const auto & mesh = std::get< CellPartIndex >( cell_parts ).mesh;
            using Mesh = std::remove_cvref_t< decltype( mesh ) >;

            function(
               SelectedCellExecutionDomain< DomainName, CellPartIndex, Mesh, true >{
                  Mesh{ mesh } } );
         } );
   }
   else
   {
      static_assert(
         dependent_false_v< IntegrationDomain >,
         "Cells<Name> requires a normalized mesh or partition integration "
         "domain." );
   }
}

template <
   typename WeakFormContext,
   StaticString DomainName,
   typename Function >
void ForEachCellExecutionDomain(
   const WeakFormContext & weak_form_context,
   Cells< DomainName >,
   Function && function )
{
   using Context = std::remove_cvref_t< WeakFormContext >;

   static_assert(
      Context::template HasDomain< DomainName >(),
      "Cells<Name> requires an integration domain registered under Name." );

   if constexpr ( Context::template HasDomain< DomainName >() )
   {
      ForEachCellExecutionDomainInDomain< DomainName >(
         weak_form_context.template GetDomain< DomainName >(),
         std::forward< Function >( function ) );
   }
}

// Interior facets use local mesh connectivity or explicit partition face parts.
template <
   StaticString DomainName,
   typename Domain,
   typename Function >
void ForEachGlobalInteriorFaceExecutionDomain(
   const InteriorFaceIntegrationDomain< Domain > & normalized_domain,
   Function && function )
{
   using IntegrationDomain = std::remove_cvref_t< Domain >;

   static_assert(
      is_partition_integration_domain_v< IntegrationDomain >,
      "InteriorFacets<Name> global traversal requires a "
      "PartitionIntegrationDomain." );

   if constexpr ( is_partition_integration_domain_v< IntegrationDomain > )
   {
      const auto & partition = normalized_domain.domain.partition;
      const auto & face_parts = partition.InteriorFaceParts();
      using FaceParts = std::remove_cvref_t< decltype( face_parts ) >;
      constexpr size_t NumFaceParts = std::tuple_size_v< FaceParts >;

      static_assert(
         NumFaceParts > 0,
         "InteriorFacets<Name>: PartitionIntegrationDomain has no interior "
         "face parts; partition domains do not fall back to local-facet "
         "traversal." );

      ConstexprLoop< NumFaceParts >(
         [&] ( auto index )
         {
            constexpr size_t FacePartIndex = decltype( index )::value;
            const auto & face_part = std::get< FacePartIndex >( face_parts );
            using FacePart = std::remove_cvref_t< decltype( face_part ) >;
            constexpr size_t MinusCellPartIndex = FacePart::minus_cell_index;
            constexpr size_t PlusCellPartIndex = FacePart::plus_cell_index;
            const auto & minus_mesh =
               std::get< MinusCellPartIndex >( partition.CellParts() ).mesh;
            const auto & plus_mesh =
               std::get< PlusCellPartIndex >( partition.CellParts() ).mesh;
            using MinusMesh = std::remove_cvref_t< decltype( minus_mesh ) >;
            using PlusMesh = std::remove_cvref_t< decltype( plus_mesh ) >;

            function(
               SelectedInteriorFaceExecutionDomain<
                  DomainName,
                  FacePartIndex,
                  FacePart,
                  MinusMesh,
                  PlusMesh >{
                     FacePart{ face_part },
                     MinusMesh{ minus_mesh },
                     PlusMesh{ plus_mesh } } );
         } );
   }
}

template <
   StaticString DomainName,
   typename Domain,
   typename Function >
void ForEachLocalInteriorFaceExecutionDomain(
   const CellIntegrationDomain< Domain > & normalized_domain,
   Function && function )
{
   using IntegrationDomain = std::remove_cvref_t< Domain >;

   if constexpr ( is_mesh_integration_domain_v< IntegrationDomain > )
   {
      using Mesh = typename IntegrationDomain::mesh_type;
      function(
         SelectedCellExecutionDomain< DomainName, 0, Mesh, false >{
            Mesh{ normalized_domain.domain.mesh } } );
   }
   else
   {
      static_assert(
         dependent_false_v< IntegrationDomain >,
         "InteriorFacets<Name>: PartitionIntegrationDomain has no interior "
         "face parts; partition domains do not fall back to local-facet "
         "traversal." );
   }
}

template <
   typename WeakFormContext,
   StaticString DomainName,
   typename Function >
void ForEachInteriorFaceExecutionDomain(
   const WeakFormContext & weak_form_context,
   InteriorFacets< DomainName >,
   Function && function )
{
   using Context = std::remove_cvref_t< WeakFormContext >;

   if constexpr ( Context::template HasInteriorFaceDomain< DomainName >() )
   {
      ForEachGlobalInteriorFaceExecutionDomain< DomainName >(
         weak_form_context.template GetInteriorFaceDomain< DomainName >(),
         std::forward< Function >( function ) );
   }
   else if constexpr ( Context::template HasDomain< DomainName >() )
   {
      ForEachLocalInteriorFaceExecutionDomain< DomainName >(
         weak_form_context.template GetDomain< DomainName >(),
         std::forward< Function >( function ) );
   }
   else
   {
      static_assert(
         dependent_false_v< Context >,
         "InteriorFacets<Name> requires an integration domain registered "
         "under Name." );
   }
}

// Boundary facets use local mesh connectivity or explicit partition face parts.
template <
   StaticString DomainName,
   typename Domain,
   typename Function >
void ForEachGlobalBoundaryFaceExecutionDomain(
   const BoundaryFaceIntegrationDomain< Domain > & normalized_domain,
   Function && function )
{
   using IntegrationDomain = std::remove_cvref_t< Domain >;

   static_assert(
      is_partition_integration_domain_v< IntegrationDomain >,
      "BoundaryFacets<Name> global traversal requires a "
      "PartitionIntegrationDomain." );

   if constexpr ( is_partition_integration_domain_v< IntegrationDomain > )
   {
      const auto & partition = normalized_domain.domain.partition;
      const auto & face_parts = partition.BoundaryFaceParts();
      using FaceParts = std::remove_cvref_t< decltype( face_parts ) >;
      constexpr size_t NumFaceParts = std::tuple_size_v< FaceParts >;

      static_assert(
         NumFaceParts > 0,
         "BoundaryFacets<Name>: PartitionIntegrationDomain has no boundary "
         "face parts; partition domains do not fall back to local-facet "
         "traversal." );

      ConstexprLoop< NumFaceParts >(
         [&] ( auto index )
         {
            constexpr size_t FacePartIndex = decltype( index )::value;
            const auto & face_part = std::get< FacePartIndex >( face_parts );
            using FacePart = std::remove_cvref_t< decltype( face_part ) >;
            constexpr size_t CellPartIndex = FacePart::cell_index;
            const auto & cell_mesh =
               std::get< CellPartIndex >( partition.CellParts() ).mesh;
            using CellMesh = std::remove_cvref_t< decltype( cell_mesh ) >;

            function(
               SelectedBoundaryFaceExecutionDomain<
                  DomainName,
                  FacePartIndex,
                  FacePart,
                  CellMesh >{
                     FacePart{ face_part },
                     CellMesh{ cell_mesh } } );
         } );
   }
}

template <
   StaticString DomainName,
   typename Domain,
   typename Function >
void ForEachLocalBoundaryFaceExecutionDomain(
   const CellIntegrationDomain< Domain > & normalized_domain,
   Function && function )
{
   using IntegrationDomain = std::remove_cvref_t< Domain >;

   if constexpr ( is_mesh_integration_domain_v< IntegrationDomain > )
   {
      using Mesh = typename IntegrationDomain::mesh_type;
      function(
         SelectedCellExecutionDomain< DomainName, 0, Mesh, false >{
            Mesh{ normalized_domain.domain.mesh } } );
   }
   else
   {
      static_assert(
         dependent_false_v< IntegrationDomain >,
         "BoundaryFacets<Name>: PartitionIntegrationDomain has no boundary "
         "face parts; partition domains do not fall back to local-facet "
         "traversal." );
   }
}

template <
   typename WeakFormContext,
   StaticString DomainName,
   typename Function >
void ForEachBoundaryFaceExecutionDomain(
   const WeakFormContext & weak_form_context,
   BoundaryFacets< DomainName >,
   Function && function )
{
   using Context = std::remove_cvref_t< WeakFormContext >;

   if constexpr ( Context::template HasBoundaryFaceDomain< DomainName >() )
   {
      ForEachGlobalBoundaryFaceExecutionDomain< DomainName >(
         weak_form_context.template GetBoundaryFaceDomain< DomainName >(),
         std::forward< Function >( function ) );
   }
   else if constexpr ( Context::template HasDomain< DomainName >() )
   {
      ForEachLocalBoundaryFaceExecutionDomain< DomainName >(
         weak_form_context.template GetDomain< DomainName >(),
         std::forward< Function >( function ) );
   }
   else
   {
      static_assert(
         dependent_false_v< Context >,
         "BoundaryFacets<Name> requires an integration domain registered "
         "under Name." );
   }
}

} // namespace gendil

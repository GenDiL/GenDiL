// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/restriction.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/multiindex.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

template < typename DofShape >
struct DofShapeRank;

template < size_t ... Sizes >
struct DofShapeRank< std::index_sequence< Sizes... > >
   : std::integral_constant< size_t, sizeof...( Sizes ) > {};

template < typename TensorRestriction, size_t I >
using tensor_product_factor_type_t =
   std::tuple_element_t< I, typename TensorRestriction::factors_type >;

template < typename TensorRestriction, size_t I >
using tensor_product_factor_dof_shape_t =
   typename tensor_product_factor_type_t< TensorRestriction, I >::dof_shape;

template < typename TensorRestriction, size_t I >
inline constexpr size_t tensor_product_factor_dof_rank_v =
   DofShapeRank<
      tensor_product_factor_dof_shape_t< TensorRestriction, I > >::value;

template < typename TensorRestriction, size_t ... I >
GENDIL_HOST_DEVICE
constexpr size_t TensorProductDofRank( std::index_sequence< I... > )
{
   return ( size_t{0} + ... +
      tensor_product_factor_dof_rank_v< TensorRestriction, I > );
}

template < typename TensorRestriction >
inline constexpr size_t tensor_product_dof_rank_v =
   TensorProductDofRank< TensorRestriction >(
      std::make_index_sequence< TensorRestriction::num_factors >{} );

template < typename TensorRestriction, size_t I, size_t ... J >
GENDIL_HOST_DEVICE
constexpr size_t TensorProductLocalDofRankOffset( std::index_sequence< J... > )
{
   return ( size_t{0} + ... +
      tensor_product_factor_dof_rank_v< TensorRestriction, J > );
}

template < typename TensorRestriction, size_t I >
inline constexpr size_t tensor_product_local_dof_rank_offset_v =
   TensorProductLocalDofRankOffset< TensorRestriction, I >(
      std::make_index_sequence< I >{} );

template < typename TensorRestriction, size_t I >
GENDIL_HOST_DEVICE
constexpr GlobalIndex TensorProductFactorLocalDofCount()
{
   return static_cast< GlobalIndex >(
      Product( tensor_product_factor_dof_shape_t< TensorRestriction, I >{} ) );
}

template < size_t N >
GENDIL_HOST_DEVICE
std::array< GlobalIndex, N > MakePrefixStrides(
   const std::array< GlobalIndex, N > & counts )
{
   std::array< GlobalIndex, N > strides{};
   GlobalIndex stride = 1;
   for ( size_t i = 0; i < N; ++i )
   {
      strides[i] = stride;
      stride *= counts[i];
   }
   return strides;
}

template < typename TensorRestriction >
GENDIL_HOST_DEVICE
void TensorProductElementIndices(
   const TensorRestriction & restriction,
   const GlobalIndex element_index,
   std::array< GlobalIndex, TensorRestriction::num_factors > & element_indices )
{
   static_assert(
      TensorRestriction::num_factors > 0,
      "TensorProductRestriction must have at least one factor." );

   GlobalIndex remaining = element_index;
   for ( size_t i = TensorRestriction::num_factors - 1; i > 0; --i )
   {
      element_indices[i] = remaining / restriction.element_strides[i];
      remaining -= restriction.element_strides[i] * element_indices[i];
   }
   element_indices[0] = remaining;
}

template < typename Restriction >
GENDIL_HOST_DEVICE
GlobalIndex DirectIndexFactorTopologyDof(
   const Restriction & restriction,
   const GlobalIndex element_index,
   const GlobalIndex local_dof_index,
   const GlobalIndex local_dof_count )
{
   using FactorRestriction = std::remove_cvref_t< Restriction >;

   if constexpr ( std::is_same_v< FactorRestriction, L2Restriction > )
   {
      return element_index * local_dof_count + local_dof_index;
   }
   else if constexpr ( std::is_same_v< FactorRestriction, H1Restriction > )
   {
      const GlobalIndex restriction_index =
         element_index * local_dof_count + local_dof_index;
      const int global_index = restriction.indices[restriction_index];
      GENDIL_VERIFY(
         global_index >= 0,
         "H1Restriction contains a negative element-to-global DoF index." );
      return static_cast< GlobalIndex >( global_index );
   }
   else
   {
      static_assert(
         dependent_false_v< FactorRestriction >,
         "TensorProductRestriction v1 supports scalar L2Restriction and H1Restriction factors only." );
      return 0;
   }
}

template <
   typename TensorRestriction,
   size_t I >
GENDIL_HOST_DEVICE
GlobalIndex TensorProductFactorTopologyDof(
   const TensorRestriction & restriction,
   const std::array< GlobalIndex, TensorRestriction::num_factors > & element_indices,
   const std::array< GlobalIndex, TensorRestriction::num_factors > & local_dof_indices )
{
   return DirectIndexFactorTopologyDof(
      std::get< I >( restriction.restrictions ),
      element_indices[I],
      local_dof_indices[I],
      TensorProductFactorLocalDofCount< TensorRestriction, I >() );
}

template <
   typename TensorRestriction,
   size_t ... I >
GENDIL_HOST_DEVICE
GlobalIndex TensorProductElementToGlobalDofIndex(
   const TensorRestriction & restriction,
   const std::array< GlobalIndex, TensorRestriction::num_factors > & element_indices,
   const std::array< GlobalIndex, TensorRestriction::num_factors > & local_dof_indices,
   std::index_sequence< I... > )
{
   GlobalIndex global_index = 0;
   (
      ( global_index +=
           TensorProductFactorTopologyDof< TensorRestriction, I >(
              restriction,
              element_indices,
              local_dof_indices ) * restriction.global_dof_strides[I] ),
      ... );
   return global_index;
}

template < typename TensorRestriction, size_t ... I >
GENDIL_HOST_DEVICE
void DecomposeTensorProductLocalDofOrdinal(
   GlobalIndex ordinal,
   std::array< GlobalIndex, TensorRestriction::num_factors > & indices,
   std::index_sequence< I... > )
{
   (
      ( indices[I] =
           ordinal % TensorProductFactorLocalDofCount<
              TensorRestriction,
              I >(),
        ordinal =
           ordinal / TensorProductFactorLocalDofCount<
              TensorRestriction,
              I >() ),
      ... );
}

template < typename TensorRestriction >
GENDIL_HOST_DEVICE
GlobalIndex TensorProductElementToGlobalDofIndex(
   const TensorRestriction & restriction,
   const GlobalIndex element_index,
   const GlobalIndex local_dof_index )
{
   std::array< GlobalIndex, TensorRestriction::num_factors > element_indices{};
   std::array< GlobalIndex, TensorRestriction::num_factors > local_dof_indices{};

   TensorProductElementIndices(
      restriction,
      element_index,
      element_indices );
   DecomposeTensorProductLocalDofOrdinal< TensorRestriction >(
      local_dof_index,
      local_dof_indices,
      std::make_index_sequence< TensorRestriction::num_factors >{} );

   return TensorProductElementToGlobalDofIndex(
      restriction,
      element_indices,
      local_dof_indices,
      std::make_index_sequence< TensorRestriction::num_factors >{} );
}

template < typename TensorRestriction >
struct TensorProductLayout
{
   static constexpr size_t rank =
      tensor_product_dof_rank_v< TensorRestriction > + 1;

   TensorRestriction restriction;

   template <
      size_t I,
      typename Tuple,
      size_t ... J >
   GENDIL_HOST_DEVICE
   GlobalIndex FactorTopologyDof(
      const Tuple & local_indices,
      const std::array<
         GlobalIndex,
         TensorRestriction::num_factors > & element_indices,
      std::index_sequence< J... > ) const
   {
      using DofShape =
         tensor_product_factor_dof_shape_t< TensorRestriction, I >;
      constexpr size_t offset =
         tensor_product_local_dof_rank_offset_v< TensorRestriction, I >;
      const std::array< GlobalIndex, sizeof...( J ) > factor_indices{
         static_cast< GlobalIndex >( std::get< offset + J >(
            local_indices ) )... };
      const GlobalIndex local_dof_index =
         FlattenMultiIndex< DofShape >( factor_indices );

      return DirectIndexFactorTopologyDof(
         std::get< I >( restriction.restrictions ),
         element_indices[I],
         local_dof_index,
         TensorProductFactorLocalDofCount< TensorRestriction, I >() );
   }

   template <
      typename Tuple,
      size_t ... I >
   GENDIL_HOST_DEVICE
   GlobalIndex Offset(
      const Tuple & local_indices,
      const std::array<
         GlobalIndex,
         TensorRestriction::num_factors > & element_indices,
      std::index_sequence< I... > ) const
   {
      GlobalIndex global_index = 0;
      (
         ( global_index +=
              FactorTopologyDof< I >(
                 local_indices,
                 element_indices,
                 std::make_index_sequence<
                    tensor_product_factor_dof_rank_v<
                       TensorRestriction,
                       I > >{} ) * restriction.global_dof_strides[I] ),
         ... );
      return global_index;
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   GlobalIndex Offset( Indices... indices ) const
   {
      static_assert(
         sizeof...( Indices ) == rank,
         "TensorProductLayout received the wrong number of indices." );

      const auto local_indices =
         std::make_tuple( static_cast< GlobalIndex >( indices )... );
      const GlobalIndex element_index =
         std::get< rank - 1 >( local_indices );
      std::array<
         GlobalIndex,
         TensorRestriction::num_factors > element_indices{};
      TensorProductElementIndices(
         restriction,
         element_index,
         element_indices );

      return Offset(
         local_indices,
         element_indices,
         std::make_index_sequence< TensorRestriction::num_factors >{} );
   }
};

template < typename ProductDofShape, typename TensorRestriction >
GENDIL_HOST_DEVICE
auto MakeTensorProductLayout( const TensorRestriction & restriction )
{
   using Restriction = std::remove_cvref_t< TensorRestriction >;
   static_assert(
      is_tensor_product_restriction_v< Restriction >,
      "MakeTensorProductLayout requires a TensorProductRestriction." );
   static_assert(
      tensor_product_dof_rank_v< Restriction > ==
         DofShapeRank< ProductDofShape >::value,
      "TensorProductRestriction factor DoF ranks must match the product finite element rank." );

   return TensorProductLayout< Restriction >{ restriction };
}

} // namespace gendil

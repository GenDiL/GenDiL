// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <array>

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/get.hpp"

namespace gendil {

template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr GlobalIndex ComputeLinearIndex(
   const std::array< GlobalIndex, Dim > & indices,
   const std::array< GlobalIndex, Dim > & sizes );

template < Integer Dim, Integer Cpt >
GENDIL_HOST_DEVICE
constexpr GlobalIndex ComputeLinearIndex(
   const std::array< GlobalIndex, Dim > & indices,
   const std::array< GlobalIndex, Dim > & sizes,
   std::integral_constant< Integer, Cpt > )
{
   if constexpr ( Cpt < Dim - 1 )
   {
      return indices[Cpt] +
         sizes[Cpt] *
            ComputeLinearIndex(
               indices,
               sizes,
               std::integral_constant< Integer, Cpt + 1 >{} );
   }
   else
   {
      return indices[Cpt];
   }
}

template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr GlobalIndex ComputeLinearIndex(
   const std::array< GlobalIndex, Dim > & indices,
   const std::array< GlobalIndex, Dim > & sizes )
{
   return ComputeLinearIndex(
      indices,
      sizes,
      std::integral_constant< Integer, 0 >{} );
}

template < Integer CurrentIndex, typename Shape, Integer Dim >
GENDIL_HOST_DEVICE
constexpr GlobalIndex FlattenMultiIndex_impl(
   const std::array< GlobalIndex, Dim > & indices )
{
   if constexpr ( CurrentIndex == Dim - 1 )
   {
      return indices[CurrentIndex];
   }
   else
   {
      return indices[CurrentIndex] +
         seq_get_v< CurrentIndex, Shape > *
            FlattenMultiIndex_impl< CurrentIndex + 1, Shape >( indices );
   }
}

template < typename Shape, Integer Dim >
GENDIL_HOST_DEVICE
constexpr GlobalIndex FlattenMultiIndex(
   const std::array< GlobalIndex, Dim > & indices )
{
   static_assert(
      Dim > 0,
      "A local DoF multi-index must have positive rank." );
   static_assert(
      Dim == static_cast< Integer >( Shape::size() ),
      "Multi-index rank must match the static DoF shape rank." );
   return FlattenMultiIndex_impl< 0, Shape >( indices );
}

template < typename Shape, Integer Dim, size_t... I >
GENDIL_HOST_DEVICE
constexpr std::array< GlobalIndex, Dim > UnflattenMultiIndex_impl(
   GlobalIndex ordinal,
   std::index_sequence< I... > )
{
   std::array< GlobalIndex, Dim > index{};
   ( ( index[I] = ordinal % seq_get_v< I, Shape >,
       ordinal /= seq_get_v< I, Shape > ),
     ... );
   return index;
}

template < typename Shape >
GENDIL_HOST_DEVICE
constexpr auto UnflattenMultiIndex( const GlobalIndex ordinal )
{
   constexpr Integer Dim = static_cast< Integer >( Shape::size() );
   static_assert( Dim > 0, "A local DoF tensor index must have positive rank." );
   return UnflattenMultiIndex_impl< Shape, Dim >(
      ordinal,
      std::make_index_sequence< Dim >{} );
}

} // namespace gendil

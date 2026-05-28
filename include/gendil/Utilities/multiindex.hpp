// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

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
   return FlattenMultiIndex_impl< 0, Shape >( indices );
}

} // namespace gendil

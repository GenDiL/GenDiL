// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

template < Integer Dim >
GENDIL_HOST_DEVICE
GlobalIndex ComputeLinearIndex( const std::array< GlobalIndex, Dim > & cell_index, const std::array< GlobalIndex, Dim > & sizes );

template < Integer Dim, Integer Cpt >
GENDIL_HOST_DEVICE
GlobalIndex ComputeLinearIndex( const std::array< GlobalIndex, Dim > & cell_index, const std::array< GlobalIndex, Dim > & sizes, std::integral_constant< Integer, Cpt > )
{
   if constexpr ( Cpt < Dim -1 )
      return cell_index[Cpt] + sizes[Cpt] * ComputeLinearIndex( cell_index, sizes, std::integral_constant< Integer, Cpt+1>{} );
   else
      return cell_index[Cpt];
}

template < Integer Dim >
GENDIL_HOST_DEVICE
GlobalIndex ComputeLinearIndex( const std::array< GlobalIndex, Dim > & cell_index, const std::array< GlobalIndex, Dim > & sizes )
{
   return ComputeLinearIndex( cell_index, sizes, std::integral_constant< Integer, 0>{} );
}

}
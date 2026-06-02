// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <type_traits>

#include "gendil/Meshes/Connectivities/orientation.hpp"
#include "gendil/Utilities/multiindex.hpp"
#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/View/Layouts/orientedlayout.hpp"

namespace gendil {

struct FullSharedFaceReadDofsPolicy
{
};

struct DirectGlobalFaceReadDofsPolicy
{
};

template < typename KernelConfiguration, typename = void >
struct face_read_dofs_policy
{
   using type = FullSharedFaceReadDofsPolicy;
};

template < typename KernelConfiguration >
struct face_read_dofs_policy<
   KernelConfiguration,
   std::void_t< typename KernelConfiguration::face_read_dofs_policy > >
{
   using type = typename KernelConfiguration::face_read_dofs_policy;
};

template < typename KernelConfiguration >
using face_read_dofs_policy_t =
   typename face_read_dofs_policy<
      std::remove_cvref_t< KernelConfiguration > >::type;

/**
 * @brief FIFO/reference offset used by MakeFIFOView/MakeFixedFIFOView.
 *
 * @details The mixed-radix order is
 *
 *   offset = i0 + sizes[0] * i1
 *              + sizes[0] * sizes[1] * i2 + ...
 */
template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr GlobalIndex FaceReadDofsFIFOOffset(
   const std::array< GlobalIndex, Dim > & reference_indices,
   const std::array< size_t, Dim > & sizes )
{
   GlobalIndex offset = 0;
   GlobalIndex stride = 1;
   for ( Integer i = 0; i < Dim; ++i )
   {
      offset += reference_indices[ i ] * stride;
      stride *= static_cast< GlobalIndex >( sizes[ i ] );
   }
   return offset;
}

/**
 * @brief Direct inverse of the current FullShared face-read orientation map.
 *
 * @details Current FullShared face ReadDofs stores native element values into
 * shared memory with OrientedLayout and reads them back with FIFO/reference
 * layout:
 *
 *   shared[OrientedLayout(sizes, orientation).Offset(native)] =
 *      global(native..., element)
 *   local(reference) = shared[FIFOOffset(reference)]
 *
 * DirectGlobal must therefore compute the native index whose oriented-layout
 * offset equals the FIFO/reference offset.  For native dimension j,
 * abs(orientation(j)) - 1 is the oriented axis index; a negative sign reverses
 * that native dimension.
 *
 * The input reference_indices are in the original DofShape coordinate order,
 * not in oriented-axis coordinates.  For anisotropic swapped axes, such as
 * sizes (3,4) with orientation (2,-1), the original reference extents are
 * (3,4) while the oriented-axis radices are (4,3).  The tempting direct formula
 * native[j] = reference_indices[abs(orientation(j)) - 1] is therefore not
 * equivalent to the FullShared path in these cases.
 */
template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr std::array< GlobalIndex, Dim >
DirectGlobalFaceReadNativeIndex(
   const std::array< GlobalIndex, Dim > & reference_indices,
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   std::array< GlobalIndex, Dim > native_indices{};
   std::array< Integer, Dim > native_dim_for_oriented_axis{};
   std::array< bool, Dim > reversed_for_oriented_axis{};

   for ( Integer j = 0; j < Dim; ++j )
   {
      const LocalIndex o = orientation( j );
      const Integer oriented_axis =
         static_cast< Integer >( o > 0 ? o - 1 : -o - 1 );
      native_dim_for_oriented_axis[ oriented_axis ] = j;
      reversed_for_oriented_axis[ oriented_axis ] = o < 0;
   }

   GlobalIndex offset =
      FaceReadDofsFIFOOffset( reference_indices, sizes );

   for ( Integer oriented_axis = 0; oriented_axis < Dim; ++oriented_axis )
   {
      const Integer native_dim =
         native_dim_for_oriented_axis[ oriented_axis ];
      const bool reversed =
         reversed_for_oriented_axis[ oriented_axis ];
      const GlobalIndex radix =
         static_cast< GlobalIndex >( sizes[ native_dim ] );
      const GlobalIndex digit = offset % radix;
      offset /= radix;

      native_indices[ native_dim ] =
         reversed ? radix - 1 - digit : digit;
   }

   return native_indices;
}

template < typename View, Integer Dim, size_t... Is >
GENDIL_HOST_DEVICE
decltype(auto) FaceReadDofsGlobalValueAt(
   const View & global_dofs,
   const std::array< GlobalIndex, Dim > & native_indices,
   const GlobalIndex element_index,
   std::index_sequence< Is... > )
{
   return global_dofs( native_indices[ Is ]..., element_index );
}

template < typename View, Integer Dim >
GENDIL_HOST_DEVICE
decltype(auto) FaceReadDofsGlobalValueAt(
   const View & global_dofs,
   const std::array< GlobalIndex, Dim > & native_indices,
   const GlobalIndex element_index )
{
   return FaceReadDofsGlobalValueAt(
      global_dofs,
      native_indices,
      element_index,
      std::make_index_sequence< Dim >{} );
}

} // namespace gendil

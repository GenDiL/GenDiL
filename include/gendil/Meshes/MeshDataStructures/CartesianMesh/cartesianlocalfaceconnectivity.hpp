// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Local (intra-mesh) face connectivity on a Cartesian grid of HyperCubes.
 *
 * This header provides `CartesianLocalFaceConnectivity<Dim>` which, given a
 * cell's linear index and a compile-time local face id, returns a lightweight
 * face descriptor carrying:
 *  - the neighbor cell's linear index (or an invalid sentinel on boundaries),
 *  - the reference orientation (permutation) and the reference normal,
 *  - a boolean boundary flag.
 *
 * The implementation is specialized for `Dim==1` to keep hot paths minimal.
 *
 * @note All functions are header-only and marked @c GENDIL_HOST_DEVICE.
 */


#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Connectivities/computelinearindex.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"

namespace gendil {

/**
 * @brief Local face connectivity for a `Dim`-dimensional Cartesian mesh.
 *
 * For a cell identified by its linear index and a compile-time local face id
 * `FaceIndex ∈ [0, 2*Dim)`, this type computes the neighboring cell along that
 * face (if any), together with orientation and a unit reference normal.
 *
 * The face indexing convention follows axis-aligned HyperCubes:
 *  - Faces `0..Dim-1` have reference normal `-e_axis`,
 *  - Faces `Dim..2*Dim-1` have reference normal `+e_axis`,
 *  with `axis = FaceIndex % Dim`.
 *
 * @tparam Dim  Topological dimension (e.g., 2 or 3).
 *
 * @par Associated types
 *  - `using geometry         = HyperCube<Dim>;`
 *  - `using orientation_type = Permutation<Dim>;`
 *  - `using boundary_type    = bool;`
 *
 * @par Data members
 *  - `sizes[d]` : number of cells along dimension @p d (Cartesian layout).
 *
 * @warning The neighbor “invalid” sentinel uses NaN for `GlobalIndex`.
 *          If `GlobalIndex` is integral, provide an alternative sentinel.
 */
template < Integer Dim >
struct CartesianLocalFaceConnectivity
{
   using geometry = HyperCube< Dim >;
   using orientation_type = Permutation< Dim >;
   // Requires C++20
   // using orientation_type = std::integral_constant< Permutation<Dim>, MakeReferencePermutation< Dim >() >;
   using boundary_type = bool;

   std::array< GlobalIndex, Dim > sizes;

   /**
    * @brief Construct the connectivity with per-dimension grid sizes.
    *
    * @tparam Sizes  Any integer-like types convertible to `GlobalIndex`.
    * @param sizes   One size per dimension; length must be exactly @p Dim.
    *
    * @note Sizes are copied into `sizes` after cast to `GlobalIndex`.
    * @par Complexity
    * O(Dim).
    */
   template < typename ... Sizes >
   CartesianLocalFaceConnectivity( const Sizes & ... sizes ):
      sizes( { (GlobalIndex)sizes... } )
   {}

   /**
    * @brief Return local face info for a cell and compile-time face id.
    *
    * Decodes @p cell_index into multi-indices, determines the neighbor along the
    * face `FaceIndex` and encodes the neighbor back to a linear index. Boundary
    * conditions are detected on the first/last slices along the face axis.
    *
    * Face semantics (Hypercube convention):
    *  - `Index  = FaceIndex % Dim` selects the axis,
    *  - `Sign   = (FaceIndex < Dim ? -1 : +1)` selects the side (minus/plus).
    *
    * @tparam FaceIndex  Compile-time local face id in `[0, 2*Dim)`.
    * @param cell_index  Linear index of the current cell.
    * @return A `FaceConnectivity<...>` with:
    *   - neighbor linear index (or invalid sentinel if boundary),
    *   - empty user payload (`Empty{}`),
    *   - reference orientation (identity permutation),
    *   - boundary flag,
    *   - canonical unit normal `CanonicalVector<Dim, Index, Sign>`.
    *
    * @par Complexity
    * O(Dim) due to index decode/encode.
    *
    * @warning Uses `std::numeric_limits<GlobalIndex>::quiet_NaN()` as invalid sentinel.
    * @todo Consider stride-based neighbor computation to avoid full decode
    *       (fewer registers on GPU).
    */
   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetLocalFaceInfo( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > ) const
   {
      static_assert(
         FaceIndex < 2*Dim,
         "FaceIndex out of bound."
      );

      // !FIXME: This is magic and specific to HyperCube
      constexpr Integer Index = FaceIndex % Dim; // HyperCube< Dim >::GetNormalDimensionIndex( face_index ) ?
      constexpr int Sign = FaceIndex < Dim ? -1 : 1; // HyperCube< Dim >::GetNormalSign( face_index ) ?

      std::array< GlobalIndex, Dim > neighbor_index = GetStructuredSubIndices( cell_index, sizes );
      // TODO: we can forgo computing all of the indices (computing only
      // neighbor_index[Index] via GetStructuredSubIndex<Index>) and use strides
      // to compute the neighbor. (Might be better for GPU since it may use
      // fewer registers).

      bool boundary = false;

      if ( sizes[Index] == 1 )
      {
         neighbor_index[Index] = std::numeric_limits< GlobalIndex >::quiet_NaN();
         boundary = true;
      }
      else if ( neighbor_index[Index] == 0 )
      {
         if constexpr ( Sign == -1 )
         {
            neighbor_index[Index] = std::numeric_limits< GlobalIndex >::quiet_NaN();
            boundary = true;
         }
         else
         {
            neighbor_index[Index]++;
         }
      }
      else if ( neighbor_index[Index] == sizes[Index] - 1 )
      {
         if constexpr ( Sign == 1 )
         {
            neighbor_index[Index] = std::numeric_limits< GlobalIndex >::quiet_NaN();
            boundary = true;
         }
         else
         {
            neighbor_index[Index]--;
         }
      }
      else
      {
         if constexpr ( Sign == 1 )
            neighbor_index[Index]++;
         else
            neighbor_index[Index]--;
      }

      GlobalIndex neighbor_linear_index =
         boundary ?
            std::numeric_limits< GlobalIndex >::quiet_NaN() :
            ComputeLinearIndex( neighbor_index, sizes );

      using normal_type = CanonicalVector< Dim, Index, Sign >;
      using FaceInfo =
         FaceConnectivity<
            FaceIndex,
            geometry,
            Empty,
            orientation_type,
            boundary_type,
            normal_type
         >;
      return FaceInfo{ neighbor_linear_index, {}, MakeReferencePermutation< Dim >(), boundary };
      // Requires C++20
      // return FaceInfo{ { neighbor_index, neighbor_linear_index } };
   }

   /**
    * @brief Return the number of cells along a given dimension.
    * @param index Dimension id in `[0, Dim)`.
    * @return `sizes[index]`.
    * @par Complexity
    * O(1).
    */
   GENDIL_HOST_DEVICE
   GlobalIndex Size( GlobalIndex index ) const
   {
      return sizes[ index ];
   }

   /**
    * @brief Total number of cells in the Cartesian mesh.
    * @return `Product(sizes)` = ∏_{d=0}^{Dim-1} sizes[d].
    * @par Complexity
    * O(Dim).
    */
   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return Product( sizes );
   }
};

template < >
struct CartesianLocalFaceConnectivity< 1 >
{
   static constexpr Integer Dim = 1;
   using geometry = HyperCube< Dim >;
   using orientation_type = Permutation< Dim >;
   // Requires C++20
   // using orientation_type = std::integral_constant< Permutation<Dim>, MakeReferencePermutation< Dim >() >;
   using boundary_type = bool;

   GlobalIndex size;

   CartesianLocalFaceConnectivity( const Integer & size ):
      size( size )
   {}

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetLocalFaceInfo( GlobalIndex neighbor_index, std::integral_constant< Integer, FaceIndex > ) const
   {
      static_assert(
         FaceIndex < 2,
         "FaceIndex out of bound."
      );

      // !FIXME: This is magic and specific to HyperCube
      constexpr Integer Index = FaceIndex % Dim;
      constexpr int Sign = FaceIndex < Dim ? -1 : 1;

      bool boundary = false;

      if ( size == 1 )
      {
         neighbor_index = std::numeric_limits< GlobalIndex >::quiet_NaN();
         boundary = true;
      }
      else if ( neighbor_index == 0 )
      {
         if constexpr ( Sign == -1 )
         {
            neighbor_index = std::numeric_limits< GlobalIndex >::quiet_NaN();
            boundary = true;
         }
         else
         {
            neighbor_index++;
         }
      }
      else if ( neighbor_index == size - 1 )
      {
         if constexpr ( Sign == 1 )
         {
            neighbor_index = std::numeric_limits< GlobalIndex >::quiet_NaN();
            boundary = true;
         }
         else
         {
            neighbor_index--;
         }
      }
      else
      {
         if constexpr ( Sign == 1 )
            neighbor_index++;
         else
            neighbor_index--;
      }

      using normal_type = CanonicalVector< Dim, Index, Sign >;
      using FaceInfo =
         FaceConnectivity<
            FaceIndex,
            geometry,
            Empty,
            orientation_type,
            boundary_type,
            normal_type
         >;
      return FaceInfo{ neighbor_index, {}, MakeReferencePermutation< Dim >(), boundary };
      // Requires C++20
      // return FaceInfo{ neighbor_index };
   }

   GENDIL_HOST_DEVICE
   GlobalIndex Size( GlobalIndex index ) const
   {
      return size;
   }

   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return size;
   }
};

}
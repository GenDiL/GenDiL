// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Utilities/debug.hpp"
#include "gendil/Utilities/productdim.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"
#include "gendil/NumericalIntegration/integrationrule.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Meshes/Cells/ReferenceCells/productcell.hpp"

namespace gendil
{

namespace details
{
   /// @brief Gets the cell_type of the mesh
   template < typename Mesh >
   using mesh_cell_t = typename Mesh::cell_type;

   template < typename ... Meshes, size_t ... Is >
   GENDIL_HOST_DEVICE
   inline Integer ProductNumCells( std::tuple< Meshes ... > const & meshes, std::index_sequence<Is...> )
   {
      return Product( std::get< Is >( meshes ).GetNumberOfCells() ... );
   }

   /// @brief Computes the number of cells in a cartesian product mesh
   template < typename ... Meshes >
   GENDIL_HOST_DEVICE
   inline Integer ProductNumCells( std::tuple< Meshes ... > const & meshes )
   {
      return ProductNumCells( meshes, std::make_index_sequence< sizeof...( Meshes ) >() );
   }

   template < typename HeadMesh, typename ... TailMeshes, size_t ... Is >
   GENDIL_HOST_DEVICE
   inline auto GetTailMeshes( std::tuple< HeadMesh, TailMeshes ... > const & meshes, std::index_sequence< Is... > )
   {
      // return std::make_tuple( std::ref( std::get< Is+1 >( meshes ) ) ... );
      return std::tie( std::get< Is+1 >( meshes ) ... );
   }

   /// @brief returns tuple< TailMeshes const& ... > referencing the tail meshes in meshes.
   template < typename HeadMesh, typename ... TailMeshes >
   GENDIL_HOST_DEVICE
   inline auto GetTailMeshes( std::tuple< HeadMesh, TailMeshes ... > const & meshes )
   {
      return GetTailMeshes( meshes, std::make_index_sequence< sizeof...( TailMeshes ) >() );
   }

   /// @brief recursively computes the neighbor index for a product mesh
   template < Integer FaceIndex, typename MeshTuple >
   GENDIL_HOST_DEVICE
   auto ComputeNeighborIndex( MeshTuple const & meshes, Integer cell_index );

   /**
    * @brief Builds the product orientation when the neighboring face belongs
    * to the head mesh factor.
    *
    * The head factor contributes @p head_orientation. Every tail mesh factor
    * contributes a compile-time identity orientation, preserving the product
    * mesh factorization without constructing a flattened permutation.
    */
   template <
      typename HeadOrientation,
      typename HeadMesh,
      typename... TailMeshes >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   auto MakeProductOrientationFromHead(
      const std::tuple< HeadMesh, TailMeshes... > &,
      const HeadOrientation & head_orientation )
   {
      return MakeTensorProductOrientation(
         head_orientation,
         IdentityOrientation<
            std::remove_cvref_t< TailMeshes >::Dim >{}... );
   }

   /**
    * @brief Builds the product orientation from an already expanded tail
    * product orientation.
    *
    * The head factor contributes an identity orientation. The existing tail
    * components are appended individually so the result retains the product
    * mesh factorization rather than nesting the tail product.
    */
   template < class HeadMesh, class TailOrientation, size_t... I >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   auto MakeProductOrientationFromTailImpl(
      const TailOrientation & tail_orientation,
      std::index_sequence< I... > )
   {
      return MakeTensorProductOrientation(
         IdentityOrientation< std::remove_cvref_t< HeadMesh >::Dim >{},
         tail_orientation.template Get< I >()... );
   }

   /**
    * @brief Builds the product orientation when the neighboring face belongs
    * to a tail mesh factor.
    *
    * The head factor contributes an identity orientation and the tail
    * contributes @p tail_orientation. The head dimension and number of tail
    * factors are deduced from the complete mesh tuple. When the recursive tail
    * contains multiple mesh factors, an existing TensorProductOrientation is
    * expanded into its components to preserve the original mesh factorization.
    */
   template <
      class TailOrientation,
      class HeadMesh,
      class... TailMeshes >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   auto MakeProductOrientationFromTail(
      const std::tuple< HeadMesh, TailMeshes... > &,
      const TailOrientation & tail_orientation )
   {
      if constexpr (
         sizeof...( TailMeshes ) > 1 &&
         is_tensor_product_orientation_v< TailOrientation > )
      {
         return MakeProductOrientationFromTailImpl< HeadMesh >(
            tail_orientation,
            std::make_index_sequence<
               std::remove_cvref_t< TailOrientation >::num_components >{} );
      }
      else
      {
         return MakeTensorProductOrientation(
            IdentityOrientation< std::remove_cvref_t< HeadMesh >::Dim >{},
            tail_orientation );
      }
   }

   template < Integer sub_face_index, typename MeshTuple >
   GENDIL_HOST_DEVICE
   inline auto GetTailNeighbor( MeshTuple const & tail_meshes, Integer tail_index )
   {
      if constexpr ( std::tuple_size_v< MeshTuple > == 1 )
      {
         using face_index = std::integral_constant< Integer, sub_face_index >;
         return std::get< 0 >( tail_meshes ).GetLocalFaceInfo( tail_index , face_index{} );
      }
      else
      {
         return ComputeNeighborIndex< sub_face_index >( tail_meshes, tail_index );
      }
   }

   template < Integer FaceIndex, typename MeshTuple >
   GENDIL_HOST_DEVICE
   auto ComputeNeighborIndex( MeshTuple const & meshes, Integer cell_index )
   {
      constexpr Integer Dim = product_dim_v< MeshTuple >;

      using HeadMeshType = std::decay_t< std::tuple_element_t< 0, MeshTuple > >;
      constexpr Integer HeadDim = HeadMeshType::Dim;
      constexpr Integer TailDim = Dim - HeadDim;

      HeadMeshType const & head_mesh = std::get< 0 >( meshes );
      auto tail_meshes = GetTailMeshes( meshes );

      const Integer HeadNumCells = std::get< 0 >( meshes ).GetNumberOfCells();

      const Integer tail_index = cell_index / HeadNumCells;
      const Integer head_index = cell_index - tail_index * HeadNumCells;

      bool boundary;

      constexpr Integer Index =
         HyperCube< Dim >::GetNormalDimensionIndex( FaceIndex );
      constexpr int Sign =
         HyperCube< Dim >::GetNormalSign( FaceIndex );

      GlobalIndex neighbor_index;

      auto orientation = [&]
      {
         if constexpr ( Index < HeadDim )
         {
            // TODO: Use num_faces instead
            // !FIXME: This is implicitely using CanonicalVector...
            constexpr Integer sub_index = Index;
            constexpr Integer sub_face_index =
               (Sign == -1) ? sub_index : (HeadDim + sub_index);

            using face_index =
               std::integral_constant< Integer, sub_face_index >;
            auto neighbor =
               head_mesh.GetLocalFaceInfo( head_index, face_index{} );

            auto head_neighbor_index =
               neighbor.PlusSide().GetCellIndex();

            neighbor_index =
               head_neighbor_index + HeadNumCells * tail_index;
            boundary = neighbor.PlusSide().boundary;
            return MakeProductOrientationFromHead(
               meshes,
               neighbor.PlusSide().GetOrientation() );
         }
         else
         {
            // !FIXME: This is implicitely using CanonicalVector...
            constexpr Integer sub_index = Index - HeadDim;
            constexpr Integer sub_face_index =
               (Sign == -1) ? sub_index : (TailDim + sub_index);

            // recursion step
            auto neighbor =
               GetTailNeighbor< sub_face_index >( tail_meshes, tail_index );

            GlobalIndex tail_neighbor_index =
               neighbor.PlusSide().GetCellIndex();

            neighbor_index =
               head_index + HeadNumCells * tail_neighbor_index;
            boundary = neighbor.PlusSide().boundary;
            return MakeProductOrientationFromTail(
               meshes,
               neighbor.PlusSide().GetOrientation() );
         }
      }();

      constexpr Integer face_id = FaceIndex;
      using geometry = HyperCube< Dim >;
      using conformity_type = ConformingFaceMap< Dim >;
      using orientation_type = std::remove_cvref_t< decltype( orientation ) >;
      using boundary_type = bool;
      using normal_type = CanonicalVector< Dim, Index, Sign >;
      using face_info_type =
         ConformingCellFaceView <
            geometry,
            std::integral_constant< Integer, face_id >,
            std::integral_constant< Integer, HyperCube< Dim >::GetOppositeFaceIndex( face_id ) >,
            orientation_type,
            CanonicalVector< Dim, Index, Sign >,
            CanonicalVector< Dim, Index, -Sign >,
            boundary_type
         >;
      return face_info_type{
         { cell_index, {}, {}, {}, {}, boundary },
         { neighbor_index, {}, orientation, {}, {}, boundary }
      };
   }
}

template < typename... Meshes >
class CartesianProductMesh
{
private:
   using MeshTuple = typename std::tuple< Meshes ... >;
   MeshTuple SubMeshes;
   static constexpr Integer NumSubMeshes = sizeof...( Meshes );
   std::array< GlobalIndex, NumSubMeshes > Sizes;

public:
   template < Integer index >
   static constexpr Integer SubDim = std::tuple_element_t< index, MeshTuple >::Dim;
   static constexpr Integer Dim = product_dim_v< Meshes ... >;
   using cell_type = ProductCell< details::mesh_cell_t< Meshes > ... >;

   /**
    * @brief Read-only access to the component meshes.
    *
    * Mesh-domain compatibility uses this view to establish recursive indexed
    * topology identity for product meshes. The product mesh continues to own
    * the component values.
    */
   GENDIL_HOST_DEVICE
   const MeshTuple & GetSubMeshes() const
   {
      return SubMeshes;
   }

   CartesianProductMesh( Meshes const & ... meshes ) : 
      SubMeshes{ meshes ... }
   {
      ConstexprLoop< NumSubMeshes >(
         [&] ( auto i ) -> void
         {
               Sizes[ i ] = std::get< i >( SubMeshes ).GetNumberOfCells();
         }
      );
   }

   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return Product( Sizes );
   }

   GENDIL_HOST_DEVICE
   std::array< GlobalIndex, NumSubMeshes > GetStructuredSubIndices( GlobalIndex index ) const
   {
      return gendil::GetStructuredSubIndices( index, Sizes );
   }

   template < Integer dim >
   GENDIL_HOST_DEVICE
   GlobalIndex GetStructuredSubIndex( GlobalIndex index ) const
   {
      return gendil::GetStructuredSubIndex< dim >( index, Sizes );
   }

   GENDIL_HOST_DEVICE
   auto GetCell( GlobalIndex cell_index ) const
   {
      return MakeProductCell( SubMeshes, GetStructuredSubIndices( cell_index ) );
   }

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetLocalFaceInfo( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > ) const
   {
      return details::ComputeNeighborIndex< FaceIndex >( SubMeshes, cell_index );
   }
};

template < typename... Meshes >
inline auto MakeCartesianProductMesh( const Meshes & ... meshes )
{
   return CartesianProductMesh< Meshes ... >( meshes... );
}

} // namespace gendil

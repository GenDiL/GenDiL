// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/arraycat.hpp"
#include "gendil/Utilities/productdim.hpp"
#include "gendil/Utilities/Loop/loops.hpp"

namespace gendil {

namespace details
{
   template < typename CellTuple, size_t ... Is >
   GENDIL_HOST_DEVICE
   inline constexpr Integer StartDim( std::index_sequence< Is... > )
   {
      return Sum( std::tuple_element_t< Is, CellTuple >::Dim ... );
   }

   /// @brief computes the offset for the SubIntegrationRule associates with the N-th cell in CellTuple
   template < Integer N, typename CellTuple >
   GENDIL_HOST_DEVICE
   inline constexpr Integer StartDim()
   {
      if constexpr (N == 0)
      {
         return 0;
      }
      else
      {
         return StartDim< CellTuple >( std::make_index_sequence< N >() );
      }
   }

   /// @brief QuadData for a single cell
   template < Integer Offset, typename CellType, typename IntegrationRule >
   using SubQuadType = typename CellType::template QuadData< decltype( GetSubIntegrationRule< Offset, CellType::Dim >(IntegrationRule{}) ) >;

   template < typename ... >
   struct GetProductCellQuadData;

   template < typename IntegrationRule, typename CellTuple, size_t ... Is >
   struct GetProductCellQuadData< IntegrationRule, CellTuple, std::index_sequence< Is ... > >
   {
      using type = std::tuple< SubQuadType< StartDim< Is, CellTuple >(), std::tuple_element_t< Is, CellTuple >, IntegrationRule > ... >;
   };

   template < typename IntegrationRule, typename ... CellTypes >
   using ProductCellQuadData = typename GetProductCellQuadData< IntegrationRule, std::tuple< CellTypes ... >, std::make_index_sequence< sizeof...(CellTypes) > >::type;
   
} // namespace details

/**
 * @brief A structure representing a tensor product cell made of multiple cells.
 * 
 * @tparam CellTypes The type of cells composing the tensor cell.
 */

template < typename ... CellTypes >
class ProductCell
{
private:
   using CellTuple = std::tuple< CellTypes ... >;
   CellTuple Cells;

public:
   template < Integer Index >
   static constexpr Integer SubDim = std::tuple_element_t< Index, CellTuple >::Dim;
   
   static constexpr Integer Dim = product_dim_v< CellTypes ... >;

   using physical_coordinates = std::array< Real, Dim >;
   using jacobian = std::tuple< typename CellTypes::jacobian ... >;

   template < typename IntegrationRule >
   using QuadData = details::ProductCellQuadData< IntegrationRule, CellTypes ... >;

   GENDIL_HOST_DEVICE
   ProductCell( CellTypes const & ... cells ) : Cells( cells ... )
   { }

   template < typename QuadData >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( const TensorIndex< Dim > & quad_index,
                         const QuadData & quad_data,
                         physical_coordinates & X,
                         jacobian & J_mesh ) const
   {
      constexpr Integer NumSubCells = sizeof...( CellTypes );

      Integer head = 0;

      ConstexprLoop< NumSubCells >(
         [&] ( auto cell_index )
         {
            constexpr Integer sub_dim = SubDim< cell_index >;
            TensorIndex< sub_dim > index;
            for ( GlobalIndex i = 0; i < sub_dim; ++i )
            {
               index[ i ] = quad_index[ head + i ];
            }

            typename std::tuple_element_t< cell_index, CellTuple >::physical_coordinates SubX;
            std::get< cell_index >( Cells ).GetValuesAndJacobian( index, std::get< cell_index >( quad_data ), SubX, std::get< cell_index >( J_mesh ) );

            for ( GlobalIndex i = 0; i < sub_dim; ++i )
            {
               X[ head + i ] = SubX[ i ];
            }

            head += sub_dim;
         }
      );
   }
};

template < Integer Dim, typename ... Meshes, size_t ... Is >
GENDIL_HOST_DEVICE GENDIL_INLINE
auto MakeProductCell( std::tuple< Meshes ... > const & meshes, std::array< GlobalIndex, Dim > const & indices, std::index_sequence<Is...> )
{
   return ProductCell( std::get< Is >( meshes ).GetCell( indices[ Is ] ) ... );
}

template < Integer Dim, typename ... Meshes >
GENDIL_HOST_DEVICE GENDIL_INLINE
auto MakeProductCell( std::tuple< Meshes ... > const & meshes, std::array< GlobalIndex, Dim > const & indices )
{
   static_assert( Dim == sizeof...(Meshes) );
   return MakeProductCell( meshes, indices, std::make_index_sequence< Dim >() );
}

}

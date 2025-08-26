// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/IndexSequenceHelperFunctions/makeindexsequencewithout.hpp"
#include "gendil/FiniteElementMethod/finiteelementorders.hpp"

namespace gendil {

/**
 * @brief A simple structure representing a tensor product of a shape
 * function collection. The shape functions can have different basis and
 * different polynomial orders for each dimension.
 * 
 * @tparam ShapeFunctions The shape function collection.
 */
template < typename... ShapeFunctions >
struct TensorShapeFunctions
{
   using shape_functions_1d_tuple =  std::tuple< ShapeFunctions... >;
   using orders = FiniteElementOrders< (ShapeFunctions::num_dofs - 1)... >;
   using dof_shape = std::index_sequence< ShapeFunctions::num_dofs ... >;

   static constexpr Integer num_dofs = Product( ShapeFunctions::num_dofs... );
   static constexpr Integer dim = sizeof...( ShapeFunctions ); // Sum( ShapeFunctions::dim... );
   using index_type = TensorIndex< dim >;
   using ref_point_type = Point< dim >;
   using value_type = Real;
   using gradient_type = Point< dim >;

   /**
    * @brief Compute the value of a shape function at the given coordinate.
    * 
    * @param dof_index The index of the shape function.
    * @param ref_point The 1D coordinate at which to evaluate the shape function.
    * @return The value of the shape function at the given coordinate.
    */
   static constexpr value_type ComputeValue( const index_type dof_index, const ref_point_type & ref_point )
   {
      return ComputeValue( dof_index, ref_point, std::make_index_sequence< dim >{} );
   }

   template < size_t... Is >
   static constexpr value_type ComputeValue( const index_type dof_index, const ref_point_type & ref_point, std::index_sequence< Is... > )
   {
      return Prod( std::tuple_element_t< Is, shape_functions_1d_tuple >::ComputeValue( dof_index[Is], ref_point[Is] )... );
   }

   /**
    * @brief Compute the value of the gradient of a shape function at the given coordinate.
    * 
    * @param dof_index The index of the shape function.
    * @param ref_point The 1D coordinate at which to evaluate the shape functions.
    * @return The value of the gradient of the shape function at the given coordinate.
    */
   static constexpr gradient_type ComputeGradientValue( const index_type dof_index, const ref_point_type & ref_point )
   {
      return ComputeGradientValue( dof_index, ref_point, std::make_index_sequence< dim >{} );
   }

   template < size_t... Is >
   static constexpr gradient_type ComputeGradientValue( const index_type dof_index, const ref_point_type & ref_point, std::index_sequence< Is... > )
   {
      return { ComputeGradientValue( dof_index, ref_point, std::index_sequence< Is >{}, make_index_sequence_without< dim, Is >{} )... };
   }

   template < size_t I, size_t... Js >
   static constexpr Real ComputeGradientValue( const index_type dof_index, const ref_point_type & ref_point, std::index_sequence< I >, std::index_sequence< Js... > )
   {
      return std::tuple_element_t< I, shape_functions_1d_tuple >::ComputeGradientValue( dof_index[I], ref_point[I] ) * Prod( std::tuple_element_t< Js, shape_functions_1d_tuple >::ComputeValue( dof_index[Js], ref_point[Js] )... );
   }
};

template < typename ShapeFunctions >
struct is_tensor_shape_functions
{
   static constexpr bool value = false;
};

template < typename... ShapeFunctions >
struct is_tensor_shape_functions< TensorShapeFunctions< ShapeFunctions ... > >
{
   static constexpr bool value = true;
};

template < typename ShapeFunctions >
static constexpr bool is_tensor_shape_functions_v = is_tensor_shape_functions< ShapeFunctions >::value;

template < typename Orders, Integer index>
struct get_num_dofs_of_dim_v;

template < Integer index, Integer... orders > 
struct get_num_dofs_of_dim_v< FiniteElementOrders< orders... >, index >
{
   static constexpr Integer value = get_order< index, orders... >::value + 1;
};

// template < typename Orders, Integer index >
// constexpr Integer get_num_dofs_of_dim = get_num_dofs_of_dim_v< Orders, index >::value;

template < typename... ShapeFunctions >
GENDIL_HOST_DEVICE
constexpr auto GetDofsSizes( TensorShapeFunctions< ShapeFunctions... > )
{
   constexpr size_t dim = sizeof...(ShapeFunctions);
   return std::array<size_t,dim>( { GetDofsSizes( ShapeFunctions{} )... } );
}

template < size_t... DofShapes >
GENDIL_HOST_DEVICE
constexpr auto GetDofsSizes( std::index_sequence< DofShapes... > )
{
   constexpr size_t dim = sizeof...(DofShapes);
   return std::array<size_t,dim>{ DofShapes... };
}

}

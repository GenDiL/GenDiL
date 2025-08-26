// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil {

template < typename... ScalarShapeFunctions >
struct VectorShapeFunctions
{
   using scalar_shape_functions_tuple = std::tuple< ScalarShapeFunctions... >;

   static constexpr Integer num_dofs = Sum( ScalarShapeFunctions::num_dofs... );
   // TODO: Check that all scalar functions have the same dim?
   static constexpr Integer dim = std::tuple_element_t< 0, scalar_shape_functions_tuple >::dim;
   static constexpr Integer vector_dim = sizeof...( ScalarShapeFunctions ); // Sum( ShapeFunctions::dim... );
   // TODO: Check that all scalar shape functions have the same index type.
   using index_type = typename std::tuple_element_t< 0, scalar_shape_functions_tuple >::index_type;
   using ref_point_type = Point< dim >;
   using value_type = Real;
   using gradient_type = Point< dim >;
   using dof_shape = std::tuple< typename ScalarShapeFunctions::dof_shape... >;

   /**
    * @brief Compute the value of a shape function at the given coordinate.
    * 
    * @tparam v_dim The vectorial dimension index.
    * @param dof_index The index of the scalar shape function.
    * @param ref_point The 1D coordinate at which to evaluate the shape function.
    * @return The value of the shape function at the given coordinate.
    */
   template < Integer v_dim >
   static constexpr value_type ComputeValue( const index_type dof_index, const ref_point_type & ref_point )
   {
      return std::tuple_element_t< v_dim, scalar_shape_functions_tuple >::ComputeValue( dof_index, ref_point );
   }

   /**
    * @brief Compute the value of the gradient of a shape function at the given coordinate.
    * 
    * @tparam v_dim The vectorial dimension index.
    * @param dof_index The index of the shape function.
    * @param ref_point The 1D coordinate at which to evaluate the shape functions.
    * @return The value of the gradient of the shape function at the given coordinate.
    */
   template < Integer v_dim >
   static constexpr gradient_type ComputeGradientValue( const index_type dof_index, const ref_point_type & ref_point )
   {
      return std::tuple_element_t< v_dim, scalar_shape_functions_tuple >::ComputeGradientValue( dof_index, ref_point, std::make_index_sequence< dim >{} );
   }
};

template < typename ShapeFunctions >
struct is_vector_shape_functions
{
   static constexpr bool value = false;
};

template < typename... ShapeFunctions >
struct is_vector_shape_functions< VectorShapeFunctions< ShapeFunctions ... > >
{
   static constexpr bool value = true;
};

template < typename ShapeFunctions >
static constexpr bool is_vector_shape_functions_v = is_vector_shape_functions< ShapeFunctions >::value;

}

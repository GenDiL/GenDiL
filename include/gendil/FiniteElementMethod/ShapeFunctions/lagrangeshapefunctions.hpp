// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Meshes/Geometries/point.hpp"

namespace gendil {

/**
 * @brief Structure representing Lagrange polynomial shape functions.
 * 
 * @note Should only be used with 1D points.
 * 
 * @tparam Points The set of points describing the Lagrange nodes.
 */
template < typename Points >
struct LagrangeShapeFunctions
{
   using index_type = Integer;
   using ref_point_type = Point< 1 >;
   using value_type = Real;
   using gradient_type = Real;
   static constexpr Integer num_dofs = Points::GetNumPoints();
   static constexpr Integer dim = 1;
   using dof_shape = std::index_sequence< num_dofs >;

   /**
    * @brief Compute the value of a shape function at the given coordinate.
    * 
    * @param dof_index The index of the shape function.
    * @param ref_point The 1D coordinate at which to evaluate the shape function.
    * @return The value of the shape function at the given coordinate.
    */
   static constexpr value_type ComputeValue( const index_type dof_index, const ref_point_type & ref_point )
   {
      Real prod_numerator_res = 1.0;
      Real prod_denominator_res = 1.0;
      for ( Integer i = 0; i < num_dofs; i++ )
      {
         if ( i != dof_index )
         {
            // TODO: Use points instead of coord.
            prod_numerator_res *= ( ref_point - Points::GetCoord( i ) );
            prod_denominator_res *= ( Points::GetCoord( dof_index ) - Points::GetCoord( i ) );
         }
      }
      return prod_numerator_res / prod_denominator_res;
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
      Real sum_res = 0.0;
      for ( Integer i = 0; i < num_dofs; i++ )
      {
         if ( i != dof_index )
         {
            Real prod_numerator_res = 1.0;
            Real prod_denominator_res = ( Points::GetCoord( dof_index ) - Points::GetCoord( i ) );
            for ( Integer k = 0; k < num_dofs; k++ )
            {
               if ( (i != k) && (dof_index != k) )
               {
                  // TODO: Use points instead of coord.
                  prod_numerator_res *= ( ref_point - Points::GetCoord( k ) );
                  prod_denominator_res *= ( Points::GetCoord( dof_index ) - Points::GetCoord( k ) ) ;
               }
            }
            sum_res += ( prod_numerator_res / prod_denominator_res );
         }
      }
      return sum_res;
   }

   /**
    * @brief Compute the values of the shape functions at the given coordinate.
    * 
    * @param coord The 1D coordinate at which to evaluate the shape functions.
    * @param values The values of the shape functions at the given coordinate.
    */
   static constexpr void ComputeValues( const ref_point_type & ref_point, value_type (& values)[ num_dofs ] )
   {
      for ( Integer j = 0; j < num_dofs; j++ )
      {
         Real prod_numerator_res = 1.0;
         Real prod_denominator_res = 1.0;
         for ( Integer i = 0; i < num_dofs; i++ )
         {
            if ( i != j )
            {
               prod_numerator_res *= ( ref_point - Points::GetCoord( i ) );
               prod_denominator_res *= ( Points::GetCoord( j ) - Points::GetCoord( i ) );
            }
         }
         values[ j ] = prod_numerator_res / prod_denominator_res;
      }
   }

   /**
    * @brief Compute the values of the gradient of the shape functions at the given coordinate.
    * 
    * @param coord The 1D coordinate at which to evaluate the shape functions.
    * @param values The values of the gradient of the shape functions at the given coordinate.
    */
   static constexpr void ComputeGradientValues( const ref_point_type & ref_point, value_type (& values)[ num_dofs ] )
   {
      for ( Integer j = 0; j < num_dofs; j++ )
      {
         Real sum_res = 0.0;
         for ( Integer i = 0; i < num_dofs; i++ )
         {
            if ( i != j )
            {
               Real prod_numerator_res = 1.0;
               Real prod_denominator_res = ( Points::GetCoord( j ) - Points::GetCoord( i ) );
               for ( Integer k = 0; k < num_dofs; k++ )
               {
                  if ( (i != k) && (j != k) )
                  {
                     prod_numerator_res *= ( ref_point - Points::GetCoord( k ) );
                     prod_denominator_res *= ( Points::GetCoord( j ) - Points::GetCoord( k ) ) ;
                  }
               }
               sum_res += ( prod_numerator_res / prod_denominator_res );
            }
         }
         values[ j ] = sum_res;
      }
   }
};

}

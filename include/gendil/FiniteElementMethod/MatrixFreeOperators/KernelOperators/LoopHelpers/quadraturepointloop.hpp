// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/loops.hpp"

namespace gendil {

/**
 * @brief A loop abstraction allowing to loop on all the quadrature point indices.
 * 
 * @tparam IntegrationRule The integration rule type.
 * @tparam Lambda The lambda function type.
 * @param lambda The function to provide quadrature point indices.
 */
template < typename IntegrationRule, typename Lambda >
GENDIL_HOST_DEVICE
void QuadraturePointLoop( Lambda && lambda );

template < typename Lambda, Integer... NumPoints  >
GENDIL_HOST_DEVICE
void QuadraturePointLoop( std::integer_sequence< Integer, NumPoints...>, Lambda && lambda )
{
   Loop< NumPoints... >(
      [&](auto & ... indices)
      {
         constexpr Integer Dim = sizeof...(indices);
         TensorIndex< Dim > quad_index( indices... );
         // auto quad_index = std::make_tuple( indices... );
         lambda( quad_index );
      }
   );
}

template < typename IntegrationRule, typename Lambda >
GENDIL_HOST_DEVICE
void QuadraturePointLoop( Lambda && lambda )
{
   // Assumes tensor integration rule.
   using num_points_tensor = typename IntegrationRule::points::num_points_tensor;
   QuadraturePointLoop( num_points_tensor{}, lambda );
}

template < typename IntegrationRule, typename KernelContext, typename Lambda >
GENDIL_HOST_DEVICE
inline void ThreadedQuadraturePointLoop( const KernelContext & thread, Lambda && lambda )
{
   using quad_shape = typename IntegrationRule::points::num_points_tensor;
   using tshape = subsequence_t< quad_shape, typename KernelContext::template threaded_dimensions< quad_shape::size() > >;
   using rshape = subsequence_t< quad_shape, typename KernelContext::template register_dimensions< quad_shape::size() > >;

   constexpr Integer Dim = quad_shape::size();

   // !FIXME Assumes the first dimensions are threaded
   ThreadLoop< tshape >( thread, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... i )
      {
         TensorIndex< Dim > quad_index(t..., i...); // TODO replace with tuple?
         lambda( quad_index );
      });
   });
   // ThreadLoop< quad_shape >(
   //    thread,
   //    [&] ( auto... indices )
   //    {
   //       constexpr Integer Dim = sizeof...(indices);
   //       TensorIndex< Dim > quad_index( indices... );
   //       lambda( quad_index );
   //    }
   // );
}

template < typename IntegrationRule, typename KernelContext, typename Lambda >
GENDIL_HOST_DEVICE
inline void QuadraturePointLoop( const KernelContext & thread, Lambda && lambda )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      return QuadraturePointLoop< IntegrationRule >( lambda );
   }
   else
   {
      return ThreadedQuadraturePointLoop< IntegrationRule >( thread, lambda );
   }
}

template < typename KernelContext, typename IntegrationRule, typename Lambda >
GENDIL_HOST_DEVICE
inline void QuadraturePointLoop( const KernelContext & thread, const IntegrationRule & integration_rule, Lambda && lambda )
{
   return QuadraturePointLoop<IntegrationRule>( thread, lambda );
}

}

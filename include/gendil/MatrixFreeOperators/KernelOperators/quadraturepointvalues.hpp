// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/RecursiveArray/instantiatearray.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Utilities/MathHelperFunctions/min.hpp"
#include "elementdof.hpp"

namespace gendil {

/**
 * @brief A helper structure that provides a container type for quadrature point data.
 * 
 * @tparam IntegrationRule The integration rule associated to the quadrature data.
 * @tparam extra_dims Extra dimensions per quadrature point.
 */
template < typename IntegrationRule, Integer... extra_dims >
struct get_quad_tensor_type_t
{
   using points = typename IntegrationRule::points;
   using Orders = typename points::num_points_tensor;
   using type = typename instantiate_array< Orders, extra_dims... >::type;
};

template < typename IntegrationRule, Integer... extra_dims >
using get_quad_tensor_type = typename get_quad_tensor_type_t< IntegrationRule, extra_dims... >::type;

/**
 * @brief A helper structure to store data at quadrature points, behaves like a multi-dimension array.
 * 
 * @tparam IntegrationRule The integration rule associated with the quadrature data.
 * @tparam extra_dims Extra dimensions per quadrature point.
 */
template < typename IntegrationRule, Integer... extra_dims >
struct QuadraturePointValues
{
   using Data = get_quad_tensor_type< IntegrationRule, extra_dims... >;
   static constexpr Integer Dim = Data::Dim;
   Data data;

   template < typename... Args >
   GENDIL_HOST_DEVICE
   Real & operator()( Args... args )
   {
      return data( args... );
   }

   template < typename... Args >
   GENDIL_HOST_DEVICE
   const Real & operator()( Args... args ) const
   {
      return data( args... );
   }

   template < typename FESpace >
   GENDIL_HOST_DEVICE
   explicit operator ElementDoF< FESpace >() const
   {
      // TODO: Add static check that the basis functions have the same points as the integration rule?
      return ElementDoF< FESpace >{ data };
   }
};

template < Integer Index, typename IntegrationRule, size_t ... extra_dims >
struct GetTensorSize< Index, QuadraturePointValues< IntegrationRule, extra_dims ... > > : GetTensorSize< Index, typename QuadraturePointValues< IntegrationRule, extra_dims ... >::Data > {};

template <
   size_t ... Dims,
   typename KernelContext,
   typename IntegrationRule >
GENDIL_HOST_DEVICE
auto MakeQuadraturePointValuesContainer( const KernelContext & kernel_conf, IntegrationRule )
{
   using quad_shape = typename IntegrationRule::points::num_points_tensor;
   using rdims = typename KernelContext::template register_dimensions< IntegrationRule::space_dim >;
   using rshape = subsequence_t< quad_shape, rdims >;
   using shape = cat_t< rshape, std::index_sequence< Dims... > >;
   return MakeStaticFIFOView< Real >( shape{} );
}

template <
   size_t ... Dims,
   typename KernelContext,
   typename IntegrationRule >
GENDIL_HOST_DEVICE
auto MakeSharedQuadraturePointValuesContainer( const KernelContext & kernel_conf, IntegrationRule )
{
   using quad_shape = typename IntegrationRule::points::num_points_tensor;
   using shape = cat_t< quad_shape, std::index_sequence< Dims... > >;
   constexpr size_t shared_size = Product( shape{} );
   Real * buffer = kernel_conf.SharedAllocator.allocate( shared_size );
   return MakeFixedFIFOView( buffer, shape{} );
}

}

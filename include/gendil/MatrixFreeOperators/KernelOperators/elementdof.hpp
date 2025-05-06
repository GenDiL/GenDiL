// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/RecursiveArray/instantiatearray.hpp"
#include "LoopHelpers/dofloop.hpp"

namespace gendil {

/**
 * @brief A helper structure that provides a container type for degrees of freedom.
 * 
 * @tparam FiniteElementSpace The finite element space associated to the degrees of freedom.
 */
template < typename FiniteElementSpace > // ?FIXME: should it take a finite element instead?
struct get_element_array_type_t
{
   // This assumes tensor shape functions
   using Orders = typename FiniteElementSpace::finite_element_type::shape_functions::orders;
   using type = typename instantiate_array< Orders >::type;
};

template < typename FiniteElementSpace >
using get_element_array_type = typename get_element_array_type_t< FiniteElementSpace >::type;

/**
 * @brief A helper structure to store degrees-of-freedom, behaves like a multi-dimension array.
 * 
 * @tparam FiniteElementSpace The finite element space associated to the degrees of freedom.
 */
template < typename FiniteElementSpace >
struct ElementDoF
{
   using element_type = Real;
   using Data = get_element_array_type< FiniteElementSpace >;
   static constexpr Integer Dim = Data::rank;
   Data data;

   template < typename... Args >
   GENDIL_HOST_DEVICE
   Real & operator()( Args... args )
   {
      return data( std::forward< Args >( args )... );
   }

   template < typename... Args >
   GENDIL_HOST_DEVICE
   const Real & operator()( Args... args ) const
   {
      return data( std::forward< Args >( args )... );
   }

   GENDIL_HOST_DEVICE
   ElementDoF& operator=( Real val )
   {
      DofLoop< FiniteElementSpace >(
         [&]( auto... indices )
         {
            data( indices... ) = val;
         }
      );
      return *this;
   }

   GENDIL_HOST_DEVICE
   ElementDoF& operator=( const ElementDoF & other )
   {
      DofLoop< FiniteElementSpace >(
         [&]( auto... indices )
         {
            data( indices... ) = other( indices... );
         }
      );
      return *this;
   }

   // TODO: Make it safer
   template < typename Tensor >
   GENDIL_HOST_DEVICE
   ElementDoF& operator=( const Tensor & other )
   {
      DofLoop< FiniteElementSpace >(
         [&]( auto... indices )
         {
            data( indices... ) = other( indices... );
         }
      );
      return *this;
   }
};

template <
   size_t ... Dims,
   typename KernelContext,
   typename FiniteElementSpace >
GENDIL_HOST_DEVICE
auto MakeElementDoFValuesContainer( const KernelContext & kernel_conf, FiniteElementSpace )
{
   using orders = typename FiniteElementSpace::finite_element_type::shape_functions::orders;
   using dof_shape = orders_to_num_dofs< orders >;
   using rdims = typename KernelContext::template register_dimensions< FiniteElementSpace::finite_element_type::space_dim >;
   using rshape = subsequence_t< dof_shape, rdims >;
   using shape = cat_t< rshape, std::index_sequence< Dims... > >;
   return MakeSerialRecursiveArray< Real >( shape{} );
}

template <
   size_t ... Dims,
   typename KernelContext,
   typename FiniteElementSpace >
GENDIL_HOST_DEVICE
auto MakeSharedElementDoFValuesContainer( const KernelContext & kernel_conf, FiniteElementSpace )
{
   using Orders = typename FiniteElementSpace::finite_element_type::shape_functions::orders;
   using dof_shape = orders_to_num_dofs< Orders >;
   using shape = cat_t< dof_shape, std::index_sequence< Dims... > >;
   constexpr size_t shared_size = Product( shape{} );
   Real * buffer = kernel_conf.SharedAllocator.allocate( shared_size );
   return MakeFixedFIFOView( buffer, shape{} );
}

template < typename FiniteElementSpace >
GENDIL_HOST_DEVICE
ElementDoF< FiniteElementSpace >& operator+=( ElementDoF< FiniteElementSpace >& lhs_dofs, const ElementDoF< FiniteElementSpace >& rhs_dofs )
{
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         lhs_dofs( indices... ) += rhs_dofs( indices... );
      }
   );
   return lhs_dofs;
}

template < typename FiniteElementSpace >
GENDIL_HOST_DEVICE
ElementDoF< FiniteElementSpace >& operator-=( ElementDoF< FiniteElementSpace >& lhs_dofs, const ElementDoF< FiniteElementSpace >& rhs_dofs )
{
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         lhs_dofs( indices... ) -= rhs_dofs( indices... );
      }
   );
   return lhs_dofs;
}

template < typename FiniteElementSpace >
GENDIL_HOST_DEVICE
ElementDoF< FiniteElementSpace > operator+( ElementDoF< FiniteElementSpace >& lhs_dofs, const ElementDoF< FiniteElementSpace >& rhs_dofs )
{
   ElementDoF< FiniteElementSpace > res;
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         res( indices... ) = lhs_dofs( indices... ) + rhs_dofs( indices... );
      }
   );
   return res;
}

template < typename FiniteElementSpace >
GENDIL_HOST_DEVICE
ElementDoF< FiniteElementSpace > operator-( ElementDoF< FiniteElementSpace >& lhs_dofs, const ElementDoF< FiniteElementSpace >& rhs_dofs )
{
   ElementDoF< FiniteElementSpace > res;
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         res( indices... ) = lhs_dofs( indices... ) - rhs_dofs( indices... );
      }
   );
   return res;
}

template < typename FiniteElementSpace >
GENDIL_HOST_DEVICE
ElementDoF< FiniteElementSpace > operator*( Real alpha, const ElementDoF< FiniteElementSpace >& dofs )
{
   ElementDoF< FiniteElementSpace > res{};
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         res( indices... ) = alpha * dofs( indices... );
      }
   );
   return res;
}

template < Integer Index, typename FiniteElementSpace >
struct GetTensorSize< Index, ElementDoF< FiniteElementSpace > > : GetTensorSize< Index, typename ElementDoF< FiniteElementSpace >::Data > {};

}

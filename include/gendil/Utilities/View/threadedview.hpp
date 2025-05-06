// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "Containers/containers.hpp"

namespace gendil {

// TODO: Generalize? < typename ElementType, typename Extents, typename Layout, typename Accessor >
template < typename Sizes, typename KernelContext, typename Container >
struct ThreadedView
{
   using element_type = typename Container::element_type;

   using sizes = Sizes;
   using thread_shape = subsequence_t< sizes, typename KernelContext::template threaded_dimensions< Sizes::size() > >;
   using register_shape = subsequence_t< sizes, typename KernelContext::template register_dimensions< Sizes::size() > >;
   static constexpr size_t rank = thread_shape::size() + register_shape::size();

   // const KernelContext & context;
   Container data;

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   auto & operator()( Indices... idx ) const
   {
      if constexpr ( sizeof...(idx) == Container::rank )
      {
         return data( idx... );
      }
      else
      {
         static_assert(
            sizeof...(idx) == rank,
            "Wrong number of arguments."
         );
         static_assert(
            register_shape::size() == Container::rank,
            "Threaded view container must have the same rank as register shape."
         );
         auto register_indices = get( register_shape{}, idx... );
         return std::apply(data, register_indices );
      }
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   auto & operator()( Indices... idx )
   {
      if constexpr ( sizeof...(idx) == Container::rank )
      {
         return data( idx... );
      }
      else
      {
         static_assert(
            sizeof...(idx) == rank,
            "Wrong number of arguments."
         );
         static_assert(
            register_shape::size() == Container::rank,
            "Threaded view container must have the same rank as register shape."
         );
         auto register_indices = get( register_shape{}, idx... );
         return std::apply(data, register_indices );
      }
   }

   GENDIL_HOST_DEVICE
   ThreadedView & operator=( const element_type & a )
   {
      data = a;
      return *this;
   }

   GENDIL_HOST_DEVICE
   ThreadedView & operator=( const Container & a )
   {
      data = a;
      return *this;
   }
};

template < typename FiniteElementSpace >
struct ElementDoF;

template < typename KernelContext, typename FiniteElementSpace >
GENDIL_HOST_DEVICE
auto MakeThreadedView( const KernelContext & kernel_conf, const ElementDoF<FiniteElementSpace> & data )
{
   using DofShape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   using Container = get_element_array_type< FiniteElementSpace >;
   // return ThreadedView< DofShape, KernelContext, Container >{ kernel_conf, data.data };
   return ThreadedView< DofShape, KernelContext, Container >{ data.data };
}

template < typename KernelContext, typename FiniteElementSpace, typename Container >
GENDIL_HOST_DEVICE
auto MakeThreadedView( const KernelContext & kernel_conf, const FiniteElementSpace & fe_space , const Container & data )
{
   using DofShape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   return ThreadedView< DofShape, KernelContext, Container >{ data };
}

template < size_t... dims, typename KernelContext, typename Container >
GENDIL_HOST_DEVICE
auto MakeThreadedView( const KernelContext & kernel_conf, const Container & data )
{
   // return ThreadedView< std::index_sequence< dims... >, KernelContext, Container >{ kernel_conf, data };
   return ThreadedView< std::index_sequence< dims... >, KernelContext, Container >{ data };
}

template < typename Sizes, typename KernelContext, typename Container >
GENDIL_HOST_DEVICE
auto MakeThreadedView( const KernelContext & kernel_conf, const Container & data )
{
   return ThreadedView< Sizes, KernelContext, Container >{ kernel_conf, data };
}

template < typename Sizes, typename KernelContext, typename Container >
GENDIL_HOST_DEVICE
ThreadedView< Sizes, KernelContext, Container > operator+(
   const ThreadedView< Sizes, KernelContext, Container > & x,
   const ThreadedView< Sizes, KernelContext, Container > & y )
{
   ThreadedView< Sizes, KernelContext, Container > res;
   res.data = x.data + y.data;
   return res;
}

template < typename Sizes, typename KernelContext, typename Container >
GENDIL_HOST_DEVICE
ThreadedView< Sizes, KernelContext, Container > operator-(
   const ThreadedView< Sizes, KernelContext, Container > & x,
   const ThreadedView< Sizes, KernelContext, Container > & y )
{
   ThreadedView< Sizes, KernelContext, Container > res;
   res.data = x.data - y.data;
   return res;
}

template < typename T, typename Sizes, typename KernelContext, typename Container >
GENDIL_HOST_DEVICE
ThreadedView< Sizes, KernelContext, Container > operator*(
   const T & a,
   const ThreadedView< Sizes, KernelContext, Container > & x )
{
   ThreadedView< Sizes, KernelContext, Container > res;
   res.data = a * x.data;
   return res;
}

template < typename Sizes, typename KernelContext, typename Container >
GENDIL_HOST_DEVICE
ThreadedView< Sizes, KernelContext, Container >& operator+=(
   const ThreadedView< Sizes, KernelContext, Container > & x,
   const ThreadedView< Sizes, KernelContext, Container > & y )
{
   x.data += y.data;
   return x;
}

template < typename Sizes, typename KernelContext, typename Container >
GENDIL_HOST_DEVICE
ThreadedView< Sizes, KernelContext, Container >& operator-=(
   const ThreadedView< Sizes, KernelContext, Container > & x,
   const ThreadedView< Sizes, KernelContext, Container > & y )
{
   x.data -= y.data;
   return x;
}

}
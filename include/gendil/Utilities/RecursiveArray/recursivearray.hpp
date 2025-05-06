// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <array>
#include <memory>

#include "gendil/Utilities/gettensorsize.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/indexsequencehelperfunctions.hpp"
#include "gendil/Utilities/Loop/unitloop.hpp"

namespace gendil {

/**
 * @brief A generic recursive array to represent multi-dimension arrays.
 * 
 * @tparam T The type of the elements contained in the array.
 * @tparam DimensionInfo Types describing the dimension types.
 */
template <typename T, typename... DimensionInfo >
struct RecursiveArray;

/**
 * @brief Specialization of a recursive array of rank 0, the structure automatically casts to its element's type.
 * 
 * @tparam T The element type.
 */
template <typename T >
struct RecursiveArray< T >
{
   using element_type = T;
   static constexpr Integer rank = 0;
   T data;

   GENDIL_HOST_DEVICE
   operator T()
   {
      return data;
   }

   GENDIL_HOST_DEVICE
   const T & operator()() const
   {
      return data;
   }

   GENDIL_HOST_DEVICE
   T & operator()()
   {
      return data;
   }

   GENDIL_HOST_DEVICE
   RecursiveArray< T >& operator=( const T & val )
   {
      data = val;
      return *this;
   }
};

/**
 * @brief A simple structure describing a dimension with statically known size.
 * 
 * @tparam Size The size of the dimension.
 */
template < size_t Size >
struct StaticDimension
{
   static constexpr bool is_static_pointer = true;
   static constexpr size_t size = Size;

   template < typename Lambda >
   GENDIL_HOST_DEVICE
   static void loop( Lambda && f )
   {
      for (size_t i = 0; i < Size; i++)
      {
         f( i );
      }      
   }
};

template < typename Lambda, typename ... Args >
GENDIL_HOST_DEVICE
void RecursiveArrayLoop( Lambda && f, Args... args )
{
   f( args... );
}

template < typename FirstDim, typename ... RestDims, typename Lambda, typename ... Args >
GENDIL_HOST_DEVICE
void RecursiveArrayLoop( Lambda && f, Args... args )
{
   FirstDim::loop([&](auto i)
   {
      RecursiveArrayLoop< RestDims... >( f , args..., i );
   });
}

/**
 * @brief Specialization of the RecursiveArray with a static dimension.
 * 
 * @tparam T The type of the elements.
 * @tparam Size The static size of the first dimension.
 * @tparam DimensionInfo The rest of the dimension informations.
 */
template< typename T, size_t Size, typename... DimensionInfo >
struct RecursiveArray< T, StaticDimension< Size >, DimensionInfo... >
{
   using element_type = T;
   using Data = RecursiveArray< T, DimensionInfo ... >;
   static constexpr Integer rank = 1 + Data::rank;
   // std::array< RecursiveArray< T, DimensionInfo... >, Size > data;
   Data data[ Size ];

   // TODO: safe accessor `at` with bound check?

   template < typename... Args >
   GENDIL_HOST_DEVICE
   const auto & operator()( size_t first_index, Args... args ) const
   {
      return data[ first_index ]( args... );
   }

   template < typename... Args >
   GENDIL_HOST_DEVICE
   auto & operator()( size_t first_index, Args... args )
   {
      return data[ first_index ]( args... );
   }

   GENDIL_HOST_DEVICE
   RecursiveArray & operator=( T const & a )
   {
      for (size_t i = 0; i < Size; i++) this->data[i] = a;
      return *this;
   }

   // GENDIL_HOST_DEVICE
   // RecursiveArray & operator=( RecursiveArray const & other )
   // {
   //    for (size_t i = 0; i < Size; i++) this->data[i] = other.data[i];
   //    return *this;
   // }
};

template < typename DimensionInfo >
struct is_static_dimension
{
   static constexpr bool value = DimensionInfo::is_static_pointer;
};

// Dynamic dimension
struct DynamicDimension
{
   static constexpr bool is_static_pointer = false;
   size_t size;
};

template < Integer NumThreads, Integer Size >
struct ThreadedDimension
{
   static constexpr Integer num_threads = NumThreads;
   static constexpr Integer size = Size;
};

template< typename T,
          Integer NumThreads,
          Integer Size,
          typename... TailDimensionInfo >
struct RecursiveArray< T, ThreadedDimension< NumThreads, Size >, TailDimensionInfo... >
{
   using Data = RecursiveArray< T, TailDimensionInfo... >;
   static constexpr Integer Dim = 1 + Data::Dim;
   Data data;

   // TODO: safe accessor `at` with bound check?
   // TODO: threaded specific safety checks

   // Note: Ignores first_index associated to the thread dimension, first_index is
   // Always assumed to be equal to the thread index.
   template < typename... Args >
   GENDIL_HOST_DEVICE
   const auto & operator()( size_t first_index, Args... args ) const
   {
      return data( args... );
   }

   template < typename... Args >
   GENDIL_HOST_DEVICE
   auto & operator()( size_t first_index, Args... args )
   {
      return data( args... );
   }
};

// Pointer dimension
template < typename DimensionInfo >
struct PointerDimension : public DimensionInfo
{
   static constexpr bool is_pointer = true;
};

template< typename T, typename DimensionInfo, typename... TailDimensionInfo >
struct RecursiveArray< T, PointerDimension< DimensionInfo >, TailDimensionInfo... >
{
   using Data = RecursiveArray< T, TailDimensionInfo... >;

   static constexpr Integer Dim = 1 + Data::Dim;

   size_t size;
   Data * data;

   // TODO: safe accessor `at` with bound check?

   template < typename... Args >
   GENDIL_HOST_DEVICE
   const auto & operator()( size_t first_index, Args... args ) const
   {
      return data[first_index]( args... );
   }

   template < typename... Args >
   GENDIL_HOST_DEVICE
   auto & operator()( size_t first_index, Args... args )
   {
      return data[first_index]( args... );
   }
};

// Reference dimension
struct ReferenceDimension;

template< typename T, typename... DimensionInfo >
struct RecursiveArray< T, ReferenceDimension, DimensionInfo... >
{
   size_t size;
   RecursiveArray< T, DimensionInfo... > & data;

   // TODO: safe accessor `at` with bound check?

   template < typename... Args >
   GENDIL_HOST_DEVICE
   const auto & operator()( size_t first_index, Args... args ) const
   {
      return data( args... );
   }

   template < typename... Args >
   GENDIL_HOST_DEVICE
   auto & operator()( size_t first_index, Args... args )
   {
      return data( args... );
   }
};

// Allocated dimension
template < typename DimensionInfo, template < typename T > typename Allocator = std::allocator >
struct AllocatedDimension : public DimensionInfo
{
   template < typename T >
   using memory_type = typename Allocator<T>::value_type;

   template < typename T >
   GENDIL_HOST_DEVICE
   static auto allocate( size_t size )
   {
      return Allocator<T>::allocate( size );
   }
};

template< typename T,
          typename DimensionInfo,
          template < typename > typename Allocator,
          typename... TailDimensionInfo >
struct RecursiveArray< T,
                       AllocatedDimension< DimensionInfo, Allocator >,
                       TailDimensionInfo... >
{
   static constexpr bool is_allocated = true;

   using memory_type = typename Allocator<T>::value_type; // FIXME: Is this correct?
   DimensionInfo dim_info;
   memory_type data;

   // FIXME: Not exactly that
   template < std::enable_if_t< is_static_dimension< DimensionInfo >::value, bool > = true >
   GENDIL_HOST_DEVICE
   RecursiveArray() : data( Allocator<T>::allocate( DimensionInfo::size ) )
   { }

   template < std::enable_if_t< !is_static_dimension< DimensionInfo >::value, bool > = true >
   GENDIL_HOST_DEVICE
   RecursiveArray( size_t size ) : data( Allocator<T>::allocate( size ) )
   { }
};

template< typename T,
          typename FirstDimensionInfo,
          typename DimensionInfo,
          template < typename > typename Allocator,
          typename... TailDimensionInfo >
struct RecursiveArray< T,
                       FirstDimensionInfo,
                       AllocatedDimension< DimensionInfo, Allocator >,
                       TailDimensionInfo... >
{
   GENDIL_HOST_DEVICE
   RecursiveArray()
   {
      static_assert(
         std::is_same<T, void>::value,
         "AllocatedDimension must be the first dimension."
      );
   }
};

// SafeDimension
struct SafeDimension;


// Should this be a virtual base containing all th default trait values?
struct Dimension
{
   static constexpr bool is_pointer = false;
   static constexpr bool is_reference = false;
   static constexpr bool is_allocated = false;
};

// template < typename Size, typename... MixedIns >
// struct Dimension : public Size, public MixedIns...
// {
//    // static constexpr is_pointer = false;
// };

/**
 * @brief A helper type to create RecursiveArrays where all the dimensions are static dimensions.
 * 
 * @tparam T The type of the elements.
 * @tparam Sizes The static dimensions of the RecursiveArray.
 */
template < typename T, size_t... Sizes >
using SerialRecursiveArray = RecursiveArray< T, StaticDimension< Sizes >... >;

template < Integer Index, typename T, size_t ... Sizes >
struct GetTensorSize< Index, SerialRecursiveArray< T, Sizes ... > >
{
   static constexpr size_t value = vseq_get_v< Index, Sizes... >;
};

template< typename T, size_t... Dims, typename Other >
GENDIL_HOST_DEVICE
SerialRecursiveArray< T, Dims... > & operator+=( SerialRecursiveArray< T, Dims... > & x, Other const & y )
{
   UnitLoop< Dims... >( [&]( auto... indices )
   {
      x( indices... ) += y( indices... );
   });
   return x;
}

template< typename T, size_t... Dims >
GENDIL_HOST_DEVICE
SerialRecursiveArray< T, Dims... > operator+(
   const SerialRecursiveArray< T, Dims... > & x,
   const SerialRecursiveArray< T, Dims... > & y )
{
   SerialRecursiveArray< T, Dims... > res;
   UnitLoop< Dims... >( [&]( auto... indices )
   {
      res( indices... ) = x( indices... ) + y( indices... );
   });
   return res;
}

template< typename T, size_t... Dims, typename Other >
GENDIL_HOST_DEVICE
SerialRecursiveArray< T, Dims... > & operator-=( SerialRecursiveArray< T, Dims... > & x, Other const & y )
{
   UnitLoop< Dims... >( [&]( auto... indices )
   {
      x( indices... ) -= y( indices... );
   });
   return x;
}

template< typename T, size_t... Dims >
GENDIL_HOST_DEVICE
SerialRecursiveArray< T, Dims... > operator-(
   const SerialRecursiveArray< T, Dims... > & x,
   const SerialRecursiveArray< T, Dims... > & y )
{
   SerialRecursiveArray< T, Dims... > res;
   UnitLoop< Dims... >( [&]( auto... indices )
   {
      res( indices... ) = x( indices... ) - y( indices... );
   });
   return res;
}

template< typename T, size_t... Dims >
GENDIL_HOST_DEVICE
SerialRecursiveArray< T, Dims... > operator*( const T & a, const SerialRecursiveArray< T, Dims... > & x )
{
   SerialRecursiveArray< T, Dims... > res;
   UnitLoop< Dims... >( [&]( auto... indices )
   {
      res( indices... ) = a * x( indices... );
   });
   return res;
}

template <typename T, size_t... Dims >
GENDIL_HOST_DEVICE
auto MakeSerialRecursiveArray( std::index_sequence< Dims... > )
{
   return SerialRecursiveArray< T, Dims... >{};
}

}


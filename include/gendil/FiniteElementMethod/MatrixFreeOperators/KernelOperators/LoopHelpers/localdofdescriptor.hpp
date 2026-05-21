// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil {

template <
   size_t Component,
   bool IsVector,
   typename ThreadShape,
   typename RegisterShape,
   Integer ThreadDim,
   Integer RegisterDim >
struct LocalDofDescriptor
{
   using component = std::integral_constant< size_t, Component >;
   using thread_shape = ThreadShape;
   using register_shape = RegisterShape;

   static constexpr size_t component_id = Component;
   static constexpr bool is_vector = IsVector;
   static constexpr Integer thread_dim = ThreadDim;
   static constexpr Integer register_dim = RegisterDim;

   std::array< GlobalIndex, ThreadDim > thread_indices;
   std::array< GlobalIndex, RegisterDim > register_indices;
   std::array< GlobalIndex, ThreadDim + RegisterDim > indices;
};

template < typename T >
struct is_local_dof_descriptor : std::false_type { };

template <
   size_t Component,
   bool IsVector,
   typename ThreadShape,
   typename RegisterShape,
   Integer ThreadDim,
   Integer RegisterDim >
struct is_local_dof_descriptor<
   LocalDofDescriptor<
      Component,
      IsVector,
      ThreadShape,
      RegisterShape,
      ThreadDim,
      RegisterDim > >
   : std::true_type { };

template < typename T >
inline constexpr bool is_local_dof_descriptor_v =
   is_local_dof_descriptor< std::remove_cvref_t< T > >::value;

template <
   size_t NumThreadIndices,
   size_t NumRegisterIndices,
   size_t ... ThreadI,
   size_t ... RegisterI >
GENDIL_HOST_DEVICE
constexpr auto CombineLocalDofIndices(
   const std::array< GlobalIndex, NumThreadIndices > & thread_indices,
   const std::array< GlobalIndex, NumRegisterIndices > & register_indices,
   std::index_sequence< ThreadI... >,
   std::index_sequence< RegisterI... > )
{
   return std::array< GlobalIndex, NumThreadIndices + NumRegisterIndices >{
      thread_indices[ThreadI]...,
      register_indices[RegisterI]... };
}

template <
   size_t Component,
   bool IsVector,
   typename ThreadShape,
   typename RegisterShape,
   size_t NumThreadIndices,
   size_t NumRegisterIndices >
GENDIL_HOST_DEVICE
constexpr auto MakeLocalDofDescriptor(
   std::integral_constant< size_t, Component >,
   std::integral_constant< bool, IsVector >,
   ThreadShape,
   RegisterShape,
   const std::array< GlobalIndex, NumThreadIndices > & thread_indices,
   const std::array< GlobalIndex, NumRegisterIndices > & register_indices )
{
   return LocalDofDescriptor<
      Component,
      IsVector,
      ThreadShape,
      RegisterShape,
      NumThreadIndices,
      NumRegisterIndices >{
         thread_indices,
         register_indices,
         CombineLocalDofIndices(
            thread_indices,
            register_indices,
            std::make_index_sequence< NumThreadIndices >{},
            std::make_index_sequence< NumRegisterIndices >{} )
      };
}

template <
   typename Descriptor,
   Integer Dim,
   size_t ... ThreadI,
   size_t ... RegisterI >
GENDIL_HOST_DEVICE
constexpr auto MakeLocalDofDescriptorFromFullIndices_impl(
   const std::array< GlobalIndex, Dim > & indices,
   std::index_sequence< ThreadI... >,
   std::index_sequence< RegisterI... > )
{
   using DofDescriptor = std::remove_cvref_t< Descriptor >;
   static_assert( Dim == DofDescriptor::thread_dim + DofDescriptor::register_dim );

   return LocalDofDescriptor<
      DofDescriptor::component_id,
      DofDescriptor::is_vector,
      typename DofDescriptor::thread_shape,
      typename DofDescriptor::register_shape,
      DofDescriptor::thread_dim,
      DofDescriptor::register_dim >{
         std::array< GlobalIndex, DofDescriptor::thread_dim >{
            indices[ThreadI]... },
         std::array< GlobalIndex, DofDescriptor::register_dim >{
            indices[DofDescriptor::thread_dim + RegisterI]... },
         indices
      };
}

template <
   typename Descriptor,
   Integer Dim >
GENDIL_HOST_DEVICE
constexpr auto MakeLocalDofDescriptorFromFullIndices(
   const Descriptor &,
   const std::array< GlobalIndex, Dim > & indices )
{
   using DofDescriptor = std::remove_cvref_t< Descriptor >;
   return MakeLocalDofDescriptorFromFullIndices_impl< DofDescriptor >(
      indices,
      std::make_index_sequence< DofDescriptor::thread_dim >{},
      std::make_index_sequence< DofDescriptor::register_dim >{} );
}

} // namespace gendil

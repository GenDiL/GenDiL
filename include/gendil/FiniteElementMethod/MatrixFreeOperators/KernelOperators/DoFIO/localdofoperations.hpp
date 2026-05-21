// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"

namespace gendil {

template <
   typename LocalDofs,
   typename Descriptor,
   typename Value,
   size_t ... I >
GENDIL_HOST_DEVICE
void SetLocalDof_impl(
   LocalDofs & local_dofs,
   const Descriptor & dof,
   const Value & value,
   std::index_sequence< I... > )
{
   if constexpr ( Descriptor::is_vector )
   {
      std::get< Descriptor::component_id >( local_dofs )( dof.register_indices[I]... ) = value;
   }
   else
   {
      local_dofs( dof.register_indices[I]... ) = value;
   }
}

template <
   typename LocalDofs,
   typename Descriptor,
   typename Value >
GENDIL_HOST_DEVICE
void SetLocalDof(
   LocalDofs & local_dofs,
   const Descriptor & dof,
   const Value & value )
{
   static_assert(
      is_local_dof_descriptor_v< Descriptor >,
      "SetLocalDof requires a compile-time local DoF descriptor." );
   SetLocalDof_impl(
      local_dofs,
      dof,
      value,
      std::make_index_sequence< Descriptor::register_dim >{} );
}

template <
   typename LocalDofs,
   typename Descriptor,
   typename Value,
   size_t ... I >
GENDIL_HOST_DEVICE
void SubtractLocalDof_impl(
   LocalDofs & local_dofs,
   const Descriptor & dof,
   const Value & value,
   std::index_sequence< I... > )
{
   if constexpr ( Descriptor::is_vector )
   {
      std::get< Descriptor::component_id >( local_dofs )( dof.register_indices[I]... ) -= value;
   }
   else
   {
      local_dofs( dof.register_indices[I]... ) -= value;
   }
}

template <
   typename LocalDofs,
   typename Descriptor,
   typename Value >
GENDIL_HOST_DEVICE
void SubtractLocalDof(
   LocalDofs & local_dofs,
   const Descriptor & dof,
   const Value & value )
{
   static_assert(
      is_local_dof_descriptor_v< Descriptor >,
      "SubtractLocalDof requires a compile-time local DoF descriptor." );
   SubtractLocalDof_impl(
      local_dofs,
      dof,
      value,
      std::make_index_sequence< Descriptor::register_dim >{} );
}

template <
   typename KernelContext,
   typename LocalDofs,
   typename Descriptor,
   typename Value >
GENDIL_HOST_DEVICE
void SetLocalDofOnOwnerThread(
   const KernelContext & kernel_context,
   LocalDofs & local_dofs,
   const Descriptor & dof,
   const Value & value )
{
   ThreadLoop< typename Descriptor::thread_shape >( kernel_context, [&] ( auto... t )
   {
      const std::array< GlobalIndex, sizeof...(t) > thread_indices{
         static_cast< GlobalIndex >( t )... };
      if ( ThreadIndicesMatch( dof, thread_indices ) )
      {
         SetLocalDof( local_dofs, dof, value );
      }
   });
}

template <
   typename KernelContext,
   typename FESpace,
   typename LocalDofs,
   typename RHSLocalDofs >
GENDIL_HOST_DEVICE
void SubtractLocalDofVector(
   const KernelContext & kernel_context,
   const FESpace & fe_space,
   LocalDofs & local_dofs,
   const RHSLocalDofs & rhs_local_dofs )
{
   ForEachLocalResidualDof(
      kernel_context,
      fe_space,
      rhs_local_dofs,
      [&] ( const auto & dof, const auto & value )
      {
         SubtractLocalDof( local_dofs, dof, value );
      });
}

} // namespace gendil

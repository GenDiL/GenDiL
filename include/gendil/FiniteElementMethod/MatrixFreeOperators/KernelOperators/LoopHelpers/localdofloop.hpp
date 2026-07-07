// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/Restrictions/doflayout.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofdescriptor.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/subsequence.hpp"
#include "gendil/Utilities/Loop/loops.hpp"

namespace gendil {

template <
   size_t Component,
   bool IsVector,
   typename ThreadShape,
   typename RegisterShape,
   typename Lambda >
GENDIL_HOST_DEVICE
void ForEachLocalDof( Lambda && lambda )
{
   UnitLoop< ThreadShape >( [&] ( auto... t )
   {
      UnitLoop< RegisterShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...(t) > thread_indices{
            static_cast< GlobalIndex >( t )... };
         const std::array< GlobalIndex, sizeof...(k) > register_indices{
            static_cast< GlobalIndex >( k )... };
         const auto dof = MakeLocalDofDescriptor(
            std::integral_constant< size_t, Component >{},
            std::integral_constant< bool, IsVector >{},
            ThreadShape{},
            RegisterShape{},
            thread_indices,
            register_indices );
         lambda( dof );
      });
   });
}

template <
   size_t Component,
   bool IsVector,
   typename ThreadShape,
   typename RegisterShape,
   typename Lambda >
GENDIL_HOST_DEVICE
void ForEachLocalDofWithShapes( Lambda && lambda )
{
   ForEachLocalDof< Component, IsVector, ThreadShape, RegisterShape >(
      std::forward< Lambda >( lambda ) );
}

template < typename FESpace, typename Lambda >
GENDIL_HOST_DEVICE
void ForEachLocalDof( const FESpace &, Lambda && lambda )
{
   using Space = std::remove_cvref_t< FESpace >;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;

   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      constexpr Integer v_dim = ShapeFunctions::vector_dim;

      ConstexprLoop< v_dim >( [&] ( auto component_ )
      {
         constexpr size_t component = decltype(component_)::value;
         using ComponentDofShape =
            component_dof_shape_t< ShapeFunctions, component >;

         UnitLoop< ComponentDofShape >( [&] ( auto... k )
         {
            const std::array< GlobalIndex, sizeof...( k ) > indices{
               static_cast< GlobalIndex >( k )... };
            lambda( std::integral_constant< size_t, component >{}, indices );
         });
      });
   }
   else
   {
      using DofShape = finite_element_dof_shape_t< ShapeFunctions >;

      UnitLoop< DofShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         lambda( std::integral_constant< size_t, 0 >{}, indices );
      });
   }
}

template < typename FESpace, typename Lambda >
GENDIL_HOST_DEVICE
void ForEachScalarLocalDof( const FESpace &, Lambda && lambda )
{
   using Space = std::remove_cvref_t< FESpace >;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;
   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "ForEachScalarLocalDof supports scalar finite element spaces only." );

   using DofShape = finite_element_dof_shape_t< ShapeFunctions >;

   UnitLoop< DofShape >( [&] ( auto... k )
   {
      const std::array< GlobalIndex, sizeof...( k ) > indices{
         static_cast< GlobalIndex >( k )... };
      lambda( std::integral_constant< size_t, 0 >{}, indices );
   });
}

template <
   typename KernelContext,
   typename FESpace,
   typename Lambda >
GENDIL_HOST_DEVICE
void ForEachLocalTrialDof(
   const KernelContext &,
   const FESpace &,
   Lambda && lambda )
{
   using Space = std::remove_cvref_t< FESpace >;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;

   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      constexpr Integer v_dim = ShapeFunctions::vector_dim;
      using dof_shape = typename ShapeFunctions::dof_shape;

      ConstexprLoop< v_dim >( [&] ( auto component_ )
      {
         constexpr size_t component = decltype(component_)::value;
         using component_dof_shape =
            std::tuple_element_t< component, dof_shape >;
         using tshape =
            subsequence_t<
               component_dof_shape,
               typename KernelContext::template threaded_dimensions<
                  component_dof_shape::size() > >;
         using rshape =
            subsequence_t<
               component_dof_shape,
               typename KernelContext::template register_dimensions<
                  component_dof_shape::size() > >;

         ForEachLocalDof< component, true, tshape, rshape >( lambda );
      });
   }
   else
   {
      using DofShape = finite_element_dof_shape_t< ShapeFunctions >;
      using tshape =
         subsequence_t<
            DofShape,
            typename KernelContext::template threaded_dimensions<
               DofShape::size() > >;
      using rshape =
         subsequence_t<
            DofShape,
            typename KernelContext::template register_dimensions<
               DofShape::size() > >;

      ForEachLocalDof< 0, false, tshape, rshape >( lambda );
   }
}

template <
   typename Descriptor,
   Integer Dim,
   size_t ... I >
GENDIL_HOST_DEVICE
bool ThreadIndicesMatch_impl(
   const Descriptor & dof,
   const std::array< GlobalIndex, Dim > & thread_indices,
   std::index_sequence< I... > )
{
   return ( ( dof.thread_indices[I] == thread_indices[I] ) && ... );
}

template <
   typename Descriptor,
   Integer Dim >
GENDIL_HOST_DEVICE
bool ThreadIndicesMatch(
   const Descriptor & dof,
   const std::array< GlobalIndex, Dim > & thread_indices )
{
   static_assert( Descriptor::thread_dim == Dim );
   return ThreadIndicesMatch_impl(
      dof,
      thread_indices,
      std::make_index_sequence< Dim >{} );
}

template <
   typename KernelContext,
   typename FESpace,
   typename ElementVector,
   typename Lambda >
GENDIL_HOST_DEVICE
void ForEachLocalResidualDof(
   const KernelContext & kernel_context,
   const FESpace &,
   const ElementVector & y,
   Lambda && lambda )
{
   using Space = std::remove_cvref_t< FESpace >;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;

   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      constexpr Integer v_dim = ShapeFunctions::vector_dim;
      using dof_shape = typename ShapeFunctions::dof_shape;

      ConstexprLoop< v_dim >( [&] ( auto component_ )
      {
         constexpr size_t component = decltype(component_)::value;
         using component_dof_shape =
            std::tuple_element_t< component, dof_shape >;
         using tshape =
            subsequence_t<
               component_dof_shape,
               typename KernelContext::template threaded_dimensions<
                  component_dof_shape::size() > >;
         using rshape =
            subsequence_t<
               component_dof_shape,
               typename KernelContext::template register_dimensions<
                  component_dof_shape::size() > >;

         ThreadLoop< tshape >( kernel_context, [&] ( auto... t )
         {
            UnitLoop< rshape >( [&] ( auto... k )
            {
               const std::array< GlobalIndex, sizeof...(t) > thread_indices{
                  static_cast< GlobalIndex >( t )... };
               const std::array< GlobalIndex, sizeof...(k) > register_indices{
                  static_cast< GlobalIndex >( k )... };
               const auto dof = MakeLocalDofDescriptor(
                  std::integral_constant< size_t, component >{},
                  std::integral_constant< bool, true >{},
                  tshape{},
                  rshape{},
                  thread_indices,
                  register_indices );
               lambda( dof, std::get< component >( y )( k... ) );
            });
         });
      });
   }
   else
   {
      using DofShape = finite_element_dof_shape_t< ShapeFunctions >;
      using tshape =
         subsequence_t<
            DofShape,
            typename KernelContext::template threaded_dimensions<
               DofShape::size() > >;
      using rshape =
         subsequence_t<
            DofShape,
            typename KernelContext::template register_dimensions<
               DofShape::size() > >;

      ThreadLoop< tshape >( kernel_context, [&] ( auto... t )
      {
         UnitLoop< rshape >( [&] ( auto... k )
         {
            const std::array< GlobalIndex, sizeof...(t) > thread_indices{
               static_cast< GlobalIndex >( t )... };
            const std::array< GlobalIndex, sizeof...(k) > register_indices{
               static_cast< GlobalIndex >( k )... };
            const auto dof = MakeLocalDofDescriptor(
               std::integral_constant< size_t, 0 >{},
               std::integral_constant< bool, false >{},
               tshape{},
               rshape{},
               thread_indices,
               register_indices );
            lambda( dof, y( k... ) );
         });
      });
   }
}

} // namespace gendil

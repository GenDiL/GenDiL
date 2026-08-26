// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"
#include "gendil/Algebra/vectoraccess.hpp"
#include "gendil/FiniteElementMethod/Restrictions/globaldofindex.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/Loop/kernelloop.hpp"
#include "gendil/Utilities/MathHelperFunctions/atomicadd.hpp"

#include <type_traits>

namespace gendil
{

// Built-in mappings execute in the memory space selected by their matvec
// backend. Parallel H1 mappings use atomic accumulation for shared true DoFs.
// The current coordinate implementation requires one unit-weight entry per
// restriction row. Multi-entry mappings can later retain the same element-block
// representation by gathering through E and scattering through its adjoint.
namespace details
{

template < typename FiniteElementSpace >
GlobalIndex BsrInternalSize(
   const FiniteElementSpace & finite_element_space )
{
   return GetNumberOfLocalDofs( finite_element_space );
}

template < typename FiniteElementSpace >
GlobalIndex BsrExternalSize(
   const FiniteElementSpace & finite_element_space )
{
   return GetAlgebraicDofExtent( finite_element_space );
}

template <
   typename FiniteElementSpace,
   typename InputValue,
   typename OutputValue >
GENDIL_HOST_DEVICE
void GatherBsrElement(
   const FiniteElementSpace & finite_element_space,
   const GlobalIndex element_index,
   const InputValue * fe_data,
   OutputValue * bsr_data )
{
   using Space = std::remove_cvref_t< FiniteElementSpace >;
   using ShapeFunctions =
      finite_element_space_shape_functions_t< Space >;
   constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();

   ForEachLocalDof(
      finite_element_space,
      [&] (
         const auto component,
         const auto & indices )
      {
         const GlobalIndex bsr_index =
            element_index * block_size +
            FlattenLocalDof(
               finite_element_space,
               component,
               indices );
         const GlobalIndex fe_index = GetGlobalDofIndex(
            finite_element_space,
            component,
            element_index,
            indices );
         bsr_data[bsr_index] = fe_data[fe_index];
      } );
}

template <
   bool Add,
   bool Atomic,
   typename FiniteElementSpace,
   typename InputValue,
   typename OutputValue >
GENDIL_HOST_DEVICE
void ScatterBsrElement(
   const FiniteElementSpace & finite_element_space,
   const GlobalIndex element_index,
   const InputValue * bsr_data,
   OutputValue * fe_data )
{
   using Space = std::remove_cvref_t< FiniteElementSpace >;
   using ShapeFunctions =
      finite_element_space_shape_functions_t< Space >;
   constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();

   ForEachLocalDof(
      finite_element_space,
      [&] (
         const auto component,
         const auto & indices )
      {
         const GlobalIndex bsr_index =
            element_index * block_size +
            FlattenLocalDof(
               finite_element_space,
               component,
               indices );
         const GlobalIndex fe_index = GetGlobalDofIndex(
            finite_element_space,
            component,
            element_index,
            indices );

         if constexpr ( Atomic )
         {
            AtomicAddInPlace( fe_data[fe_index], bsr_data[bsr_index] );
         }
         else if constexpr ( Add )
         {
            fe_data[fe_index] += bsr_data[bsr_index];
         }
         else
         {
            fe_data[fe_index] = bsr_data[bsr_index];
         }
      } );
}

template <
   bool OnDevice,
   typename FiniteElementSpace,
   typename InputVector,
   typename OutputVector >
requires
   KernelAccessibleVector< OnDevice, InputVector > &&
   KernelAccessibleVector< OnDevice, OutputVector >
void GatherBsrKernel(
   const FiniteElementSpace & finite_element_space,
   const GlobalIndex num_elements,
   const InputVector & x_fe,
   OutputVector & x_bsr )
{
   const auto * fe_data = ReadKernelVector< OnDevice >( x_fe );
   auto * bsr_data = WriteKernelVector< OnDevice >( x_bsr );
   const auto kernel_space = finite_element_space;

   KernelLoop< OnDevice >(
      num_elements,
      [=] GENDIL_HOST_DEVICE ( const GlobalIndex element_index )
      {
         GatherBsrElement(
            kernel_space,
            element_index,
            fe_data,
            bsr_data );
      } );
}

template <
   bool Add,
   bool Injective,
   bool OnDevice,
   typename FiniteElementSpace,
   typename InputVector,
   typename OutputVector >
requires
   KernelAccessibleVector< OnDevice, InputVector > &&
   KernelAccessibleVector< OnDevice, OutputVector >
void ScatterBsrKernel(
   const FiniteElementSpace & finite_element_space,
   const GlobalIndex num_elements,
   const GlobalIndex output_size,
   const InputVector & y_bsr,
   OutputVector & y_fe )
{
   const auto * bsr_data = ReadKernelVector< OnDevice >( y_bsr );
   using OutputValue =
      std::remove_pointer_t<
         decltype( WriteKernelVector< OnDevice >( y_fe ) ) >;
   OutputValue * fe_data = nullptr;
   if constexpr ( Add )
   {
      fe_data = ReadWriteKernelVector< OnDevice >( y_fe );
   }
   else
   {
      fe_data = WriteKernelVector< OnDevice >( y_fe );
   }

   if constexpr ( !Injective && !Add )
   {
      KernelLoop< OnDevice >(
         output_size,
         [=] GENDIL_HOST_DEVICE ( const GlobalIndex i )
         {
            fe_data[i] = OutputValue( 0 );
         } );
   }

   const auto kernel_space = finite_element_space;
   constexpr bool accumulate = Add || !Injective;
   constexpr bool use_atomic = !Injective;
   KernelLoop< OnDevice >(
      num_elements,
      [=] GENDIL_HOST_DEVICE ( const GlobalIndex element_index )
      {
         ScatterBsrElement< accumulate, use_atomic >(
            kernel_space,
            element_index,
            bsr_data,
            fe_data );
      } );
}

template <
   typename Backend,
   typename FiniteElementSpace,
   typename InputVector,
   typename OutputVector >
void GatherBsr(
   const Backend &,
   const FiniteElementSpace & finite_element_space,
   const GlobalIndex num_elements,
   const InputVector & x_fe,
   OutputVector & x_bsr )
{
   constexpr bool on_host = is_host_matvec_backend_v< Backend >;
   constexpr bool on_device = is_device_matvec_backend_v< Backend >;
   static_assert(
      on_host != on_device,
      "SGBSR gather requires a backend derived from exactly one of "
      "HostMatVecBackend or DeviceMatVecBackend." );

   if constexpr (
      KernelAccessibleVector< on_device, InputVector > &&
      KernelAccessibleVector< on_device, OutputVector > )
   {
#if !defined(GENDIL_USE_DEVICE)
      if constexpr ( on_device )
      {
         static_assert(
            dependent_false_v< Backend >,
            "Device SGBSR gather requires GenDiL device support." );
      }
      else
#endif
      {
         ValidateRestrictionMemoryAccess< on_device >(
            GetRestriction( finite_element_space ),
            num_elements );
         GatherBsrKernel< on_device >(
            finite_element_space,
            num_elements,
            x_fe,
            x_bsr );
      }
   }
   else
   {
      static_assert(
         dependent_false_v< Backend, InputVector, OutputVector >,
         "SGBSR gather requires input and output vectors accessible in the "
         "selected backend's memory space." );
   }
}

template <
   bool Add,
   bool Injective,
   typename Backend,
   typename FiniteElementSpace,
   typename InputVector,
   typename OutputVector >
void ScatterBsr(
   const Backend &,
   const FiniteElementSpace & finite_element_space,
   const GlobalIndex num_elements,
   const GlobalIndex output_size,
   const InputVector & y_bsr,
   OutputVector & y_fe )
{
   constexpr bool on_host = is_host_matvec_backend_v< Backend >;
   constexpr bool on_device = is_device_matvec_backend_v< Backend >;
   static_assert(
      on_host != on_device,
      "SGBSR scatter requires a backend derived from exactly one of "
      "HostMatVecBackend or DeviceMatVecBackend." );

   if constexpr (
      KernelAccessibleVector< on_device, InputVector > &&
      KernelAccessibleVector< on_device, OutputVector > )
   {
#if !defined(GENDIL_USE_DEVICE)
      if constexpr ( on_device )
      {
         static_assert(
            dependent_false_v< Backend >,
            "Device SGBSR scatter requires GenDiL device support." );
      }
      else
#endif
      {
         ValidateRestrictionMemoryAccess< on_device >(
            GetRestriction( finite_element_space ),
            num_elements );
         ScatterBsrKernel< Add, Injective, on_device >(
            finite_element_space,
            num_elements,
            output_size,
            y_bsr,
            y_fe );
      }
   }
   else
   {
      static_assert(
         dependent_false_v< Backend, InputVector, OutputVector >,
         "SGBSR scatter requires input and output vectors accessible in the "
         "selected backend's memory space." );
   }
}

} // namespace details

/**
 * @brief Whether a finite-element space has the semantic operations required
 * by the default element-block BSR mapping.
 */
template < typename FESpace >
concept DefaultBsrMappingSpace =
   requires
   {
      typename std::remove_cvref_t< FESpace >::restriction_type;
   } &&
   ElementDoFRestriction<
      typename std::remove_cvref_t< FESpace >::restriction_type > &&
   static_restriction_entry_count_v<
      typename std::remove_cvref_t< FESpace >::restriction_type > == 1 &&
   restriction_supports_element_reference_view_v<
      typename std::remove_cvref_t< FESpace >::restriction_type > &&
   requires(
      const typename std::remove_cvref_t<
         FESpace >::restriction_type & restriction,
      const GlobalIndex active_row_count )
   {
      ValidateRestrictionMemoryAccess< false >(
         restriction,
         active_row_count );
      ValidateRestrictionMemoryAccess< true >(
         restriction,
         active_row_count );
   };

/** @brief Gather algebraic DoFs into the element-block BSR vector. */
template < typename FiniteElementSpace >
   requires DefaultBsrMappingSpace< FiniteElementSpace >
struct RestrictionGatherToBsr
{
   FiniteElementSpace finite_element_space;

   GlobalIndex ExternalSize() const
   {
      return details::BsrExternalSize( finite_element_space );
   }

   GlobalIndex InternalSize() const
   {
      return details::BsrInternalSize( finite_element_space );
   }

   template <
      typename Backend,
      typename InputVector,
      typename OutputVector >
   void operator()(
      const Backend & backend,
      const InputVector & x_fe,
      OutputVector & x_bsr ) const
   {
      using Space = std::remove_cvref_t< FiniteElementSpace >;
      using ShapeFunctions =
         finite_element_space_shape_functions_t< Space >;
      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size =
         LocalDofCount< ShapeFunctions >();

      GENDIL_VERIFY(
         GetVectorSize( x_bsr ) ==
            static_cast< size_t >( num_elements * block_size ),
         "RestrictionGatherToBsr output vector has the wrong BSR size." );
      GENDIL_VERIFY(
         GetVectorSize( x_fe ) >= static_cast< size_t >(
            GetAlgebraicDofExtent( finite_element_space ) ),
         "RestrictionGatherToBsr input vector is too small for the finite "
         "element space." );

      details::GatherBsr(
         backend,
         finite_element_space,
         num_elements,
         x_fe,
         x_bsr );
   }
};

/** @brief Scatter an element-block BSR vector through a completed restriction. */
template < typename FiniteElementSpace >
   requires DefaultBsrMappingSpace< FiniteElementSpace >
struct RestrictionScatterFromBsr
{
   FiniteElementSpace finite_element_space;

   GlobalIndex ExternalSize() const
   {
      return details::BsrExternalSize( finite_element_space );
   }

   GlobalIndex InternalSize() const
   {
      return details::BsrInternalSize( finite_element_space );
   }

   template <
      typename Backend,
      typename InputVector,
      typename OutputVector >
   void operator()(
      const Backend & backend,
      const InputVector & y_bsr,
      OutputVector & y_fe ) const
   {
      Scatter< false >( backend, y_bsr, y_fe );
   }

   template <
      typename Backend,
      typename InputVector,
      typename OutputVector >
   void ApplyAdd(
      const Backend & backend,
      const InputVector & y_bsr,
      OutputVector & y_fe ) const
   {
      Scatter< true >( backend, y_bsr, y_fe );
   }

private:
   template <
      bool Add,
      typename Backend,
      typename InputVector,
      typename OutputVector >
   void Scatter(
      const Backend & backend,
      const InputVector & y_bsr,
      OutputVector & y_fe ) const
   {
      using Space = std::remove_cvref_t< FiniteElementSpace >;
      using Restriction = typename Space::restriction_type;
      using ShapeFunctions =
         finite_element_space_shape_functions_t< Space >;
      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size =
         LocalDofCount< ShapeFunctions >();

      GENDIL_VERIFY(
         GetVectorSize( y_bsr ) ==
            static_cast< size_t >( num_elements * block_size ),
         "RestrictionScatterFromBsr input vector has the wrong BSR size." );
      GENDIL_VERIFY(
         GetVectorSize( y_fe ) >= static_cast< size_t >(
            GetAlgebraicDofExtent( finite_element_space ) ),
         "RestrictionScatterFromBsr output vector is too small for the finite "
         "element space." );

      constexpr bool injective =
         !restriction_may_share_global_dofs_v< Restriction >;
      details::ScatterBsr< Add, injective >(
         backend,
         finite_element_space,
         num_elements,
         GetAlgebraicDofExtent( finite_element_space ),
         y_bsr,
         y_fe );
   }
};

// Compatibility names retained for one release. They contain no family
// selection and all use the same semantic mapping implementation.
template < typename FiniteElementSpace >
using DGGatherToBsr =
   RestrictionGatherToBsr< FiniteElementSpace >;

template < typename FiniteElementSpace >
using DGScatterFromBsr =
   RestrictionScatterFromBsr< FiniteElementSpace >;

template < typename FiniteElementSpace >
using CGGatherToBsr =
   RestrictionGatherToBsr< FiniteElementSpace >;

template < typename FiniteElementSpace >
using CGScatterFromBsr =
   RestrictionScatterFromBsr< FiniteElementSpace >;

template < typename FiniteElementSpace >
using VectorCGGatherToBsr =
   RestrictionGatherToBsr< FiniteElementSpace >;

template < typename FiniteElementSpace >
using VectorCGScatterFromBsr =
   RestrictionScatterFromBsr< FiniteElementSpace >;

template < typename FESpace >
struct DefaultBsrGatherFor
{
   using space_type = std::remove_cvref_t< FESpace >;
   static_assert(
      DefaultBsrMappingSpace< space_type >,
      "DefaultBsrGatherFor requires a statically one-entry, unit-weight "
      "completed restriction with backend-access validation." );
   using type = RestrictionGatherToBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace >
using default_bsr_gather_t =
   typename DefaultBsrGatherFor< FESpace >::type;

template < typename FESpace >
struct DefaultBsrScatterFor
{
   using space_type = std::remove_cvref_t< FESpace >;
   static_assert(
      DefaultBsrMappingSpace< space_type >,
      "DefaultBsrScatterFor requires a statically one-entry, unit-weight "
      "completed restriction with backend-access validation." );
   using type = RestrictionScatterFromBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace >
using default_bsr_scatter_t =
   typename DefaultBsrScatterFor< FESpace >::type;

} // namespace gendil

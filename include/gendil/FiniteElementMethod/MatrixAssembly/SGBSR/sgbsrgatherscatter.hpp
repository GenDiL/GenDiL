// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"
#include "gendil/Algebra/vectoraccess.hpp"
#include "gendil/FiniteElementMethod/Restrictions/doflayout.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/Loop/kernelloop.hpp"
#include "gendil/Utilities/MathHelperFunctions/atomicadd.hpp"

#include <type_traits>

namespace gendil
{

struct IdentityBsrGather
{
   static constexpr bool is_identity = true;
};

struct IdentityBsrScatter
{
   static constexpr bool is_identity = true;
};

// Built-in mappings execute in the memory space selected by their matvec
// backend. Parallel H1 mappings use atomic accumulation for shared true DoFs.
namespace details
{

template < typename FiniteElementSpace >
void VerifyDeviceRestrictionMap(
   const FiniteElementSpace & finite_element_space,
   const GlobalIndex num_elements )
{
   using Space = std::remove_cvref_t< FiniteElementSpace >;
   using Restriction = typename Space::restriction_type;
   if constexpr ( is_h1_restriction_v< Restriction > )
   {
      GENDIL_VERIFY(
         num_elements == 0 ||
            finite_element_space.restriction.indices.device_pointer != nullptr,
         "Device SGBSR gather/scatter requires a device-resident H1 "
         "restriction map." );
   }
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
      typename Space::finite_element_type::shape_functions;
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
         const GlobalIndex fe_index =
            GlobalDofIndex(
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
      typename Space::finite_element_type::shape_functions;
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
         const GlobalIndex fe_index =
            GlobalDofIndex(
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
         if constexpr ( on_device )
         {
            VerifyDeviceRestrictionMap(
               finite_element_space,
               num_elements );
         }
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
         if constexpr ( on_device )
         {
            VerifyDeviceRestrictionMap(
               finite_element_space,
               num_elements );
         }
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

template < typename FiniteElementSpace >
struct DGGatherToBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

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
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      static_assert(
         std::is_same_v< typename Space::restriction_type, L2Restriction >,
         "DGGatherToBsr only supports L2Restriction finite element spaces." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();

      GENDIL_VERIFY(
         GetVectorSize( x_bsr ) ==
            static_cast< size_t >( num_elements * block_size ),
         "DGGatherToBsr output vector has the wrong BSR size." );
      GENDIL_VERIFY(
         GetVectorSize( x_fe ) >= static_cast< size_t >(
            finite_element_space.restriction.shift +
            finite_element_space.GetNumberOfFiniteElementDofs() ),
         "DGGatherToBsr input vector is too small for the finite element space." );

      details::GatherBsr(
         backend,
         finite_element_space,
         num_elements,
         x_fe,
         x_bsr );
   }
};

template < typename FiniteElementSpace >
struct DGScatterFromBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

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
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      static_assert(
         std::is_same_v< typename Space::restriction_type, L2Restriction >,
         "DGScatterFromBsr only supports L2Restriction finite element spaces." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();

      GENDIL_VERIFY(
         GetVectorSize( y_bsr ) ==
            static_cast< size_t >( num_elements * block_size ),
         "DGScatterFromBsr input vector has the wrong BSR size." );
      GENDIL_VERIFY(
         GetVectorSize( y_fe ) >= static_cast< size_t >(
            finite_element_space.restriction.shift +
            finite_element_space.GetNumberOfFiniteElementDofs() ),
         "DGScatterFromBsr output vector is too small for the finite element space." );

      const GlobalIndex output_size =
         finite_element_space.restriction.shift +
         finite_element_space.GetNumberOfFiniteElementDofs();
      details::ScatterBsr< Add, true >(
         backend,
         finite_element_space,
         num_elements,
         output_size,
         y_bsr,
         y_fe );
   }
};

template < typename FiniteElementSpace >
struct CGGatherToBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

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
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      static_assert(
         std::is_same_v< typename Space::restriction_type, H1Restriction >,
         "CGGatherToBsr only supports H1Restriction finite element spaces." );
      static_assert(
         !is_vector_shape_functions_v< ShapeFunctions >,
         "CGGatherToBsr currently supports scalar H1 finite element spaces only." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();
      const GlobalIndex expected_bsr_size =
         num_elements * block_size;

      GENDIL_VERIFY(
         GetVectorSize( x_bsr ) == static_cast< size_t >( expected_bsr_size ),
         "CGGatherToBsr output BSR vector size is inconsistent with the element-local DoF count." );
      GENDIL_VERIFY(
         GetVectorSize( x_fe ) >=
            static_cast< size_t >( finite_element_space.restriction.num_dofs ),
         "CGGatherToBsr input vector is smaller than the conforming H1 vector size." );

      // This gathers into the raw element-block BSR vector. The wrapped BSR
      // matrix is not a true-DoF globally assembled sparse matrix.
      details::GatherBsr(
         backend,
         finite_element_space,
         num_elements,
         x_fe,
         x_bsr );
   }
};

template < typename FiniteElementSpace >
struct CGScatterFromBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

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
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      static_assert(
         std::is_same_v< typename Space::restriction_type, H1Restriction >,
         "CGScatterFromBsr only supports H1Restriction finite element spaces." );
      static_assert(
         !is_vector_shape_functions_v< ShapeFunctions >,
         "CGScatterFromBsr currently supports scalar H1 finite element spaces only." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();
      const GlobalIndex expected_bsr_size =
         num_elements * block_size;

      GENDIL_VERIFY(
         GetVectorSize( y_bsr ) == static_cast< size_t >( expected_bsr_size ),
         "CGScatterFromBsr input BSR vector size is inconsistent with the element-local DoF count." );
      GENDIL_VERIFY(
         GetVectorSize( y_fe ) >=
            static_cast< size_t >( finite_element_space.restriction.num_dofs ),
         "CGScatterFromBsr output vector is smaller than the conforming H1 vector size." );

      details::ScatterBsr< Add, false >(
         backend,
         finite_element_space,
         num_elements,
         static_cast< GlobalIndex >(
            finite_element_space.restriction.num_dofs ),
         y_bsr,
         y_fe );
   }
};

template < typename FiniteElementSpace >
struct VectorCGGatherToBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

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
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      using Restriction = typename Space::restriction_type;
      static_assert(
         is_vector_h1_restriction_v< Restriction >,
         "VectorCGGatherToBsr only supports VectorH1Restriction<NComp> finite element spaces." );
      static_assert(
         is_vector_shape_functions_v< ShapeFunctions >,
         "VectorCGGatherToBsr requires a vector finite element space." );
      static_assert(
         Restriction::num_comp == ShapeFunctions::vector_dim,
         "VectorH1Restriction<NComp> must match the vector finite element component count." );
      static_assert(
         VectorComponentDofShapesMatchFirst< ShapeFunctions >(),
         "VectorH1Restriction currently requires identical scalar component DoF shapes." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();
      const GlobalIndex expected_bsr_size =
         num_elements * block_size;
      const GlobalIndex expected_fe_size =
         static_cast< GlobalIndex >( Restriction::num_comp ) *
         static_cast< GlobalIndex >(
            finite_element_space.restriction.scalar_num_dofs );

      GENDIL_VERIFY(
         GetVectorSize( x_bsr ) == static_cast< size_t >( expected_bsr_size ),
         "VectorCGGatherToBsr output BSR vector size is inconsistent with the element-local DoF count." );
      GENDIL_VERIFY(
         GetVectorSize( x_fe ) >= static_cast< size_t >( expected_fe_size ),
         "VectorCGGatherToBsr input vector is smaller than the vector conforming H1 vector size." );

      // Gather from component-major vector true DoFs into the element-block
      // BSR layout. The BSR local position is still the full vector local DoF.
      details::GatherBsr(
         backend,
         finite_element_space,
         num_elements,
         x_fe,
         x_bsr );
   }
};

template < typename FiniteElementSpace >
struct VectorCGScatterFromBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

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
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      using Restriction = typename Space::restriction_type;
      static_assert(
         is_vector_h1_restriction_v< Restriction >,
         "VectorCGScatterFromBsr only supports VectorH1Restriction<NComp> finite element spaces." );
      static_assert(
         is_vector_shape_functions_v< ShapeFunctions >,
         "VectorCGScatterFromBsr requires a vector finite element space." );
      static_assert(
         Restriction::num_comp == ShapeFunctions::vector_dim,
         "VectorH1Restriction<NComp> must match the vector finite element component count." );
      static_assert(
         VectorComponentDofShapesMatchFirst< ShapeFunctions >(),
         "VectorH1Restriction currently requires identical scalar component DoF shapes." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();
      const GlobalIndex expected_bsr_size =
         num_elements * block_size;
      const GlobalIndex expected_fe_size =
         static_cast< GlobalIndex >( Restriction::num_comp ) *
         static_cast< GlobalIndex >(
            finite_element_space.restriction.scalar_num_dofs );

      GENDIL_VERIFY(
         GetVectorSize( y_bsr ) == static_cast< size_t >( expected_bsr_size ),
         "VectorCGScatterFromBsr input BSR vector size is inconsistent with the element-local DoF count." );
      GENDIL_VERIFY(
         GetVectorSize( y_fe ) >= static_cast< size_t >( expected_fe_size ),
         "VectorCGScatterFromBsr output vector is smaller than the vector conforming H1 vector size." );

      details::ScatterBsr< Add, false >(
         backend,
         finite_element_space,
         num_elements,
         expected_fe_size,
         y_bsr,
         y_fe );
   }
};

template <
   typename FESpace,
   typename Restriction =
      typename std::remove_cvref_t< FESpace >::restriction_type >
struct DefaultBsrGatherFor
{
   static_assert(
      dependent_false_v< FESpace >,
      "DefaultBsrGatherFor supports only L2Restriction, scalar H1Restriction, and VectorH1Restriction finite element spaces." );
};

template < typename FESpace >
struct DefaultBsrGatherFor< FESpace, L2Restriction >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using type = DGGatherToBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace >
struct DefaultBsrGatherFor< FESpace, H1Restriction >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using ShapeFunctions =
      typename space_type::finite_element_type::shape_functions;

   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "CGGatherToBsr currently supports scalar H1 finite element spaces only." );

   using type = CGGatherToBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace, size_t NComp >
struct DefaultBsrGatherFor< FESpace, VectorH1Restriction< NComp > >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using type = VectorCGGatherToBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace >
using default_bsr_gather_t = typename DefaultBsrGatherFor< FESpace >::type;

template <
   typename FESpace,
   typename Restriction =
      typename std::remove_cvref_t< FESpace >::restriction_type >
struct DefaultBsrScatterFor
{
   static_assert(
      dependent_false_v< FESpace >,
      "DefaultBsrScatterFor supports only L2Restriction, scalar H1Restriction, and VectorH1Restriction finite element spaces." );
};

template < typename FESpace >
struct DefaultBsrScatterFor< FESpace, L2Restriction >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using type = DGScatterFromBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace >
struct DefaultBsrScatterFor< FESpace, H1Restriction >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using ShapeFunctions =
      typename space_type::finite_element_type::shape_functions;

   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "CGScatterFromBsr currently supports scalar H1 finite element spaces only." );

   using type = CGScatterFromBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace, size_t NComp >
struct DefaultBsrScatterFor< FESpace, VectorH1Restriction< NComp > >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using type = VectorCGScatterFromBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace >
using default_bsr_scatter_t = typename DefaultBsrScatterFor< FESpace >::type;

} // namespace gendil

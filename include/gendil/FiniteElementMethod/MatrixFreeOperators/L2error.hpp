// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Algebra/Vector/vector.hpp"
#include "gendil/Algebra/accessors.hpp"
#include "gendil/Algebra/staticalgebraconcepts.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"
#include "gendil/Utilities/KernelContext/isthreadeddim.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

#include <cmath>
#include <type_traits>
#include <utility>

namespace gendil {

template < typename KernelConfiguration >
static constexpr size_t required_l2_error_reduction_shared_memory_v =
   is_threaded_v< KernelConfiguration > ? 1 : 0;

namespace details
{

template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename MeshQuadData,
   typename ElementQuadData,
   typename Sigma,
   typename DofsInView >
GENDIL_HOST_DEVICE
void L2ErrorElementOperatorSerial(
   KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const IntegrationRule & int_rule,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   Sigma & sigma,
   const DofsInView & dofs_in,
   Real * sum_ptr )
{
   using Mesh = typename FiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );
   auto Bu = InterpolateValues( kernel_conf, element_quad_data, u );
   const auto cell = fe_space.GetCell( element_index );

   Real error = 0.0;
   QuadraturePointLoop< IntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      PhysicalCoordinates X;
      Jacobian J_mesh;

      const auto Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      const Real detJ = Determinant( J_mesh );
      const Real weight = GetWeight( quad_index, element_quad_data );
      const auto analytical_sol = sigma( X );

      const Real squared_difference = [&]
      {
         if constexpr (std::is_arithmetic_v<std::remove_cvref_t<decltype(Bu_q)>>)
         {
            const Real diff = Bu_q - analytical_sol;
            return diff * diff;
         }
         else
         {
            static_assert(
               static_extent_v<decltype(Bu_q)> ==
                  static_extent_v<decltype(analytical_sol)>,
               "Vector L2 exact values must match the field component count." );
            Real result = 0.0;
            ConstexprLoop<static_extent_v<decltype(Bu_q)>>([&](auto component)
            {
               const Real diff =
                  GetValue(Bu_q, component) -
                  GetValue(analytical_sol, component);
               result += diff * diff;
            });
            return result;
         }
      }();
      const Real Du_q = weight * detJ * squared_difference;

      error += Du_q;
   } );

   AtomicAddInPlace( *sum_ptr, error );
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename MeshQuadData,
   typename ElementQuadData,
   typename Sigma,
   typename DofsInView >
GENDIL_HOST_DEVICE
void L2ErrorElementOperatorThreaded(
   KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const IntegrationRule & int_rule,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   Sigma & sigma,
   const DofsInView & dofs_in,
   Real * sum_ptr )
{
   using Mesh = typename FiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );
   auto Bu = InterpolateValues( kernel_conf, element_quad_data, u );
   const auto cell = fe_space.GetCell( element_index );

   auto mark = kernel_conf.SharedAllocator.mark();
   Real * error = kernel_conf.SharedAllocator.allocate( 1 );

   // DeviceKernelConfiguration uses threadIdx.x as the per-work-item logical
   // thread index, so this elects one leader per BatchIndex lane.
   if ( kernel_conf.GetLinearThreadIndex() == 0 )
   {
      *error = 0.0;
   }
   kernel_conf.Sync();

   QuadraturePointLoop< IntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      PhysicalCoordinates X;
      Jacobian J_mesh;

      const auto Bu_q =
         ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      const Real detJ = Determinant( J_mesh );
      const Real weight = GetWeight( quad_index, element_quad_data );
      const auto analytical_sol = sigma( X );

      const Real squared_difference = [&]
      {
         if constexpr (std::is_arithmetic_v<std::remove_cvref_t<decltype(Bu_q)>>)
         {
            const Real diff = Bu_q - analytical_sol;
            return diff * diff;
         }
         else
         {
            static_assert(
               static_extent_v<decltype(Bu_q)> ==
                  static_extent_v<decltype(analytical_sol)>,
               "Vector L2 exact values must match the field component count." );
            Real result = 0.0;
            ConstexprLoop<static_extent_v<decltype(Bu_q)>>([&](auto component)
            {
               const Real diff =
                  GetValue(Bu_q, component) -
                  GetValue(analytical_sol, component);
               result += diff * diff;
            });
            return result;
         }
      }();
      const Real Du_q = weight * detJ * squared_difference;

      AtomicAddInPlace( *error, Du_q );
   } );

   kernel_conf.Sync();
   if ( kernel_conf.GetLinearThreadIndex() == 0 )
   {
      AtomicAddInPlace( *sum_ptr, *error );
   }
   kernel_conf.Sync();

   kernel_conf.SharedAllocator.restore( mark );
}

} // namespace details

/**
 * @brief Implementation of the L2Error operator at the element level.
 *
 * @tparam KernelContext Contextual information for the kernel.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @tparam Sigma The type of the function to evaluate the L2Error at physical coordinates.
 * @param fe_space The finite element space.
 * @param int_rule The integration rule used by the operator.
 * @param element_index The index of the finite element in the finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param sigma The function to evaluate the L2Error at physical coordinates.
 * @param dofs_in The input degrees of freedom.
 */
template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename MeshQuadData,
   typename ElementQuadData,
   typename Sigma,
   typename DofsInView >
GENDIL_HOST_DEVICE
void L2ErrorElementOperator(
   KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const IntegrationRule & int_rule,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   Sigma & sigma,
   const DofsInView & dofs_in,
   Real * sum_ptr )
{
   if constexpr ( is_threaded_v< KernelContext > )
   {
      details::L2ErrorElementOperatorThreaded(
         kernel_conf,
         fe_space,
         int_rule,
         element_index,
         mesh_quad_data,
         element_quad_data,
         sigma,
         dofs_in,
         sum_ptr );
   }
   else
   {
      details::L2ErrorElementOperatorSerial(
         kernel_conf,
         fe_space,
         int_rule,
         element_index,
         mesh_quad_data,
         element_quad_data,
         sigma,
         dofs_in,
         sum_ptr );
   }
}

/**
 * @brief Homogeneous-space L2-error implementation used by public wrappers.
 * 
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam Sigma The type of the function to evaluate the L2Error at physical coordinates.
 * @param fe_space The finite element space.
 * @param int_rule The integration rule used by the operator.
 * @param sigma The function to evaluate the L2Error  at physical coordinates.
 * @param dofs_vector_in The input degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FiniteElementSpace,
   typename Sigma,
   typename DofsVector
   >
Real ComputeHomogeneousL2Error(
   const FiniteElementSpace & fe_space,
   const IntegrationRule & int_rule,
   Sigma & sigma,
   const DofsVector & dofs_vector_in )
{
   using finite_element_type = typename FiniteElementSpace::finite_element_type;
   using shape_functions = typename finite_element_type::shape_functions;
   using Mesh = typename FiniteElementSpace::mesh_type;
   using MeshQuadData = typename Mesh::cell_type::template QuadData< IntegrationRule >;

   const MeshQuadData mesh_quad_data{};
   const auto element_quad_data = MakeDofToQuad<shape_functions,IntegrationRule>();
   auto dofs_in = MakeReadOnlyElementTensorView< KernelConfiguration >( fe_space, dofs_vector_in );

   HostDevicePointer< Real > sum_ptr;
   AllocateHostPointer< Real >( 1, sum_ptr );
   *sum_ptr = 0.0;
   if constexpr ( is_device_configuration_v< KernelConfiguration > )
   {
      AllocateDevicePointer< Real >( 1, sum_ptr );
      ToDevice( 1, sum_ptr );
   }

   mesh::CellIterator< KernelConfiguration >(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem =
            required_shared_memory_v<
               KernelConfiguration,
               IntegrationRule > +
            required_l2_error_reduction_shared_memory_v<
               KernelConfiguration >;
         using Context =
            KernelContext< KernelConfiguration, required_shared_mem >;
         GENDIL_SHARED Real _shared_mem[Context::shared_memory_block_size];

         Context kernel_conf( _shared_mem );

         L2ErrorElementOperator(
            kernel_conf,
            fe_space,
            int_rule,
            element_index,
            mesh_quad_data,
            element_quad_data,
            sigma,
            dofs_in,
            sum_ptr );
      }
   );

   // Host policies already accumulated into the host copy, even in GPU builds.
   if constexpr ( is_device_configuration_v< KernelConfiguration > )
   {
      ToHost( 1, sum_ptr );
      FreeDevicePointer( sum_ptr );
   }
   Real sum = Sqrt( *sum_ptr );
   FreeHostPointer( sum_ptr );
   return sum;
}

/**
 * @brief Compute a scalar or component-vector L2 error over one field space.
 *
 * Mixed spaces accumulate the squared child errors and take one final square
 * root. Child restrictions read directly from the common algebraic vector.
 */
template<class KernelPolicy,
         class Space,
         class IntegrationRule,
         class ExactFunction,
         class DofsVector>
Real ComputeL2Error(
   const Space& space,
   const IntegrationRule& integration_rule,
   ExactFunction exact,
   const DofsVector& dofs)
{
   GENDIL_VERIFY(
      GetVectorSize(dofs) ==
         static_cast<size_t>(GetAlgebraicDofExtent(space)),
      "L2-error vector extent does not match the finite-element space." );
   Real squared_error = 0.0;
   ForEachHomogeneousSpace(
      space,
      [&](const auto& cell_space)
      {
         const Real child_error = ComputeHomogeneousL2Error<KernelPolicy>(
            cell_space,
            integration_rule,
            exact,
            dofs);
         squared_error += child_error * child_error;
      });
   return std::sqrt(squared_error);
}

/**
 * @brief Compute a scalar or component-vector L2 error with a named or temporary
 * exact-solution callable.
 *
 * Forwards to ComputeL2Error, which owns the callable by value and copies it
 * into the execution kernels.
 */
template<typename KernelPolicy,
         typename IntegrationRule,
         typename FiniteElementSpace,
         typename Sigma,
         typename DofsVector>
Real L2Error(
   const FiniteElementSpace& finite_element_space,
   const IntegrationRule& integration_rule,
   Sigma&& exact,
   const DofsVector& dofs)
{
   return ComputeL2Error<KernelPolicy>(
      finite_element_space,
      integration_rule,
      std::forward<Sigma>(exact),
      dofs);
}

}

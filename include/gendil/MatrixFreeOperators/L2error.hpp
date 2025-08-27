// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/vector.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

namespace gendil {

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
   const KernelContext & kernel_conf,
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

   GENDIL_SHARED Real error;
#ifdef GENDIL_USE_DEVICE
   if( kernel_conf.GetLinearThreadIndex() == 0 ) error = 0.0;
   GENDIL_SYNC_THREADS();
#else
   error = 0.0;
#endif

   QuadraturePointLoop< IntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      PhysicalCoordinates X;
      Jacobian J_mesh;

      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      const Real detJ = Determinant( J_mesh );
      const Real weight = GetWeight( quad_index, element_quad_data );
      const Real analytical_sol = sigma( X );

      const Real diff = Bu_q - analytical_sol;
      const Real Du_q = weight * detJ * diff * diff;

      AtomicAdd( error, Du_q );
   } );

   AtomicAdd( *sum_ptr, error );
}

/**
 * @brief L2Error operator.
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
   typename Sigma
   >
Real L2Error(
   const FiniteElementSpace & fe_space,
   const IntegrationRule & int_rule,
   Sigma & sigma,
   const Vector & dofs_vector_in )
{
   using finite_element_type = typename FiniteElementSpace::finite_element_type;
   using shape_functions = typename finite_element_type::shape_functions;
   using Mesh = typename FiniteElementSpace::mesh_type;
   using MeshQuadData = typename Mesh::cell_type::template QuadData< IntegrationRule >;

   const MeshQuadData mesh_quad_data{};
   const auto element_quad_data = MakeDofToQuad<shape_functions,IntegrationRule>();
   auto dofs_in = MakeReadOnlyEVectorView< KernelConfiguration >( fe_space, dofs_vector_in );

   HostDevicePointer< Real > sum_ptr;
   AllocateHostPointer< Real >( 1, sum_ptr );
   AllocateDevicePointer< Real >( 1, sum_ptr );
   *sum_ptr = 0.0;
   ToDevice( 1, sum_ptr );

   mesh::CellIterator< KernelConfiguration >(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

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
   ToHost( 1, sum_ptr );
   Real sum = Sqrt( *sum_ptr );
   FreeHostPointer( sum_ptr );
   FreeDevicePointer( sum_ptr );
   return sum;
}

}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/kerneloperators.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/vector.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/DoFIO/evectorview.hpp"

namespace gendil {

/**
 * @brief Implementation of the linear form operator at the element level.
 * 
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam KernelContext Contextual information for the kernel.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam Lambda The source function type.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @param fe_space The finite element space.
 * @param element_index The index of the finite element in the finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param lambda The input function.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename IntegrationRule,
   typename KernelContext,
   typename FiniteElementSpace,
   typename Lambda,
   typename MeshQuadData,
   typename ElementQuadData >
GENDIL_HOST_DEVICE
void LinearFormElementOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   Lambda && lambda,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   using Mesh = typename FiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   const auto cell = fe_space.GetCell( element_index );

   // Container to store values at all the quadrature points
   auto DBu = MakeQuadraturePointValuesContainer( kernel_conf, IntegrationRule{} );

   // Application of the QFunction
   QuadraturePointLoop< IntegrationRule >(
   [&] ( auto const & quad_index )
   {
      PhysicalCoordinates X;
      Jacobian J_mesh;

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      Jacobian inv_J_mesh;
      const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J_mesh );

      const Real weight = GetWeight( quad_index, element_quad_data );

      const Real Du_q = lambda( X );

      const Real Du_q_w = weight * detJ * Du_q;

      WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q_w, DBu );
   } );

   // Application of the test functions
   auto BDBu = ApplyTestFunctions( kernel_conf, element_quad_data, DBu );

   WriteDofs( kernel_conf, fe_space, element_index, BDBu, dofs_out );
}

/**
 * @brief Assemble all the element contributions together.
 * 
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam Lambda The source function type.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @param fe_space The finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param lambda The input function.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FiniteElementSpace,
   typename Lambda,
   typename MeshQuadData,
   typename ElementQuadData >
void LinearFormOperator( const FiniteElementSpace & fe_space,
                         const MeshQuadData & mesh_quad_data,
                         const ElementQuadData & element_quad_data,
                         Lambda && lambda,
                         StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   mesh::CellIterator< KernelConfiguration >(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         LinearFormElementOperator< IntegrationRule >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            element_quad_data,
            lambda,
            dofs_out );
      }
   );
}

/**
 * @brief Represent a linear form.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule >
class LinearForm : public Vector
{
   using fe_space_type = FiniteElementSpace;
   using finite_element_type = typename fe_space_type::finite_element_type;

   using integration_rule = IntegrationRule;

   const FiniteElementSpace & finite_element_space;
   const IntegrationRule & int_rules;

   using Mesh = typename FiniteElementSpace::mesh_type;

   /**
    * @brief Data structure storing information to compute physical coordinates
    * and mesh jacobians at the provided integration rule quadrature points.
    * 
    */
   using MeshQuadData = typename Mesh::cell_type::template QuadData< IntegrationRule >;

   /**
    * @brief Data structure storing information to compute field values
    * and field gradients at the provided integration rule quadrature points.
    * 
    */
   using ElementQuadData = decltype(
                              MakeDofToQuad<
                                 typename finite_element_type::shape_functions,
                                 integration_rule
                              >()
                           );

public:
   /**
    * @brief Construct a new LinearForm object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    * @param lambda The analytical function to initialize the LinearForm.
    */
   template < typename Lambda >
   LinearForm( const FiniteElementSpace & finite_element_space,
               const IntegrationRule & int_rules,
               Lambda && lambda ) :
      Vector( finite_element_space.GetNumberOfFiniteElementDofs() ),
      finite_element_space( finite_element_space ),
      int_rules( int_rules )
   {
      MeshQuadData mesh_quad_data;
      ElementQuadData element_quad_data;
      auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( finite_element_space, *this );

      LinearFormOperator< KernelPolicy, integration_rule >( finite_element_space, mesh_quad_data, element_quad_data, lambda, dofs_out );
   }
};

/**
 * @brief Factory to build linear forms. Useful to hide the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return The linear form, can be used as an mfem::Vector.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Lambda >
auto MakeLinearForm( const FiniteElementSpace & finite_element_space,
                     const IntegrationRule & int_rule,
                     Lambda && lambda )
{
   return LinearForm<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule, lambda );
}

/**
 * @brief Factory to build linear forms. Useful to hide the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return The linear form, can be used as an mfem::Vector.
 */
template <
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Lambda >
auto MakeLinearForm( const FiniteElementSpace & finite_element_space,
                     const IntegrationRule & int_rule,
                     Lambda && lambda )
{
   using KernelPolicy = SerialKernelConfiguration;

   return LinearForm<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule, lambda );
}

}

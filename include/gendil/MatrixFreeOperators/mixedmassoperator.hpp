// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

namespace gendil {

/**
 * @brief Implementation of the "volume" contributions of the Mass operator at the element level.
 * 
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam KernelPolicy Contextual information for the kernel.
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param fe_space The finite element space.
 * @param element_index The index of the finite element in the finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename IntegrationRule,
   typename KernelPolicy,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename MeshQuadData,
   typename TrialElementQuadData,
   typename TestElementQuadData,
   typename Sigma >
GENDIL_HOST_DEVICE
void MixedMassElementOperator(
   const KernelPolicy & kernel_conf,
   const TrialFiniteElementSpace & trial_fe_space,
   const TestFiniteElementSpace & test_fe_space,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const TrialElementQuadData & trial_element_quad_data,
   const TestElementQuadData & test_element_quad_data,
   Sigma & sigma,
   const StridedView< TrialFiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< TestFiniteElementSpace::Dim + 1, Real > dofs_out )
{
   using Mesh = typename TrialFiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   static_assert(
      TrialFiniteElementSpace::Dim == TestFiniteElementSpace::Dim,
      "Trial and test finite element space dimension must match!"
   );

   auto u = ReadDofs( kernel_conf, trial_fe_space, element_index, dofs_in );

   auto Bu = InterpolateValues( kernel_conf, trial_element_quad_data, u );

   const auto cell = trial_fe_space.GetCell( element_index );

   // Container to store values at all the quadrature points
   auto& Duq = Bu;

   // Application of the QFunction
   QuadraturePointLoop< IntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      PhysicalCoordinates X;
      Jacobian J_mesh;

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      Jacobian inv_J_mesh;
      const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J_mesh );

      const Real weight = GetWeight( quad_index, trial_element_quad_data );

      const Real D_Mass = sigma( X );

      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      const Real Du_q = D_Mass * Bu_q;

      const Real Du_q_w = weight * detJ * Du_q;

      WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q_w, Duq );
   } );

   // Application of the test functions
   auto BDBu = ApplyTestFunctions( kernel_conf, test_element_quad_data, Duq );

   WriteDofs( kernel_conf, test_fe_space, element_index, BDBu, dofs_out );
}

/**
 * @brief Explicit mass operator containing only the "volume" contributions.
 * 
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param fe_space The finite element space.
 * @param fe_space The finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param trial_element_quad_data The trial finite element data at quadrature points need to perform the computation.
 * @param test_element_quad_data The test finite element data at quadrature points need to perform the computation.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename MeshQuadData,
   typename TrialElementQuadData,
   typename TestElementQuadData,
   typename Sigma >
void MixedMassExplicitOperator(
   const TrialFiniteElementSpace & trial_fe_space,
   const TestFiniteElementSpace & test_fe_space,
   const MeshQuadData & mesh_quad_data,
   const TrialElementQuadData & trial_element_quad_data,
   const TestElementQuadData & test_element_quad_data,
   Sigma sigma,
   const StridedView< TrialFiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< TestFiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   // Assumes same underlying mesh for trial_fe_space and test_fe_space
   mesh::CellIterator< KernelConfiguration >(
      trial_fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index )
      {
         constexpr size_t required_shared_mem = required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         MixedMassElementOperator< IntegrationRule >( 
            kernel_conf,
            trial_fe_space,
            test_fe_space,
            element_index,
            mesh_quad_data,
            trial_element_quad_data,
            test_element_quad_data,
            sigma,
            dofs_in,
            dofs_out );
      }
   );
}

/**
 * @brief Represent a Mass finite element operrator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param fe_space The finite element space.
 */
template <
   typename KernelPolicy,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename IntegrationRule,
   typename Sigma >
class MixedMassOperator
   : public MatrixFreeMixedBilinearFiniteElementOperator< TrialFiniteElementSpace, TestFiniteElementSpace, IntegrationRule >
{
   using base = MatrixFreeMixedBilinearFiniteElementOperator< TrialFiniteElementSpace, TestFiniteElementSpace, IntegrationRule >;
   Sigma sigma;

   using input = StridedView< TrialFiniteElementSpace::Dim + 1, const Real >;
   using output = StridedView< TestFiniteElementSpace::Dim + 1, Real >;

public:
   /**
    * @brief Construct a new MixedMassOperator object.
    * 
    * @param trial_finite_element_space The trial finite element space associated to the operator.
    * @param test_finite_element_space The test finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    * @param sigma The function to evaluate the mass density at physical coordinates.
    */
   MixedMassOperator(
      const TrialFiniteElementSpace & trial_finite_element_space,
      const TestFiniteElementSpace & test_finite_element_space,
      const IntegrationRule & int_rules,
      Sigma & sigma ) :
         base( trial_finite_element_space, test_finite_element_space, int_rules ),
         sigma( sigma )
   { }

   /**
    * @brief Apply the mixed-mass operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      MixedMassExplicitOperator< KernelPolicy, typename base::integration_rule >(
         this->trial_finite_element_space,
         this->test_finite_element_space,
         this->mesh_quad_data,
         this->trial_element_quad_data,
         this->test_element_quad_data,
         sigma,
         dofs_in,
         dofs_out );
   }

   void operator()( const Vector & dofs_vector_in, Vector & dofs_vector_out ) const
   {
      auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
      auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }

   #ifdef GENDIL_USE_MFEM
   /**
    * @brief Apply the mixed-mass operator.
    * 
    * @param dofs_vector_in The input degrees of freedom.
    * @param dofs_vector_out The output degrees of freedom.
    */
   void Mult( const mfem::Vector & dofs_vector_in,
              mfem::Vector & dofs_vector_out ) const override
   {
      auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->trial_finite_element_space, dofs_vector_in );
      auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( this->test_finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }
   #endif // GENDIL_USE_MFEM
};

/**
 * @brief Factory to build Mass operators. Useful to hide the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param trial_finite_element_space The trial finite element space associated to the operator.
 * @param test_finite_element_space The test finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @return auto The Mass operator.
 */
template <
   typename KernelPolicy,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename IntegrationRule,
   typename Sigma >
auto MakeMixedMassOperator(
   const TrialFiniteElementSpace & trial_finite_element_space,
   const TestFiniteElementSpace & test_finite_element_space,
   const IntegrationRule & int_rule,
   Sigma & sigma )
{
   return MixedMassOperator<
             KernelPolicy,
             TrialFiniteElementSpace,
             TestFiniteElementSpace,
             IntegrationRule,
             Sigma
          >( trial_finite_element_space, test_finite_element_space, int_rule, sigma );
}

template <
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename IntegrationRule,
   typename Sigma >
auto MakeMixedMassOperator(
   const TrialFiniteElementSpace & trial_finite_element_space,
   const TestFiniteElementSpace & test_finite_element_space,
   const IntegrationRule & int_rule,
   Sigma & sigma )
{
   using KernelPolicy = SerialKernelConfiguration;

   return MixedMassOperator<
             KernelPolicy,
             TrialFiniteElementSpace,
             TestFiniteElementSpace,
             IntegrationRule,
             Sigma
          >( trial_finite_element_space, test_finite_element_space, int_rule, sigma );
}

}

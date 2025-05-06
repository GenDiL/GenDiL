// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

namespace gendil {

// TODO: Generalize
// !FIXME This is really a subspace projection using integration and Linf projection
/**
 * @brief Implementation of Linf projection at the element level.
 * 
 * @tparam TrialIntegrationRule The type of the integration rule used by the element operator.
 * @tparam TestIntegrationRule The type of the integration rule used by the element operator.
 * @tparam KernelContext Contextual information for the kernel.
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @param fe_space The finite element space.
 * @param element_index The index of the finite element in the finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename TrialIntegrationRule,
   typename TestIntegrationRule,
   typename KernelContext,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename MeshQuadData,
   typename TrialElementQuadData,
   typename TestElementQuadData >
GENDIL_HOST_DEVICE
void LinfProjectionElementOperator(
   const KernelContext & kernel_conf,
   const TrialFiniteElementSpace & trial_fe_space,
   const TestFiniteElementSpace & test_fe_space,
   const GlobalIndex trial_element_index,
   const MeshQuadData & mesh_quad_data,
   const TrialElementQuadData & trial_element_quad_data,
   const TestElementQuadData & test_element_quad_data,
   const StridedView< TrialFiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< TestFiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   using Mesh = typename TrialFiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   static_assert(
      TrialFiniteElementSpace::Dim >= TestFiniteElementSpace::Dim,
      "Trial finite element space dimension must be higher than the test finite element space!"
   );
   constexpr Integer TestDim = TestFiniteElementSpace::Dim;

   auto u = ReadDofs( kernel_conf, trial_fe_space, trial_element_index, dofs_in );

   auto Bu = InterpolateValues( kernel_conf, trial_element_quad_data, u );

   const auto cell = mesh::GetCell( trial_fe_space, trial_element_index );

   // Container to store values at all the quadrature points
   auto DBu = MakeQuadraturePointValuesContainer( kernel_conf, TrialIntegrationRule{} );

   // Application of the QFunction
   QuadraturePointLoop< TrialIntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      PhysicalCoordinates X;
      Jacobian J_mesh;

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      constexpr Integer n_tuple_elements = std::tuple_size< Jacobian >{};
      using sub_indices = make_index_sequence_without< n_tuple_elements, 0 >;

      const Real detJ = Determinant( get_subtuple( J_mesh, sub_indices{} ) );

      auto sub_quad_index = quad_index.template Sub< 0, TestDim >();

      const Real test_weight = GetWeight( sub_quad_index, test_element_quad_data );
      const Real trial_weight = GetWeight( quad_index, trial_element_quad_data );
      const Real weight = trial_weight / test_weight;

      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      const Real Du_q_w = weight * detJ * Bu_q;

      // WriteAddQuadratureLocalValues( kernel_conf, sub_quad_index, Du_q_w, DBu );
      WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q_w, DBu );
   } );

   // The static cast of the quadrature values to dofs values is Linf projection
   // ElementDoF< TestFiniteElementSpace > BDBu = static_cast< ElementDoF< TestFiniteElementSpace > >( Duq );
   
   auto DBu_low =
      AggregateDimensions< TrialIntegrationRule, TestIntegrationRule >(
         kernel_conf,
         DBu,
         std::make_index_sequence< TestDim >{} );

   // Assumes tensor mesh
   GlobalIndex test_element_index = trial_fe_space.template GetStructuredSubIndex< 0 >( trial_element_index );
   WriteAddDofs( kernel_conf, test_fe_space, test_element_index, DBu_low, dofs_out );
}

/**
 * @brief Gloabl Linf projection operator for DG fields.
 * 
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @param fe_space The finite element space.
 * @param fe_space The finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param trial_element_quad_data The trial finite element data at quadrature points need to perform the computation.
 * @param test_element_quad_data The test finite element data at quadrature points need to perform the computation.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename TrialIntegrationRule,
   typename TestIntegrationRule,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename MeshQuadData,
   typename TrialElementQuadData,
   typename TestElementQuadData >
void LinfProjectionOperator(
   const TrialFiniteElementSpace & trial_fe_space,
   const TestFiniteElementSpace & test_fe_space,
   const MeshQuadData & mesh_quad_data,
   const TrialElementQuadData & trial_element_quad_data,
   const TestElementQuadData & test_element_quad_data,
   const StridedView< TrialFiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< TestFiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   mesh::CellIterator< KernelConfiguration >(
      trial_fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem =
            Max(
               required_shared_memory_v< KernelConfiguration, TrialIntegrationRule >, // Interpolation
               required_shared_memory_v< KernelConfiguration, TestIntegrationRule >, // Interpolation
               Product( typename TestIntegrationRule::points::num_points_tensor{} ) // Accumulation at quadrature point
            );
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         LinfProjectionElementOperator< TrialIntegrationRule, TestIntegrationRule >(
            kernel_conf,
            trial_fe_space,
            test_fe_space,
            element_index,
            mesh_quad_data,
            trial_element_quad_data,
            test_element_quad_data,
            dofs_in,
            dofs_out );
      }
   );
}

/**
 * @brief Represent a Linf projection finite element operrator.
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
   typename TrialIntegrationRule,
   typename TestIntegrationRule >
class LinfProjection : public mfem::Operator
{
protected:
   using trial_fe_space_type = TrialFiniteElementSpace;
   using test_fe_space_type = TestFiniteElementSpace;
   using trial_finite_element_type = typename trial_fe_space_type::finite_element_type;
   using test_finite_element_type = typename test_fe_space_type::finite_element_type;

   using trial_integration_rule = TrialIntegrationRule;
   using test_integration_rule = TestIntegrationRule; // TODO: Should this be automatically deduced?

   const trial_fe_space_type & trial_finite_element_space;
   const test_fe_space_type & test_finite_element_space;
   const trial_integration_rule & trial_int_rule;
   const test_integration_rule & test_int_rule;

   using Mesh = typename TrialFiniteElementSpace::mesh_type;

   /**
    * @brief Data structure storing information to compute physical coordinates
    * and mesh jacobians at the provided integration rule quadrature points.
    * 
    */
   using MeshQuadData = typename Mesh::cell_type::template QuadData< TrialIntegrationRule >;
   MeshQuadData mesh_quad_data;

   /**
    * @brief Data structure storing information to compute field values
    * and field gradients on the trial finite element space at the provided
    * integration rule quadrature points.
    * 
    */
   using TrialElementQuadData = decltype(
                              MakeDofToQuad<
                                 trial_finite_element_type,
                                 trial_integration_rule
                              >()
                           );
   TrialElementQuadData trial_element_quad_data;

   /**
    * @brief Data structure storing information to compute field values
    * and field gradients on the test finite element space at the provided
    * integration rule quadrature points.
    * 
    */
   using TestElementQuadData = decltype(
                              MakeDofToQuad<
                                 test_finite_element_type,
                                 test_integration_rule
                              >()
                           );
   TestElementQuadData test_element_quad_data;

   using input = StridedView< TrialFiniteElementSpace::Dim + 1, const Real >;
   using output = StridedView< TestFiniteElementSpace::Dim + 1, Real >;

public:
   /**
    * @brief Construct a new LinfProjection object.
    * 
    * @param trial_finite_element_space The trial finite element space associated to the operator.
    * @param test_finite_element_space The test finite element space associated to the operator.
    * @param trial_int_rule The integration rule used by the operator.
    * @param test_int_rule The integration rule used by the operator.
    */
   LinfProjection(
      const TrialFiniteElementSpace & trial_finite_element_space,
      const TestFiniteElementSpace & test_finite_element_space,
      const TrialIntegrationRule & trial_int_rule,
      const TestIntegrationRule & test_int_rule ) :
         mfem::Operator( test_finite_element_space.GetNumberOfFiniteElementDofs(), trial_finite_element_space.GetNumberOfFiniteElementDofs() ),
         trial_finite_element_space( trial_finite_element_space ),
         test_finite_element_space( test_finite_element_space ),
         trial_int_rule( trial_int_rule ),
         test_int_rule( test_int_rule ),
         mesh_quad_data{},
         trial_element_quad_data{},
         test_element_quad_data{}
   { }

   /**
    * @brief Apply the Linf projection operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      // TODO: dofs_out = 0.0
      LinfProjectionOperator< KernelPolicy, trial_integration_rule, test_integration_rule >(
         this->trial_finite_element_space,
         this->test_finite_element_space,
         this->mesh_quad_data,
         this->trial_element_quad_data,
         this->test_element_quad_data,
         dofs_in,
         dofs_out );
   }

   #ifdef GENDIL_USE_MFEM
   /**
    * @brief Apply the Linf projection operator.
    * 
    * @param dofs_vector_in The input degrees of freedom.
    * @param dofs_vector_out The output degrees of freedom.
    */
   void Mult( const mfem::Vector & dofs_vector_in,
              mfem::Vector & dofs_vector_out ) const override
   {
      dofs_vector_out = 0.0;
      auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->trial_finite_element_space, dofs_vector_in );
      auto dofs_out = MakeReadWriteEVectorView< KernelPolicy >( this->test_finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }
   #endif // GENDIL_USE_MFEM
};

/**
 * @brief Factory to build Linf projection operators. Useful to hide the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam TrialIntegrationRule The type of the integration rule used by the operator on the trial space.
 * @tparam TestIntegrationRule The type of the integration rule used by the operator on the test space.
 * @param trial_finite_element_space The trial finite element space associated to the operator.
 * @param test_finite_element_space The test finite element space associated to the operator.
 * @param trial_int_rule The integration rule used by the operator on the trial space.
 * @param test_int_rule The integration rule used by the operator on the test space.
 * @return auto The Linf projection operator.
 */
template <
   typename KernelPolicy,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename TrialIntegrationRule,
   typename TestIntegrationRule >
auto MakeLinfProjection(
   const TrialFiniteElementSpace & trial_finite_element_space,
   const TestFiniteElementSpace & test_finite_element_space,
   const TrialIntegrationRule & trial_int_rule,
   const TestIntegrationRule & test_int_rule )
{
   return LinfProjection<KernelPolicy,TrialFiniteElementSpace,TestFiniteElementSpace,TrialIntegrationRule,TestIntegrationRule>( trial_finite_element_space, test_finite_element_space, trial_int_rule, test_int_rule );
}

/**
 * @brief Factory to build Linf projection operators. Useful to hide the type of the operator.
 * 
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam TrialIntegrationRule The type of the integration rule used by the operator on the trial space.
 * @tparam TestIntegrationRule The type of the integration rule used by the operator on the test space.
 * @param trial_finite_element_space The trial finite element space associated to the operator.
 * @param test_finite_element_space The test finite element space associated to the operator.
 * @param trial_int_rule The integration rule used by the operator on the trial space.
 * @param test_int_rule The integration rule used by the operator on the test space.
 * @return auto The Linf projection operator.
 */
template <
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename TrialIntegrationRule,
   typename TestIntegrationRule >
auto MakeLinfProjection(
   const TrialFiniteElementSpace & trial_finite_element_space,
   const TestFiniteElementSpace & test_finite_element_space,
   const TrialIntegrationRule & trial_int_rule,
   const TestIntegrationRule & test_int_rule )
{
   using KernelPolicy = SerialKernelConfiguration;

   return LinfProjection<KernelPolicy,TrialFiniteElementSpace,TestFiniteElementSpace,TrialIntegrationRule,TestIntegrationRule>( trial_finite_element_space, test_finite_element_space, trial_int_rule, test_int_rule );
}

}

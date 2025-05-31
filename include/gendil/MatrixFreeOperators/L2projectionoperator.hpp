// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"
#include "gendil/Solvers/conjugategradient.hpp"

namespace gendil {

// !FIXME This is really a subspace L2 projection
/**
 * @brief Implementation of L2 projection at the element level.
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
void L2ProjectionElementOperator(
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

   const auto cell = trial_fe_space.GetCell( trial_element_index );

   // Container to store values at all the quadrature points
   auto DBu = MakeQuadraturePointValuesContainer( kernel_conf, TrialIntegrationRule{} );

   // Application of the QFunction
   QuadraturePointLoop< TrialIntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      PhysicalCoordinates X;
      Jacobian J_mesh;

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      const Real detJ = Determinant( J_mesh );
      const Real weight = GetWeight( quad_index, trial_element_quad_data );

      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      const Real Du_q_w = weight * detJ * Bu_q;

      // auto sub_quad_index = quad_index.template Sub< 0, TestDim >();
      // WriteAddQuadratureLocalValues( kernel_conf, sub_quad_index, Du_q_w, DBu );
      WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q_w, DBu );
   } );

   auto DBu_low = AggregateDimensions< TrialIntegrationRule, TestIntegrationRule >( kernel_conf, DBu, std::make_index_sequence< TestDim >{} );

   // Application of the test functions
   auto BDBu = ApplyTestFunctions( kernel_conf, test_element_quad_data, DBu_low );

   // Assumes tensor mesh
   const auto test_element_index = trial_fe_space.template GetStructuredSubIndex< 0 >( trial_element_index );
   WriteAddDofs( kernel_conf, test_fe_space, test_element_index, BDBu, dofs_out );
}

template <
   typename IntegrationRule,
   typename KernelContext,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename ElementQuadData,
   typename DofsIn,
   typename DofsOut >
GENDIL_HOST_DEVICE
void UnitMassElementOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   const DofsIn & dofs_in,
   DofsOut & dofs_out )
{
   using Mesh = typename FiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   auto Bu = InterpolateValues( kernel_conf, element_quad_data, dofs_in );

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


      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      const Real Du_q_w = weight * detJ * Bu_q;

      WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q_w, DBu );
   } );

   // Application of the test functions
   dofs_out = ApplyTestFunctions( kernel_conf, element_quad_data, DBu );
}

/**
 * @brief Gloabl L2 projection operator for DG fields.
 * 
 * @tparam TrialKernelConfiguration The execution policy for the hardware on the trial space.
 * @tparam TestKernelConfiguration The execution policy for the hardware on the test space.
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
   typename TrialKernelConfiguration,
   typename TestKernelConfiguration,
   typename TrialIntegrationRule,
   typename TestIntegrationRule,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename TrialMeshQuadData,
   typename TestMeshQuadData,
   typename TrialElementQuadData,
   typename TestElementQuadData >
void L2ProjectionOperator(
   const TrialFiniteElementSpace & trial_fe_space,
   const TestFiniteElementSpace & test_fe_space,
   const TrialMeshQuadData & trial_mesh_quad_data,
   const TestMeshQuadData & test_mesh_quad_data,
   const TrialElementQuadData & trial_element_quad_data,
   const TestElementQuadData & test_element_quad_data,
   const StridedView< TrialFiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< TestFiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   mesh::CellIterator< TrialKernelConfiguration >(
      trial_fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem =
            Max(
               required_shared_memory_v< TrialKernelConfiguration, TrialIntegrationRule >, // Interpolation
               required_shared_memory_v< TrialKernelConfiguration, TestIntegrationRule >, // Interpolation
               Product( typename TestIntegrationRule::points::num_points_tensor{} ) // Accumulation at quadrature point
            );
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< TrialKernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         L2ProjectionElementOperator< TrialIntegrationRule, TestIntegrationRule >(
            kernel_conf,
            trial_fe_space,
            test_fe_space,
            element_index,
            trial_mesh_quad_data,
            trial_element_quad_data,
            test_element_quad_data,
            dofs_in,
            dofs_out );
      }
   );
   mesh::CellIterator< TestKernelConfiguration >(
      test_fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = required_shared_memory_v< TestKernelConfiguration, TestIntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< TestKernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );
      
         auto op = [&]( const auto & in, auto & out )
         {
            UnitMassElementOperator< TestIntegrationRule >(
               kernel_conf,
               test_fe_space,
               element_index,
               test_mesh_quad_data,
               test_element_quad_data,
               in,
               out );
         };
         
         auto rhs = MakeThreadedView( kernel_conf, test_fe_space, ReadDofs( kernel_conf, test_fe_space, element_index, dofs_out ) );
         decltype(rhs) x{};

         Integer max_iters = 1000;
         Real tolerance = 1e-10;
         ConjugateGradient( kernel_conf, op, rhs, max_iters, tolerance, x );
         // auto result = ConjugateGradient( op, rhs, max_iters, tolerance, x );
         // std::cout << "Element " << element_index.linear_index << ": ";
         // if ( std::get< 0 >( result ) )
         // {
         //    std::cout << " SUCCESS, " << std::get< 1 >( result ) << " iterations " << std::endl;
         //    ElementDoF< TestFiniteElementSpace > y;
         //    op( x, y );
         //    Real norm = Norml2( y - rhs );
         //    std:: cout << "norm y=" << Norml2( y ) << std::endl;
         //    std:: cout << "norm rhs=" << Norml2( rhs ) << std::endl;
         //    std:: cout << "norm diff=" << norm << std::endl;
         // }
         // else
         // {
         //    std::cout << " FAILED!!! " << std::get< 1 >( result ) << " iterations " << std::endl;
         // }

         WriteDofs( kernel_conf, test_fe_space, element_index, x, dofs_out );
      }
   );
}

/**
 * @brief Represent an L2 projection operrator.
 * 
 * @tparam TrialKernelConfiguration The execution policy for the hardware on the inputs.
 * @tparam TestKernelConfiguration The execution policy for the hardware on the ouputs.
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam TrialIntegrationRule The integration rule used by the operator.
 * @tparam TestIntegrationRule The integration rule used by the operator.
 * @param fe_space The finite element space.
 */
template <
   typename TrialKernelConfiguration,
   typename TestKernelConfiguration,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename TrialIntegrationRule,
   typename TestIntegrationRule >
class L2Projection
#ifdef GENDIL_USE_MFEM
: public mfem::Operator
#endif // GENDIL_USE_MFEM
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

   using TrialMesh = typename TrialFiniteElementSpace::mesh_type;
   using TestMesh = typename TestFiniteElementSpace::mesh_type;

   /**
    * @brief Data structure storing information to compute physical coordinates
    * and mesh jacobians at the provided integration rule quadrature points.
    * 
    */
   using TrialMeshQuadData = typename TrialMesh::cell_type::template QuadData< TrialIntegrationRule >;
   TrialMeshQuadData trial_mesh_quad_data;

   using TestMeshQuadData = typename TestMesh::cell_type::template QuadData< TrialIntegrationRule >;
   TestMeshQuadData test_mesh_quad_data;

   /**
    * @brief Data structure storing information to compute field values
    * and field gradients on the trial finite element space at the provided
    * integration rule quadrature points.
    * 
    */
   using TrialElementQuadData = decltype(
                              MakeDofToQuad<
                                 typename trial_finite_element_type::shape_functions,
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
                                 typename test_finite_element_type::shape_functions,
                                 test_integration_rule
                              >()
                           );
   TestElementQuadData test_element_quad_data;

   using input = StridedView< TrialFiniteElementSpace::Dim + 1, const Real >;
   using output = StridedView< TestFiniteElementSpace::Dim + 1, Real >;

public:
   /**
    * @brief Construct a new L2Projection object.
    * 
    * @param trial_finite_element_space The trial finite element space associated to the operator.
    * @param test_finite_element_space The test finite element space associated to the operator.
    * @param trial_int_rule The integration rule used by the operator.
    * @param test_int_rule The integration rule used by the operator.
    */
   L2Projection(
      const TrialFiniteElementSpace & trial_finite_element_space,
      const TestFiniteElementSpace & test_finite_element_space,
      const TrialIntegrationRule & trial_int_rule,
      const TestIntegrationRule & test_int_rule ) :
#ifdef GENDIL_USE_MFEM
         mfem::Operator( test_finite_element_space.GetNumberOfFiniteElementDofs(), trial_finite_element_space.GetNumberOfFiniteElementDofs() ),
#endif
         trial_finite_element_space( trial_finite_element_space ),
         test_finite_element_space( test_finite_element_space ),
         trial_int_rule( trial_int_rule ),
         test_int_rule( test_int_rule ),
         trial_mesh_quad_data{},
         test_mesh_quad_data{},
         trial_element_quad_data{},
         test_element_quad_data{}
   {
      static_assert(
         std::is_same_v< typename TrialFiniteElementSpace::restriction_type, L2Restriction > &&
         std::is_same_v< typename TestFiniteElementSpace::restriction_type, L2Restriction >,
         "L2 projection is only supported for broken Sobolev spaces (DG spaces)."
      );
   }

   /**
    * @brief Apply the L2 projection operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      // TODO: dofs_out = 0.0
      L2ProjectionOperator< TrialKernelConfiguration, TestKernelConfiguration, trial_integration_rule, test_integration_rule >(
         this->trial_finite_element_space,
         this->test_finite_element_space,
         this->trial_mesh_quad_data,
         this->test_mesh_quad_data,
         this->trial_element_quad_data,
         this->test_element_quad_data,
         dofs_in,
         dofs_out );
   }

   void operator()( const Vector & dofs_vector_in, Vector & dofs_vector_out ) const
   {
      dofs_vector_out = 0.0;
      auto dofs_in = MakeReadOnlyEVectorView< TrialKernelConfiguration >( this->trial_finite_element_space, dofs_vector_in );
      auto dofs_out = MakeReadWriteEVectorView< TestKernelConfiguration >( this->test_finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }

   #ifdef GENDIL_USE_MFEM
   /**
    * @brief Apply the L2 projection operator.
    * 
    * @param dofs_vector_in The input degrees of freedom.
    * @param dofs_vector_out The output degrees of freedom.
    */
   void Mult( const mfem::Vector & dofs_vector_in,
              mfem::Vector & dofs_vector_out ) const override
   {
      dofs_vector_out = 0.0;
      auto dofs_in = MakeReadOnlyEVectorView< TrialKernelConfiguration >( this->trial_finite_element_space, dofs_vector_in );
      auto dofs_out = MakeReadWriteEVectorView< TestKernelConfiguration >( this->test_finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }
   #endif // GENDIL_USE_MFEM
};

/**
 * @brief Factory to build L2 projection operators. Useful to hide the type of the operator.
 * 
 * @tparam TrialKernelPolicy The execution policy for the hardware on the inputs.
 * @tparam TestKernelPolicy The execution policy for the hardware on the outputs.
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam TrialIntegrationRule The type of the integration rule used by the operator on the trial space.
 * @tparam TestIntegrationRule The type of the integration rule used by the operator on the test space.
 * @param trial_finite_element_space The trial finite element space associated to the operator.
 * @param test_finite_element_space The test finite element space associated to the operator.
 * @param trial_int_rule The integration rule used by the operator on the trial space.
 * @param test_int_rule The integration rule used by the operator on the test space.
 * @return auto The L2 projection operator.
 */
template <
   typename TrialKernelPolicy,
   typename TestKernelPolicy,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename TrialIntegrationRule,
   typename TestIntegrationRule >
auto MakeL2Projection(
   const TrialFiniteElementSpace & trial_finite_element_space,
   const TestFiniteElementSpace & test_finite_element_space,
   const TrialIntegrationRule & trial_int_rule,
   const TestIntegrationRule & test_int_rule )
{
   return L2Projection<TrialKernelPolicy,TestKernelPolicy,TrialFiniteElementSpace,TestFiniteElementSpace,TrialIntegrationRule,TestIntegrationRule>( trial_finite_element_space, test_finite_element_space, trial_int_rule, test_int_rule );
}

/**
 * @brief Factory to build L2 projection operators. Useful to hide the type of the operator.
 * 
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam TrialIntegrationRule The type of the integration rule used by the operator on the trial space.
 * @tparam TestIntegrationRule The type of the integration rule used by the operator on the test space.
 * @param trial_finite_element_space The trial finite element space associated to the operator.
 * @param test_finite_element_space The test finite element space associated to the operator.
 * @param trial_int_rule The integration rule used by the operator on the trial space.
 * @param test_int_rule The integration rule used by the operator on the test space.
 * @return auto The L2 projection operator.
 */
template <
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename TrialIntegrationRule,
   typename TestIntegrationRule >
auto MakeL2Projection(
   const TrialFiniteElementSpace & trial_finite_element_space,
   const TestFiniteElementSpace & test_finite_element_space,
   const TrialIntegrationRule & trial_int_rule,
   const TestIntegrationRule & test_int_rule )
{
   using TrialKernelPolicy = SerialKernelConfiguration;
   using TestKernelPolicy = SerialKernelConfiguration;
   
   return L2Projection<TrialKernelPolicy,TestKernelPolicy,TrialFiniteElementSpace,TestFiniteElementSpace,TrialIntegrationRule,TestIntegrationRule>( trial_finite_element_space, test_finite_element_space, trial_int_rule, test_int_rule );
}

}

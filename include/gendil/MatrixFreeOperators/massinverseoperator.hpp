// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/Utilities/View/threadedview.hpp"

namespace gendil {

/**
 * @brief Implementation of the "volume" contributions of the (inverse) Mass operator at the element level.
 * 
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam KernelContext Contextual information for the kernel.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
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
   typename KernelContext,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename ElementQuadData,
   typename Sigma,
   typename DofsIn,
   typename DofsOut >
GENDIL_HOST_DEVICE
void MassInvElementOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   Sigma & sigma,
   const DofsIn & dofs_in,
   DofsOut & dofs_out )
{
   using Mesh = typename FiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   auto Bu = InterpolateValues( kernel_conf, element_quad_data, dofs_in );

   const auto cell = fe_space.GetCell( element_index );

   // Container to store values at all the quadrature points
   auto& DBu = Bu;

   // Application of the QFunction
   QuadraturePointLoop< IntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      PhysicalCoordinates X;
      Jacobian J_mesh;

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      Jacobian inv_J_mesh;
      const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J_mesh );

      const Real weight = GetWeight( quad_index, element_quad_data );

      const Real D_Mass = sigma( X );

      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      const Real Du_q = D_Mass * Bu_q;

      const Real Du_q_w = weight * detJ * Du_q;

      WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q_w, DBu );
   } );

   // Application of the test functions
   dofs_out = ApplyTestFunctions( kernel_conf, element_quad_data, DBu );
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
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename ElementQuadData,
   typename Sigma >
void MassInverseExplicitOperator(
   const FiniteElementSpace & fe_space,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   Sigma & sigma,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   mesh::CellIterator< KernelConfiguration >(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         auto op = [&]( const auto & in, auto & out )
         {
           MassInvElementOperator< IntegrationRule >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            element_quad_data,
            sigma,
            in,
            out );
         };
         
         auto rhs = MakeThreadedView( kernel_conf, fe_space, ReadDofs( kernel_conf, fe_space, element_index, dofs_in ) );
         decltype(rhs) x{};

         Integer max_iters = 10000;
         Real tolerance = Real(1e-14);
         ConjugateGradient( kernel_conf, op, rhs, max_iters, tolerance, x );
         // std:: cout << "norm x=" << Norml2( x ) << std::endl;
         // auto result = ConjugateGradient( op, rhs, max_iters, tolerance, x );
         // std::cout << "Element " << element_index << ": ";
         // if ( std::get< 0 >( result ) )
         // {
         //    std::cout << " SUCCESS, " << std::get< 1 >( result ) << " iterations " << std::endl;
         //    ElementDoF< FiniteElementSpace > y;
         //    op( x, y );
         //    Real norm = Norml2( y - rhs );
         //    std:: cout << "norm y=" << Norml2( y ) << std::endl;
         //    std:: cout << "norm rhs=" << Norml2( rhs ) << std::endl;
         //    std:: cout << "norm diff=" << norm << std::endl;
         //    std:: cout << "norm x=" << Norml2( x ) << std::endl;
         // }
         // else
         // {
         //    std::cout << " FAILED!!! " << std::get< 1 >( result ) << " iterations " << std::endl;
         //    ElementDoF< FiniteElementSpace > y;
         //    op( x, y );
         //    Real norm = Norml2( y - rhs );
         //    std:: cout << "norm y=" << Norml2( y ) << std::endl;
         //    std:: cout << "norm rhs=" << Norml2( rhs ) << std::endl;
         //    std:: cout << "norm diff=" << norm << std::endl;
         //    std:: cout << "norm x=" << Norml2( x ) << std::endl;
         // }

         WriteDofs( kernel_conf, fe_space, element_index, x, dofs_out );
      }
   );
   
}

/**
 * @brief Represent a Mass finite element operrator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Sigma >
class MassInverseFiniteElementOperator
   : public MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >
{
   using base = MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >;
   Sigma & sigma;

   using input = StridedView< FiniteElementSpace::Dim + 1, const Real >;
   using output = StridedView< FiniteElementSpace::Dim + 1, Real >;

public:
   /**
    * @brief Construct a new MassFiniteElementOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    * @param sigma The function to evaluate the mass density at physical coordinates.
    */
   MassInverseFiniteElementOperator( const FiniteElementSpace & finite_element_space,
                              const IntegrationRule & int_rules,
                              Sigma & sigma ) :
      base( finite_element_space, int_rules ),
      sigma( sigma )
   { }

   /**
    * @brief Apply the mass operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      MassInverseExplicitOperator< KernelPolicy, typename base::integration_rule >
         ( this->finite_element_space,
           this->mesh_quad_data,
           this->element_quad_data,
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
    * @brief Apply the mass operator.
    * 
    * @param dofs_vector_in The input degrees of freedom.
    * @param dofs_vector_out The output degrees of freedom.
    */
   void Mult( const mfem::Vector & dofs_vector_in,
              mfem::Vector & dofs_vector_out ) const override
   {
      auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
      auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }
   #endif // GENDIL_USE_MFEM
};

/**
 * @brief Factory to build Mass operators. Useful to hide the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @return auto The Mass operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Sigma >
auto MakeMassInverseFiniteElementOperator(
   const FiniteElementSpace & finite_element_space,
   const IntegrationRule & int_rule,
   Sigma & sigma )
{
   return MassInverseFiniteElementOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule,
             Sigma
          >( finite_element_space, int_rule, sigma );
}

/**
 * @brief Factory to build Mass operators. Useful to hide the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @return auto The Mass operator.
 */
template <
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Sigma >
auto MakeMassInverseFiniteElementOperator(
   const FiniteElementSpace & finite_element_space,
   const IntegrationRule & int_rule,
   Sigma & sigma )
{
   using KernelPolicy = SerialKernelConfiguration;

   return MassInverseFiniteElementOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule,
             Sigma
          >( finite_element_space, int_rule, sigma );
}

}

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
   typename DofsInView,
   typename DofsOutView >
GENDIL_HOST_DEVICE
void MassElementOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   Sigma & sigma,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out )
{
   using Mesh = typename FiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   auto Bu = InterpolateValues( kernel_conf, element_quad_data, u ); // TODO: Change type of Bu to work with quad_index
   // TODO: Do a threaded wraper?

   const auto cell = fe_space.GetCell( element_index );

   // TODO Pass the integration rule as a variable?
   // Container to store values at all the quadrature points
   auto & DBu = Bu;

   // Application of the QFunction
   QuadraturePointLoop< IntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      PhysicalCoordinates X;
      Jacobian J_mesh;

      const auto Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      // TODO Vector Finite Element to compute X and J_Mesh
      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );
      // TODO X( kernel_conf, quad_index ) and J_mesh( kernel_conf, quad_index )

      const Real detJ = Determinant( J_mesh );
      const Real weight = GetWeight( quad_index, element_quad_data );
      const Real D_Mass = sigma( X );

      const auto Du_q = weight * detJ * D_Mass * Bu_q;

      WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q, DBu );
   } );

   // Application of the test functions
   auto BDBu = ApplyTestFunctions( kernel_conf, element_quad_data, DBu );

   WriteDofs( kernel_conf, fe_space, element_index, BDBu, dofs_out );
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
   typename Sigma,
   typename DofsInView,
   typename DofsOutView >
void MassExplicitOperator(
   const FiniteElementSpace & fe_space,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   Sigma & sigma,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out )
{
   mesh::CellIterator< KernelConfiguration >(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem =
            Max(
               required_shared_memory_v< KernelConfiguration, IntegrationRule >,
               FiniteElementSpace::finite_element_type::GetNumDofs()
            );
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         MassElementOperator< IntegrationRule >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            element_quad_data,
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
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Sigma >
class MassFiniteElementOperator
   : public MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >
{
   using base = MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >;
   Sigma & sigma;

public:
   /**
    * @brief Construct a new MassFiniteElementOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    * @param sigma The function to evaluate the mass density at physical coordinates.
    */
   MassFiniteElementOperator( const FiniteElementSpace & finite_element_space,
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
   template < typename input, typename output >
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      MassExplicitOperator< KernelPolicy, typename base::integration_rule >
         ( this->finite_element_space,
           this->mesh_quad_data,
           this->element_quad_data,
           sigma,
           dofs_in,
           dofs_out );
   }

   void operator()( const Vector & dofs_vector_in, Vector & dofs_vector_out ) const
   {
      if constexpr ( std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction > )
      {
         auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
         auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );
         Apply( dofs_in, dofs_out );
      }
      else // H1
      {
         dofs_vector_out = 0.0;
         auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
         auto dofs_out = MakeReadWriteEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );
         Apply( dofs_in, dofs_out );
      }
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
      if constexpr ( std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction > )
      {
         auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
         auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );
         Apply( dofs_in, dofs_out );
      }
      else // H1
      {
         dofs_vector_out = 0.0;
         auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
         auto dofs_out = MakeReadWriteEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );
         Apply( dofs_in, dofs_out );
      }
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
auto MakeMassFiniteElementOperator( const FiniteElementSpace & finite_element_space,
                                    const IntegrationRule & int_rule,
                                    Sigma & sigma )
{
   return MassFiniteElementOperator<
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
template < typename FiniteElementSpace, typename IntegrationRule, typename Sigma >
auto MakeMassFiniteElementOperator( const FiniteElementSpace & finite_element_space,
                                    const IntegrationRule & int_rule,
                                    Sigma & sigma )
{
   using KernelPolicy = SerialKernelConfiguration;

   return MassFiniteElementOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule,
             Sigma
          >( finite_element_space, int_rule, sigma );
}

}

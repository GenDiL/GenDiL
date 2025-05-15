// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

namespace gendil {

/**
 * @brief Implementation of the "volume" contributions of the "speed of light" operator at the element level.
 * 
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam KernelContext Contextual information for the kernel.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
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
   typename IntegrationRule,
   typename KernelContext,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename ElementQuadData >
GENDIL_HOST_DEVICE inline
void SoLElementOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   WriteDofs( kernel_conf, fe_space, element_index, u, dofs_out );
}

/**
 * @brief Explicit "speed of light" operator containing only the "volume" contributions.
 * 
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @param fe_space The finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename ElementQuadData >
void SoLExplicitOperator(
   const FiniteElementSpace & fe_space,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   mesh::CellIterator<KernelConfiguration>(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         SoLElementOperator< IntegrationRule >( 
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            element_quad_data,
            dofs_in,
            dofs_out );
      }
   );
}

/**
 * @brief Represent a "speed of light" finite element operrator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 */
template < typename KernelPolicy, typename FiniteElementSpace, typename IntegrationRule >
class SpeedOfLightOperator
   : public MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >
{
   using base = MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >;

   using input = StridedView< FiniteElementSpace::Dim + 1, const Real >;
   using output = StridedView< FiniteElementSpace::Dim + 1, Real >;

public:
   /**
    * @brief Construct a new SpeedOfLightOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    */
   SpeedOfLightOperator( const FiniteElementSpace & finite_element_space,
                         const IntegrationRule & int_rules ) :
      base( finite_element_space, int_rules )
   { }

   /**
    * @brief Apply the speed-of-light operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      SoLExplicitOperator< KernelPolicy, typename base::integration_rule >
         ( this->finite_element_space,
           this->mesh_quad_data,
           this->element_quad_data,
           dofs_in,
           dofs_out );
      // SoLExplicitOperator< KernelPolicy, typename base::integration_rule >
      //    ( this->finite_element_space,dofs_in,
      //      dofs_out );
   }

   void operator()( const Vector & dofs_vector_in, Vector & dofs_vector_out ) const
   {
      auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
      auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }

   #ifdef GENDIL_USE_MFEM
   /**
    * @brief Apply the speed-of-light operator.
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
 * @brief Factory to build speed-of-light operators. Useful to hide the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The speed-of-light operator.
 */
template < typename KernelPolicy, typename FiniteElementSpace, typename IntegrationRule >
auto MakeSpeedOfLightOperator( const FiniteElementSpace & finite_element_space,
                               const IntegrationRule & int_rule )
{
   return SpeedOfLightOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule );
}

/**
 * @brief Factory to build speed-of-light operators. Useful to hide the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The speed-of-light operator.
 */
template < typename FiniteElementSpace, typename IntegrationRule >
auto MakeSpeedOfLightOperator( const FiniteElementSpace & finite_element_space,
                               const IntegrationRule & int_rule )
{
   using KernelPolicy = SerialKernelConfiguration;

   return SpeedOfLightOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule );
}

}

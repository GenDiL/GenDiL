// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

namespace gendil {

/**
 * @brief Implementation of the "face" speed-of-light operator at the element level.
 * 
 * @tparam FaceIntegrationRulesTuple List of the types of integration rules used on each face of the element.
 * @tparam KernelContext The type of the kernel context.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam MeshFaceDofToQuad The type of the mesh data structure needed to perform computation at quadrature points on each face.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @tparam ElementFaceDofToQuad The type of the finite element data structure needed to perform computation at quadrature points on each face.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @param kernel_conf The kernel configuration.
 * @param fe_space The finite element space.
 * @param element_index The index of the finite element in the finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param mesh_face_quad_data The mesh data at quadrature points needed to perform the computation on each face.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param element_face_quad_data The finite element data at quadrature points need to perform the computation on each face.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename FaceIntegrationRulesTuple,
   typename KernelContext,
   typename FiniteElementSpace >
GENDIL_HOST_DEVICE
void FaceReadSpeedOfLightElementOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   FaceLoop( fe_space, element_index,
      [&]( auto const & face_info )
      {
         auto neighbor_u = ReadDofs( kernel_conf, fe_space, face_info, dofs_in );

         u += neighbor_u;
      }
   );

   WriteDofs( kernel_conf, fe_space, element_index, u, dofs_out );
}

template <
   typename FaceIntegrationRulesTuple,
   typename KernelContext,
   typename FiniteElementSpace >
GENDIL_HOST_DEVICE
void FaceWriteSpeedOfLightElementOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   // Read the current element DOFs
   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   FaceLoop( fe_space, element_index,
      [&]( auto const & face_info )
      {
         // Write the result back to the neighbor
         WriteAddDofs( kernel_conf, fe_space, face_info, u, dofs_out );
      }
   );

   // Write own contribution to own dofs_out
   WriteAddDofs( kernel_conf, fe_space, element_index, u, dofs_out );
}

/**
 * @brief Explicit speed-of-light "face" operator.
 * 
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam FaceIntegrationRulesTuple List of the types of integration rules used on each face of the element.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam MeshFaceDofToQuad The type of the mesh data structure needed to perform computation at quadrature points on each face.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @tparam ElementFaceDofToQuad The type of the finite element data structure needed to perform computation at quadrature points on each face.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @param fe_space The finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param mesh_face_quad_data The mesh data at quadrature points needed to perform the computation on each face.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param element_face_quad_data The finite element data at quadrature points need to perform the computation on each face.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   bool Read = true,
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FaceIntegrationRulesTuple,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename MeshFaceDofToQuad,
   typename ElementQuadData,
   typename ElementFaceDofToQuad >
void FaceSpeedOfLightExplicitOperator(
   const FiniteElementSpace & fe_space,
   const MeshQuadData & mesh_quad_data,
   const MeshFaceDofToQuad & mesh_face_quad_data,
   const ElementQuadData & element_quad_data,
   const ElementFaceDofToQuad & element_face_quad_data,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   if constexpr ( Read )
      mesh::CellIterator<KernelConfiguration>(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = FiniteElementSpace::finite_element_type::GetNumDofs();
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         FaceReadSpeedOfLightElementOperator< FaceIntegrationRulesTuple >(
            kernel_conf,
            fe_space,
            element_index,
            dofs_in,
            dofs_out );
      });
   else
      mesh::CellIterator<KernelConfiguration>(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = FiniteElementSpace::finite_element_type::GetNumDofs();
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         FaceWriteSpeedOfLightElementOperator< FaceIntegrationRulesTuple >(
            kernel_conf,
            fe_space,
            element_index,
            dofs_in,
            dofs_out );
      });
}

/**
 * @brief Represent a speed-of-light face operator.
 * 
 * @tparam Read Whether the operator reads or writes the neighbors degrees of freedom.
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 */
template <
   bool Read,
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule >
class FaceSpeedOfLightOperator
   : public MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >
{
   using base = MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >;

   using input = StridedView< FiniteElementSpace::Dim + 1, const Real >;
   using output = StridedView< FiniteElementSpace::Dim + 1, Real >;

public:
   /**
    * @brief Construct a new FaceSpeedOfLightOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    */
   FaceSpeedOfLightOperator( const FiniteElementSpace & finite_element_space,
                             const IntegrationRule & int_rules ) :
      base( finite_element_space, int_rules )
   { }

   /**
    * @brief Apply the speed-of-light face operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      FaceSpeedOfLightExplicitOperator< Read, KernelPolicy, typename base::integration_rule, typename base::face_integration_rules >
         ( this->finite_element_space,
           this->mesh_quad_data,
           this->mesh_face_quad_data,
           this->element_quad_data,
           this->element_face_quad_data,
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
    * @brief Apply the speed-of-light face operator.
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
 * @brief Factory to build speed-of-light face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The advection operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule >
auto MakeFaceSpeedOfLightOperator(
   const FiniteElementSpace & finite_element_space,
   const IntegrationRule & int_rule )
{
   return FaceSpeedOfLightOperator<
             true,
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule );
}

/**
 * @brief Factory to build speed-of-light face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The advection operator.
 */
template <
   typename FiniteElementSpace,
   typename IntegrationRule >
auto MakeFaceSpeedOfLightOperator(
   const FiniteElementSpace & finite_element_space,
   const IntegrationRule & int_rule )
{
   using KernelPolicy = SerialKernelConfiguration;

   return FaceSpeedOfLightOperator<
             true,
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule );
}

/**
 * @brief Factory to build speed-of-light face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The advection operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule >
auto MakeWriteFaceSpeedOfLightOperator(
   const FiniteElementSpace & finite_element_space,
   const IntegrationRule & int_rule )
{
   return FaceSpeedOfLightOperator<
             false,
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule );
}

/**
 * @brief Factory to build speed-of-light face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The advection operator.
 */
template <
   typename FiniteElementSpace,
   typename IntegrationRule >
auto MakeWriteFaceSpeedOfLightOperator(
   const FiniteElementSpace & finite_element_space,
   const IntegrationRule & int_rule )
{
   using KernelPolicy = SerialKernelConfiguration;

   return FaceSpeedOfLightOperator<
             false,
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule );
}

}

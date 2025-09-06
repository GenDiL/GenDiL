// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Interfaces/interfaces.hpp"
#include "gendil/NumericalIntegration/faceintegrationrules.hpp"
#include "gendil/Meshes/makefacequaddata.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

namespace gendil {

/**
 * @brief Implementation of the "volume" contributions of the diffusion operator at the element level.
 * 
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam KernelContext Contextual information for the kernel.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
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
   typename KernelContext,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename MeshQuadData,
   typename ElementQuadData,
   typename Velocity,
   typename DofsInView,
   typename DofsOutView >
GENDIL_HOST_DEVICE
void DiffusionElementOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const IntegrationRule & integration_rule,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   Velocity & velocity,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out )
{
   constexpr Integer Dim = FiniteElementSpace::Dim;

   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   auto Gu = InterpolateGradient( kernel_conf, element_quad_data, u );

   const auto cell = fe_space.GetCell( element_index );

   // Container to store values at all the quadrature points
   // !FIXME: not the right container for vector stuff
   auto DGuq = MakeQuadraturePointValuesContainer< Dim >( kernel_conf, integration_rule );

   // Application of the QFunction
   QuadraturePointLoop( kernel_conf, integration_rule, [&] ( auto const & quad_index )
   {
      constexpr Integer Dim = FiniteElementSpace::Dim;

      using Mesh = typename FiniteElementSpace::mesh_type;
      using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
      using Jacobian = typename Mesh::cell_type::jacobian;

      PhysicalCoordinates X;
      Jacobian J_mesh;

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      Jacobian inv_J_mesh;
      const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J_mesh );

      const Real weight = GetWeight( quad_index, element_quad_data );

      const Real D = velocity( X );

      // Compute: weight * detJ * D_A * J^-T * Gu_q
      Real Gu_q[Dim];
      ReadQuadratureLocalValues( kernel_conf, quad_index, Gu, Gu_q );

      ApplyMapping( inv_J_mesh, Gu_q );

      ApplyMappingTranspose( inv_J_mesh, Gu_q );
      const Real w = weight * detJ * D;
      for( size_t i = 0; i < Dim; i++)
      {
         Gu_q[i] = w * Gu_q[i];
      }

      WriteQuadratureLocalValues( kernel_conf, quad_index, Gu_q, DGuq );
   } );

   // Application of the test functions
   auto GDGu = ApplyGradientTestFunctionsAtQPoints( kernel_conf, element_quad_data, DGuq );
   auto BGDGu = ApplyTestFunctions( kernel_conf, element_quad_data, GDGu );

   WriteDofs( kernel_conf, fe_space, element_index, BGDGu, dofs_out );
}

/**
 * @brief Implementation of the "face" contributions of the SIPDG diffusion operator at the element level.
 * 
 * @tparam FaceIntegrationRulesTuple List of the types of integration rules used on each face of the element.
 * @tparam KernelContext Contextual information for the kernel.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam MeshFaceDofToQuad The type of the mesh data structure needed to perform computation at quadrature points on each face.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @tparam ElementFaceDofToQuad The type of the finite element data structure needed to perform computation at quadrature points on each face.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param fe_space The finite element space.
 * @param element_index The index of the finite element in the finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param mesh_face_quad_data The mesh data at quadrature points needed to perform the computation on each face.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param face_quad_data The finite element data at quadrature points need to perform the computation on each face.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename FaceIntegrationRulesTuple,
   typename MeshQuadData,
   typename MeshFaceDofToQuad,
   typename ElementQuadData,
   typename ElementFaceDofToQuad,
   typename Velocity,
   typename DofsInView,
   typename DofsOutView >
GENDIL_HOST_DEVICE
void DiffusionFaceOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const FaceIntegrationRulesTuple & face_integration_rules,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const MeshFaceDofToQuad & mesh_face_quad_data,
   const ElementQuadData & element_quad_data,
   const ElementFaceDofToQuad & face_quad_data,
   Velocity & velocity,
   const Real sigma,
   const Real kappa,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out )
{
   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   const auto cell = fe_space.GetCell( element_index );

   // Container to store values at all the quadrature points
   auto BfDBfu = MakeElementDoFValuesContainer( kernel_conf, fe_space );

   FaceLoop( fe_space, element_index,
      [&]( auto const & face_info )
      {
         constexpr Integer Dim = FiniteElementSpace::Dim;

         auto neighbor_u = ReadDofs( kernel_conf, fe_space, face_info, dofs_in );

         auto Bu = InterpolateValues( kernel_conf, face_info.minus_side(), face_quad_data, u );
         auto Gu = InterpolateGradient( kernel_conf, face_info.minus_side(), face_quad_data, u );

         auto neighbor_Bu = InterpolateValues( kernel_conf, face_info.plus_side(), face_quad_data, neighbor_u );
         auto neighbor_Gu = InterpolateGradient( kernel_conf, face_info.plus_side(), face_quad_data, neighbor_u );

         auto face_int_rule = GetFaceIntegrationRule( face_info.minus_side(), face_integration_rules );
         auto Duq = MakeQuadraturePointValuesContainer( kernel_conf, face_int_rule );
         auto DGuq = MakeQuadraturePointValuesContainer<Dim>( kernel_conf, face_int_rule );

         QuadraturePointLoop( kernel_conf, face_int_rule,
            [&] ( auto const & quad_index )
            {
               using Mesh = typename FiniteElementSpace::mesh_type;
               using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
               using Jacobian = typename Mesh::cell_type::jacobian;
               
               PhysicalCoordinates X;
               Jacobian J_mesh;

               mesh::ComputePhysicalCoordinatesAndJacobian( cell, face_info.minus_side(), quad_index, mesh_face_quad_data, X, J_mesh );

               Jacobian inv_J;
               const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J );

               const Real weight = GetWeight( face_info.minus_side(), quad_index, face_quad_data );

               Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );
               Real Gu_q[ Dim ];
               ReadQuadratureLocalValues( kernel_conf, quad_index, Gu, Gu_q );
               ApplyMapping( inv_J, Gu_q );

               Real neighbor_Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, neighbor_Bu );
               Real neighbor_Gu_q[ Dim ];
               ReadQuadratureLocalValues( kernel_conf, quad_index, neighbor_Gu, neighbor_Gu_q );
               ApplyMapping( inv_J, neighbor_Gu_q );

               const auto reference_normal = GetReferenceNormal( face_info );
               const auto physical_normal = ComputePhysicalNormal( inv_J, reference_normal );
                
               const Real velocity_q = velocity( X );
               
               Real average_Gu_q[ Dim ];
               for (size_t i = 0; i < Dim; i++)
               {
                  average_Gu_q[i] = 0.5 * velocity_q * (Gu_q[i] + neighbor_Gu_q[i]);
               }

               Real jump = Bu_q - neighbor_Bu_q;
               const Real dot = Dot( average_Gu_q, physical_normal );
               const Real norm = Sqrt( Dot( physical_normal, physical_normal ) );
               const Real h_inv = norm;
               const Real Du_q = - weight * detJ * dot + weight * detJ * norm * kappa * h_inv * velocity_q * jump;

               Real DGu_q[Dim];
               const Real w = sigma * weight * detJ * jump * velocity_q * 0.5;
               for (size_t i = 0; i < Dim; i++)
               {
                  DGu_q[i] = w * physical_normal[i];
               }
               ApplyMappingTranspose( inv_J, DGu_q );

               WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q, Duq );
               WriteQuadratureLocalValues( kernel_conf, quad_index, DGu_q, DGuq );
            }
         );

         // Application of the test functions
         ApplyValuesAndGradientTestFunctions<true>( kernel_conf, face_info.minus_side(), face_quad_data, Duq, DGuq, BfDBfu );
      },
      [&]( auto const & face_info )
      {
         constexpr Integer Dim = FiniteElementSpace::Dim;

         auto Bu = InterpolateValues( kernel_conf, face_info.minus_side(), face_quad_data, u );
         auto Gu = InterpolateGradient( kernel_conf, face_info.minus_side(), face_quad_data, u );

         auto face_int_rule = GetFaceIntegrationRule( face_info.minus_side(), face_integration_rules );
         auto Duq = MakeQuadraturePointValuesContainer( kernel_conf, face_int_rule );
         auto DGuq = MakeQuadraturePointValuesContainer<Dim>( kernel_conf, face_int_rule );

         QuadraturePointLoop( kernel_conf, face_int_rule,
            [&] ( auto const & quad_index )
            {
               constexpr Integer Dim = FiniteElementSpace::Dim;

               using Mesh = typename FiniteElementSpace::mesh_type;
               using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
               using Jacobian = typename Mesh::cell_type::jacobian;
               
               PhysicalCoordinates X;
               Jacobian J_mesh;

               mesh::ComputePhysicalCoordinatesAndJacobian( cell, face_info.minus_side(), quad_index, mesh_face_quad_data, X, J_mesh );

               Jacobian inv_J;
               const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J );

               const Real weight = GetWeight( face_info.minus_side(), quad_index, face_quad_data );

               Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );
               Real Gu_q[ Dim ];
               ReadQuadratureLocalValues( kernel_conf, quad_index, Gu, Gu_q );
               ApplyMapping( inv_J, Gu_q );

               const auto reference_normal = GetReferenceNormal( face_info );
               const auto physical_normal = ComputePhysicalNormal( inv_J, reference_normal );
                
               const Real velocity_q = velocity( X );
               
               Real average_Gu_q[ Dim ];
               for (size_t i = 0; i < Dim; i++)
               {
                  average_Gu_q[i] = velocity_q * Gu_q[i];
               }

               Real jump = Bu_q;
               const Real dot = Dot( average_Gu_q, physical_normal );
               const Real norm = Sqrt( Dot( physical_normal, physical_normal ) );
               const Real h_inv = norm;
               const Real Du_q = - weight * detJ * dot + weight * detJ * norm * kappa * h_inv * velocity_q * jump;

               Real DGu_q[Dim];
               const Real w = sigma * weight * detJ * jump * velocity_q;
               for (size_t i = 0; i < Dim; i++)
               {
                  DGu_q[i] = w * physical_normal[i];
               }
               ApplyMappingTranspose( inv_J, DGu_q );

               WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q, Duq );
               WriteQuadratureLocalValues( kernel_conf, quad_index, DGu_q, DGuq );
            }
         );

         // Application of the test functions
         ApplyValuesAndGradientTestFunctions<true>( kernel_conf, face_info.minus_side(), face_quad_data, Duq, DGuq, BfDBfu );
      }
   );
   WriteAddDofs( kernel_conf, fe_space, element_index, BfDBfu, dofs_out );
}

/**
 * @brief Explicit SIPDG diffusion operator containing the "volume" and "face" contributions.
 * 
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam FaceIntegrationRulesTuple List of the types of integration rules used on each face of the element.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam MeshFaceDofToQuad The type of the mesh data structure needed to perform computation at quadrature points on each face.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @tparam ElementFaceDofToQuad The type of the finite element data structure needed to perform computation at quadrature points on each face.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param fe_space The finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param mesh_face_quad_data The mesh data at quadrature points needed to perform the computation on each face.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param face_quad_data The finite element data at quadrature points need to perform the computation on each face.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FaceIntegrationRulesTuple,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename MeshFaceDofToQuad,
   typename ElementQuadData,
   typename ElementFaceDofToQuad,
   typename Velocity,
   typename DofsInView,
   typename DofsOutView >
void DiffusionExplicitOperator(
   const FiniteElementSpace & fe_space,
   const MeshQuadData & mesh_quad_data,
   const MeshFaceDofToQuad & mesh_face_quad_data,
   const ElementQuadData & element_quad_data,
   const ElementFaceDofToQuad & face_quad_data,
   Velocity & velocity,
   const Real sigma,
   const Real kappa,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out )
{
   mesh::CellIterator< KernelConfiguration >(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         DiffusionElementOperator(
            kernel_conf,
            fe_space,
            IntegrationRule{},
            element_index,
            mesh_quad_data,
            element_quad_data,
            velocity,
            dofs_in,
            dofs_out );
         DiffusionFaceOperator(
            kernel_conf,
            fe_space,
            FaceIntegrationRulesTuple{},
            element_index,
            mesh_quad_data,
            mesh_face_quad_data,
            element_quad_data,
            face_quad_data,
            velocity,
            sigma,
            kappa,
            dofs_in,
            dofs_out );
      }
   );
}

/**
 * @brief Represent a diffusion finite element operrator.
 * 
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Velocity >
class DiffusionOperator
   : public MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >
{
   using base = MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >;
   Velocity & velocity;
   const Real sigma;
   const Real kappa;

public:
   /**
    * @brief Construct a new DiffusionOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    * @param sigma The function to evaluate the mass density at physical coordinates.
    */
   DiffusionOperator( const FiniteElementSpace & finite_element_space,
                      const IntegrationRule & int_rules,
                      Velocity & velocity,
                      const Real sigma,
                      const Real kappa ) :
      base( finite_element_space, int_rules ),
      velocity( velocity ),
      sigma( sigma ),
      kappa( kappa )
   { }

   /**
    * @brief Apply the diffusion operator.
    * 
    * @param dofs_vector_in The input degrees of freedom.
    * @param dofs_vector_out The output degrees of freedom.
    */
   template < typename input, typename output >
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      static_assert(
         std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction >,
         "DiffusionOperator::operator() only supports L2Restriction" );
      DiffusionExplicitOperator< KernelPolicy, typename base::integration_rule, typename base::face_integration_rules >
         ( this->finite_element_space,
           this->mesh_quad_data,
           this->mesh_face_quad_data,
           this->element_quad_data,
           this->element_face_quad_data,
           velocity,
           sigma,
           kappa,
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
    * @brief Apply the diffusion operator.
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
 * @brief Factory to build diffusion operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @return auto The diffusion operator.
 */
template < typename FiniteElementSpace,
           typename IntegrationRule,
           typename Velocity >
auto MakeDiffusionOperator( const FiniteElementSpace & finite_element_space,
                                const IntegrationRule & int_rule,
                                Velocity & velocity,
                                const Real sigma,
                                const Real kappa )
{
   using KernelPolicy = SerialKernelConfiguration;

   return DiffusionOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule,
             Velocity
          >( finite_element_space, int_rule, velocity, sigma, kappa );
}

/**
 * @brief Factory to build diffusion operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @return auto The diffusion operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Velocity >
auto MakeDiffusionOperator( const FiniteElementSpace & finite_element_space,
                                const IntegrationRule & int_rule,
                                Velocity & velocity,
                                const Real sigma,
                                const Real kappa )
{
   return DiffusionOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule,
             Velocity
          >( finite_element_space, int_rule, velocity, sigma, kappa );
}

}

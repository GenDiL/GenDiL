// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctions.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctions.hpp"

namespace gendil {

/**
 * @brief Implementation of the "face" contributions of the advection operator at the element level.
 * 
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam FaceIntegrationRulesTuple List of the types of integration rules used on each face of the element.
 * @tparam KernelContext Contextual information for the kernel.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam MeshFaceDofToQuad The type of the mesh data structure needed to perform computation at quadrature points on each face.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @tparam ElementFaceDofToQuad The type of the finite element data structure needed to perform computation at quadrature points on each face.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @param fe_space The finite element space.
 * @param element_index The index of the finite element in the finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param mesh_face_quad_data The mesh data at quadrature points needed to perform the computation on each face.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param element_face_quad_data The finite element data at quadrature points need to perform the computation on each face.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename IntegrationRule,
   typename FaceIntegrationRulesTuple,
   typename KernelContext,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename MeshFaceDofToQuad,
   typename ElementQuadData,
   typename ElementFaceDofToQuad,
   typename Adv,
   typename DofsInView,
   typename DofsOutView >
GENDIL_HOST_DEVICE
void AdvectionFusedOperatorWithoutBC(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const MeshFaceDofToQuad & mesh_face_quad_data,
   const ElementQuadData & element_quad_data,
   const ElementFaceDofToQuad & element_face_quad_data,
   Adv & adv,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out )
{
   using Mesh = typename FiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   constexpr Integer Dim = FiniteElementSpace::Dim;

   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   // Element contributions
   auto Bu = InterpolateValues( kernel_conf, element_quad_data, u );

   const auto cell = fe_space.GetCell( element_index );

   // Container to store values at all the quadrature points
   auto DBu = MakeQuadraturePointValuesContainer< Dim >( kernel_conf, IntegrationRule{} );

   // Application of the QFunction
   QuadraturePointLoop< IntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      constexpr Integer Dim = FiniteElementSpace::Dim;

      PhysicalCoordinates X;
      Jacobian J_mesh;

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      Jacobian inv_J_mesh;
      const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J_mesh );

      const Real weight = GetWeight( quad_index, element_quad_data );

      // Compute: weight * detJ * D_A * J^-T * Gu_q
      Real D_Advection[ Dim ];
      adv( X, D_Advection );

      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      ApplyMappingTranspose( inv_J_mesh, D_Advection );

      const Real w = - weight * detJ * Bu_q;
      Real DBu_q[Dim];
      for( size_t i = 0; i < Dim; i++)
      {
         DBu_q[i] = w * D_Advection[i];
      }

      WriteQuadratureLocalValues( kernel_conf, quad_index, DBu_q, DBu );
   } );

   // Application of the test functions
   auto GDBu = ApplyGradientTestFunctionsAtQPoints( kernel_conf, element_quad_data, DBu );

   auto BGDBu = ApplyTestFunctions( kernel_conf, element_quad_data, GDBu );

   // Face contributions
   FaceLoop( fe_space, element_index,
      [&]( auto const & face_info )
      {
         using FaceType = std::remove_reference_t< decltype( face_info ) >;

         auto neighbor_u = ReadDofs( kernel_conf, fe_space, face_info, dofs_in );

         constexpr Integer face_index = FaceType::local_face_index;
         constexpr Integer neighbor_face_index = FaceType::neighbor_local_face_index;
         using FaceIntegrationRule = std::tuple_element_t< face_index, FaceIntegrationRulesTuple >;

         auto Bu = InterpolateValues( kernel_conf, std::get< face_index >( element_face_quad_data ), u );

         auto neighbor_Bu = InterpolateValues( kernel_conf, std::get< neighbor_face_index >( element_face_quad_data ), neighbor_u );

         auto & DBu = Bu;

         QuadraturePointLoop< FaceIntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
            {
               constexpr Integer Dim = FiniteElementSpace::Dim;

               using Mesh = typename FiniteElementSpace::mesh_type;
               using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
               using Jacobian = typename Mesh::cell_type::jacobian;
               
               PhysicalCoordinates X;
               Jacobian J_mesh;

               cell.GetValuesAndJacobian( quad_index, std::get< face_index >( mesh_face_quad_data ), X, J_mesh );

               Jacobian inv_J;
               const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J );

               const Real weight = GetWeight( quad_index, std::get< face_index >( element_face_quad_data ) );

               Real D_Advection[ Dim ];
               adv( X, D_Advection );

               const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

               const Real neighbor_Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, neighbor_Bu );

               const auto reference_normal = GetReferenceNormal( face_info );
               const auto physical_normal = ComputePhysicalNormal( inv_J, reference_normal );

               const Real dot = Dot( D_Advection, physical_normal );
               const Real upwind = dot > 0.0 ? dot * Bu_q : dot * neighbor_Bu_q;

               const Real Du_q = weight * detJ * upwind;

               WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q, DBu );
            }
         );

         // Application of the test functions
         ApplyAddTestFunctions( kernel_conf, std::get< face_index >( element_face_quad_data ), DBu, BGDBu );
      }
   );
   WriteDofs( kernel_conf, fe_space, element_index, BGDBu, dofs_out );
}

/**
 * @brief Implementation of the "face" contributions of the advection operator at the element level.
 * 
 * @tparam IntegrationRule The type of the integration rule used by the element operator.
 * @tparam FaceIntegrationRulesTuple List of the types of integration rules used on each face of the element.
 * @tparam KernelContext Contextual information for the kernel.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam MeshFaceDofToQuad The type of the mesh data structure needed to perform computation at quadrature points on each face.
 * @tparam ElementQuadData The type of the finite element data structure needed to perform computation at quadrature points.
 * @tparam ElementFaceDofToQuad The type of the finite element data structure needed to perform computation at quadrature points on each face.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam BCType The advected field on the boundary conditions described by a function.
 * @param fe_space The finite element space.
 * @param element_index The index of the finite element in the finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param mesh_face_quad_data The mesh data at quadrature points needed to perform the computation on each face.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param element_face_quad_data The finite element data at quadrature points need to perform the computation on each face.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @param boundary_field The function to evaluate the advected field on the boundary.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename IntegrationRule,
   typename FaceIntegrationRulesTuple,
   typename KernelContext,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename MeshFaceDofToQuad,
   typename ElementQuadData,
   typename ElementFaceDofToQuad,
   typename Adv,
   typename BCType,
   typename DofsInView,
   typename DofsOutView >
GENDIL_HOST_DEVICE
void AdvectionFusedOperatorWithBC(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const MeshFaceDofToQuad & mesh_face_quad_data,
   const ElementQuadData & element_quad_data,
   const ElementFaceDofToQuad & element_face_quad_data,
   Adv & adv,
   BCType & boundary_field,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out )
{
   using Mesh = typename FiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   constexpr Integer Dim = FiniteElementSpace::Dim;

   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   // Element contributions
   auto Bu = InterpolateValues( kernel_conf, element_quad_data, u );

   const auto cell = fe_space.GetCell( element_index );

   // Container to store values at all the quadrature points
   auto DBu = MakeQuadraturePointValuesContainer< Dim >( kernel_conf, IntegrationRule{} );

   // Application of the QFunction
   QuadraturePointLoop< IntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      constexpr Integer Dim = FiniteElementSpace::Dim;

      PhysicalCoordinates X;
      Jacobian J_mesh;

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      Jacobian inv_J_mesh;
      const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J_mesh );

      const Real weight = GetWeight( quad_index, element_quad_data );

      // Compute: weight * detJ * D_A * J^-T * Gu_q
      Real D_Advection[ Dim ];
      adv( X, D_Advection );

      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      ApplyMappingTranspose( inv_J_mesh, D_Advection );

      const Real w = - weight * detJ * Bu_q;
      Real DBu_q[Dim];
      for( size_t i = 0; i < Dim; i++)
      {
         DBu_q[i] = w * D_Advection[i];
      }

      WriteQuadratureLocalValues( kernel_conf, quad_index, DBu_q, DBu );
   } );

   // Application of the test functions
   auto GDBu = ApplyGradientTestFunctionsAtQPoints( kernel_conf, element_quad_data, DBu );

   auto BGDBu = ApplyTestFunctions( kernel_conf, element_quad_data, GDBu );

   // Face contributions
   FaceLoop( fe_space, element_index,
      [&]( auto const & face_info )
      {
         using FaceType = std::remove_reference_t< decltype( face_info ) >;

         auto neighbor_u = ReadDofs( kernel_conf, fe_space, face_info, dofs_in );

         constexpr Integer face_index = FaceType::local_face_index;
         constexpr Integer neighbor_face_index = FaceType::neighbor_local_face_index;
         using FaceIntegrationRule = std::tuple_element_t< face_index, FaceIntegrationRulesTuple >;

         auto Bu = InterpolateValues( kernel_conf, std::get< face_index >( element_face_quad_data ), u );

         auto neighbor_Bu = InterpolateValues( kernel_conf, std::get< neighbor_face_index >( element_face_quad_data ), neighbor_u );

         auto & DBu = Bu;
      
         QuadraturePointLoop< FaceIntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
            {
               constexpr Integer Dim = FiniteElementSpace::Dim;

               using Mesh = typename FiniteElementSpace::mesh_type;
               using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
               using Jacobian = typename Mesh::cell_type::jacobian;
               
               PhysicalCoordinates X;
               Jacobian J_mesh;

               cell.GetValuesAndJacobian( quad_index, std::get< face_index >( mesh_face_quad_data ), X, J_mesh );

               Jacobian inv_J;
               const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J );

               const Real weight = GetWeight( quad_index, std::get< face_index >( element_face_quad_data ) );

               Real D_Advection[ Dim ];
               adv( X, D_Advection );

               const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

               const Real neighbor_Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, neighbor_Bu );

               const auto reference_normal = GetReferenceNormal( face_info );
               const auto physical_normal = ComputePhysicalNormal( inv_J, reference_normal );

               const Real dot = Dot( D_Advection, physical_normal );
               const Real upwind = ( dot > 0.0  ) ? ( dot * Bu_q ) : ( dot * neighbor_Bu_q );

               const Real Du_q = weight * detJ * upwind;

               WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q, DBu );
            }
         );

         // Application of the test functions
         ApplyAddTestFunctions( kernel_conf, std::get< face_index >( element_face_quad_data ), DBu, BGDBu );
      },
      [&]( auto const & face_info )
      {
         using FaceType = std::remove_reference_t< decltype( face_info ) >;

         constexpr Integer face_index = FaceType::local_face_index;
         using FaceIntegrationRule = std::tuple_element_t< face_index, FaceIntegrationRulesTuple >;

         auto Bu = InterpolateValues( kernel_conf, std::get< face_index >( element_face_quad_data ), u );

         auto & DBu = Bu;

         QuadraturePointLoop< FaceIntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
            {
               constexpr Integer Dim = FiniteElementSpace::Dim;

               using Mesh = typename FiniteElementSpace::mesh_type;
               using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
               using Jacobian = typename Mesh::cell_type::jacobian;
               
               PhysicalCoordinates X;
               Jacobian J_mesh;

               cell.GetValuesAndJacobian( quad_index, std::get< face_index >( mesh_face_quad_data ), X, J_mesh );

               Jacobian inv_J;
               const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J );

               const Real weight = GetWeight( quad_index, std::get< face_index >( element_face_quad_data ) );

               Real D_Advection[ Dim ];
               adv( X, D_Advection );

               const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

               const Real neighbor_Bu_q = boundary_field( X );

               const auto reference_normal = GetReferenceNormal( face_info );
               const auto physical_normal = ComputePhysicalNormal( inv_J, reference_normal );

               const Real dot = Dot( D_Advection, physical_normal );
               const Real upwind = dot > 0.0 ? dot * Bu_q : dot * neighbor_Bu_q;

               const Real Du_q = weight * detJ * upwind;

               WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q, DBu );
            }
         );

         // Application of the test functions
         ApplyAddTestFunctions( kernel_conf, std::get< face_index >( element_face_quad_data ), DBu, BGDBu );
      }
   );
   WriteDofs( kernel_conf, fe_space, element_index, BGDBu, dofs_out );
}

/**
 * @brief Explicit advection operator containing the "volume" and "face" contributions.
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
 * @param adv The function to evaluate the advection vector at physical coordinates.
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
   typename Adv,
   typename DofsInView,
   typename DofsOutView >
void AdvectionExplicitOperatorWithoutBC(
   const FiniteElementSpace & fe_space,
   const MeshQuadData & mesh_quad_data,
   const MeshFaceDofToQuad & mesh_face_quad_data,
   const ElementQuadData & element_quad_data,
   const ElementFaceDofToQuad & element_face_quad_data,
   Adv adv,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out )
{
   mesh::CellIterator<KernelConfiguration>(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         AdvectionFusedOperatorWithoutBC< IntegrationRule, FaceIntegrationRulesTuple >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            mesh_face_quad_data,
            element_quad_data,
            element_face_quad_data,
            adv,
            dofs_in,
            dofs_out );
      }
   );
}

/**
 * @brief Explicit advection operator containing the "volume" and "face" contributions.
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
 * @tparam BCType The advected field on the boundary conditions described by a function.
 * @param fe_space The finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param mesh_face_quad_data The mesh data at quadrature points needed to perform the computation on each face.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param element_face_quad_data The finite element data at quadrature points need to perform the computation on each face.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @param boundary_field The function to evaluate the advected field on the boundary.
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
   typename Adv,
   typename BCType,
   typename DofsInView,
   typename DofsOutView >
void AdvectionExplicitOperatorWithBC(
   const FiniteElementSpace & fe_space,
   const MeshQuadData & mesh_quad_data,
   const MeshFaceDofToQuad & mesh_face_quad_data,
   const ElementQuadData & element_quad_data,
   const ElementFaceDofToQuad & element_face_quad_data,
   Adv adv,
   BCType & boundary_field,
   const DofsInView dofs_in,
   DofsOutView & dofs_out )
{
   mesh::CellIterator<KernelConfiguration>(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         AdvectionFusedOperatorWithBC< IntegrationRule, FaceIntegrationRulesTuple >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            mesh_face_quad_data,
            element_quad_data,
            element_face_quad_data,
            adv,
            boundary_field,
            dofs_in,
            dofs_out );
      }
   );
}

/**
 * @brief Represent an advection finite element operrator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam BCType The advected field on the boundary conditions described by a function.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Adv,
   typename BCType >
class AdvectionOperator
   : public MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >
{
   using base = MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >;
   Adv adv;
   BCType boundary_field;

public:
   /**
    * @brief Construct a new AdvectionOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    * @param adv The function to evaluate the advection vector at physical coordinates.
    * @param boundary_field The function to evaluate the advected field on the boundary.
    */
   AdvectionOperator( const FiniteElementSpace & finite_element_space,
                      const IntegrationRule & int_rules,
                      Adv && adv,
                      BCType && boundary_field ) :
      base( finite_element_space, int_rules ),
      adv( adv ),
      boundary_field( boundary_field )
   { }

   /**
    * @brief Apply the advection operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   template < typename input, typename output >
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      static_assert(
         std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction >,
         "AdvectionOperator::operator() only supports L2Restriction" );
      if constexpr ( !std::is_same_v< BCType, Empty > )
      {
         AdvectionExplicitOperatorWithBC< KernelPolicy, typename base::integration_rule, typename base::face_integration_rules >(
            this->finite_element_space,
            this->mesh_quad_data,
            this->mesh_face_quad_data,
            this->element_quad_data,
            this->element_face_quad_data,
            adv,
            boundary_field,
            dofs_in,
            dofs_out );
      }
      else
      {
         AdvectionExplicitOperatorWithoutBC< KernelPolicy, typename base::integration_rule, typename base::face_integration_rules >(
            this->finite_element_space,
            this->mesh_quad_data,
            this->mesh_face_quad_data,
            this->element_quad_data,
            this->element_face_quad_data,
            adv,
            dofs_in,
            dofs_out );
      }
   }

   void operator()( const Vector & dofs_vector_in, Vector & dofs_vector_out ) const
   {
      auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
      auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }

   #ifdef GENDIL_USE_MFEM
   /**
    * @brief Apply the advection operator.
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
 * @brief Factory to build advection operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @return auto The advection operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Adv >
auto MakeAdvectionOperator( const FiniteElementSpace & finite_element_space,
                            const IntegrationRule & int_rule,
                            Adv && adv )
{
   return AdvectionOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule,
             Adv,
             Empty
          >( finite_element_space, int_rule, adv, Empty{} );
}

/**
 * @brief Factory to build advection operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @return auto The advection operator.
 */
template <
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Adv >
auto MakeAdvectionOperator( const FiniteElementSpace & finite_element_space,
                            const IntegrationRule & int_rule,
                            Adv && adv )
{
   using KernelPolicy = SerialKernelConfiguration;

   return AdvectionOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule,
             Adv,
             Empty
          >( finite_element_space, int_rule, adv, Empty{} );
}

/**
 * @brief Factory to build advection operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam BCType The advected field on the boundary conditions described by a function.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @param boundary_field The function to evaluate the advected field on the boundary.
 * @return auto The advection operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Adv,
   typename BCType >
auto MakeAdvectionOperator( const FiniteElementSpace & finite_element_space,
                            const IntegrationRule & int_rule,
                            Adv && adv,
                            BCType && boundary_field )
{
   return AdvectionOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule,
             Adv,
             BCType
          >( finite_element_space, int_rule, adv, boundary_field );
}

/**
 * @brief Factory to build advection operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam BCType The advected field on the boundary conditions described by a function.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @param boundary_field The function to evaluate the advected field on the boundary.
 * @return auto The advection operator.
 */
template <
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Adv,
   typename BCType >
auto MakeAdvectionOperator( const FiniteElementSpace & finite_element_space,
                            const IntegrationRule & int_rule,
                            Adv && adv,
                            BCType && boundary_field )
{
   using KernelPolicy = SerialKernelConfiguration;

   return AdvectionOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule,
             Adv,
             BCType
          >( finite_element_space, int_rule, adv, boundary_field );
}

}

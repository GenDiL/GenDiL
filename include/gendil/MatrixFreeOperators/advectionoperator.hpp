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
   typename KernelContext,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename FaceIntegrationRulesTuple,
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
   const IntegrationRule & integration_rule,
   const FaceIntegrationRulesTuple & face_integration_rules,
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
   auto DBu = MakeQuadraturePointValuesContainer< Dim >( kernel_conf, integration_rule );

   // Application of the QFunction
   QuadraturePointLoop( kernel_conf, integration_rule, [&] ( auto const & quad_index )
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
         auto neighbor_u = ReadDofs( kernel_conf, fe_space, face_info, dofs_in );

         auto Bu = InterpolateValues( kernel_conf, face_info.minus_side(), element_face_quad_data, u );

         auto neighbor_Bu = InterpolateValues( kernel_conf, face_info.plus_side(), element_face_quad_data, neighbor_u );

         auto & DBu = Bu;
         auto face_int_rule = GetFaceIntegrationRule( face_info.minus_side(), face_integration_rules );

         QuadraturePointLoop( kernel_conf, face_int_rule, [&] ( auto const & quad_index )
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

               const Real weight = GetWeight( face_info.minus_side(), quad_index, element_face_quad_data );

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
         ApplyAddTestFunctions( kernel_conf, face_info.minus_side(), element_face_quad_data, DBu, BGDBu );
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
   typename KernelContext,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename FaceIntegrationRulesTuple,
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
   const IntegrationRule & integration_rule,
   const FaceIntegrationRulesTuple & face_integration_rules,
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
   auto DBu = MakeQuadraturePointValuesContainer< Dim >( kernel_conf, integration_rule );

   // Application of the QFunction
   QuadraturePointLoop( kernel_conf, integration_rule, [&] ( auto const & quad_index )
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
         auto neighbor_u = ReadDofs( kernel_conf, fe_space, face_info, dofs_in );

         auto Bu = InterpolateValues( kernel_conf, face_info.minus_side(), element_face_quad_data, u );

         auto neighbor_Bu = InterpolateValues( kernel_conf, face_info.plus_side(), element_face_quad_data, neighbor_u );

         auto & DBu = Bu;
         auto face_int_rule = GetFaceIntegrationRule( face_info.minus_side(), face_integration_rules );

         QuadraturePointLoop( kernel_conf, face_int_rule, [&] ( auto const & quad_index )
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

               const Real weight = GetWeight( face_info.minus_side(), quad_index, element_face_quad_data );

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
         ApplyAddTestFunctions( kernel_conf, face_info.minus_side(), element_face_quad_data, DBu, BGDBu );
      },
      [&]( auto const & face_info )
      {

         auto Bu = InterpolateValues( kernel_conf, face_info.minus_side(), element_face_quad_data, u );

         auto & DBu = Bu;
         auto face_int_rule = GetFaceIntegrationRule( face_info.minus_side(), face_integration_rules );

         QuadraturePointLoop( kernel_conf, face_int_rule, [&] ( auto const & quad_index )
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

               const Real weight = GetWeight( face_info.minus_side(), quad_index, element_face_quad_data );

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
         ApplyAddTestFunctions( kernel_conf, face_info.minus_side(), element_face_quad_data, DBu, BGDBu );
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
   typename KernelContext,
   typename FiniteElementSpace,
   typename FaceMesh,
   typename FaceIntegrationRules,
   typename MeshFaceDofToQuad,
   typename ElementFaceDofToQuad,
   typename Adv,
   typename BCType >
GENDIL_HOST_DEVICE
void AdvectionLocalFaceOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const FaceMesh & face_mesh,
   const FaceIntegrationRules & face_integration_rules,
   const GlobalIndex face_index,
   const MeshFaceDofToQuad & mesh_face_quad_data,
   const ElementFaceDofToQuad & face_quad_data,
   Adv & adv,
   BCType & boundary_field,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   const auto face_info = face_mesh.GetGlobalFaceInfo( face_index );
   
   // Read the current element DOFs
   auto u_minus = ReadDofs( kernel_conf, fe_space, face_info.minus_side(), dofs_in );
   auto u_plus = ReadDofs( kernel_conf, fe_space, face_info.plus_side(), dofs_in );

   // Element contributions
   auto Bu_minus = InterpolateValues( kernel_conf, face_info.minus_side(), face_quad_data, u_minus );
   auto Bu_plus = InterpolateValues( kernel_conf, face_info.plus_side(), face_quad_data, u_plus );

   const auto cell = fe_space.GetCell( face_info.minus_side().get_cell_index() );

   // Container to store values at all the quadrature points
   auto int_rule = GetFaceIntegrationRule( face_info.minus_side(), face_integration_rules );
   auto DBu = MakeQuadraturePointValuesContainer( kernel_conf, int_rule );

   QuadraturePointLoop( kernel_conf, int_rule, [&] ( auto const & quad_index )
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

      Real D_Advection[ Dim ];
      adv( X, D_Advection );

      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu_minus );

      const Real neighbor_Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu_plus );

      const auto reference_normal = face_info.get_reference_normal();
      const auto physical_normal = ComputePhysicalNormal( inv_J, reference_normal );

      const Real dot = Dot( D_Advection, physical_normal );
      const Real upwind = ( dot > 0.0  ) ? ( dot * Bu_q ) : ( dot * neighbor_Bu_q );

      const Real Du_q = weight * detJ * upwind;

      WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q, DBu );
   });

   // Application of the test functions
   auto BDBu_minus = ApplyTestFunctions( kernel_conf, face_info.minus_side(), face_quad_data, DBu );
   WriteAddDofs( kernel_conf, fe_space, face_info.minus_side(), BDBu_minus, dofs_out );
   auto BDBu_plus = ApplyTestFunctions( kernel_conf, face_info.plus_side(), face_quad_data, DBu );
   WriteSubDofs( kernel_conf, fe_space, face_info.plus_side(), BDBu_plus, dofs_out );
}


/**
 * @brief Implementation of the "face" contributions of the advection operator at the face level.
 * 
 * @tparam KernelContext Contextual information for the kernel.
 * @tparam FiniteElementSpaceLHS The type of the finite element space associated to the operator.
 * @tparam FiniteElementSpaceRHS The type of the finite element space associated to the operator.
 * @tparam FaceIntegrationRule The type of the integration rule used by the face operator.
 * @tparam MeshFaceDofToQuadLHS The type of the mesh data structure needed to perform computation at quadrature points on each face for the LHS.
 * @tparam MeshFaceDofToQuadRHS The type of the mesh data structure needed to perform computation at quadrature points on each face for the RHS.
 * @tparam ElementFaceDofToQuadLHS The type of the finite element data structure needed to perform computation at quadrature points on each face for the LHS.
 * @tparam ElementFaceDofToQuadRHS The type of the finite element data structure needed to perform computation at quadrature points on each face for the RHS.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam BCType The advected field on the boundary conditions described by a function.
 * @param lhs_fe_space The finite element space on the left-hand side.
 * @param rhs_fe_space The finite element space on the right-hand side.
 * @param face_mesh The mesh data structure for faces.
 * @param face_integration_rule The integration rule used on faces.
 * @param face_index The index of the face in the mesh.
 * @param lhs_mesh_face_quad_data The mesh data at quadrature points on each face for the LHS.
 * @param rhs_mesh_face_quad_data The mesh data at quadrature points on each face for the RHS.
 * @param lhs_face_quad_data The finite element data at quadrature points on each face for the LHS.
 * @param rhs_face_quad_data The finite element data at quadrature points on each face for the RHS.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @param boundary_field The function to evaluate the advected field on the boundary.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelContext,
   typename FiniteElementSpaceMinus,
   typename FiniteElementSpacePlus,
   typename FaceMesh,
   typename FaceIntegrationRulesMinus,
   typename FaceIntegrationRulesPlus,
   typename MeshFaceDofToQuadMinus,
   typename MeshFaceDofToQuadPlus,
   typename ElementFaceDofToQuadMinus,
   typename ElementFaceDofToQuadPlus,
   typename Adv,
   typename BCType,
   typename InputDofsMinus,
   typename InputDofsPlus,
   typename OutputDofsMinus,
   typename OutputDofsPlus >
GENDIL_HOST_DEVICE
void AdvectionNonconformingLocalFaceOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpaceMinus & minus_fe_space,
   const FiniteElementSpacePlus & plus_fe_space,
   const FaceMesh & face_mesh,
   const FaceIntegrationRulesMinus & minus_face_integration_rule,
   const FaceIntegrationRulesPlus & plus_face_integration_rule,
   const GlobalIndex face_index,
   const MeshFaceDofToQuadMinus & minus_mesh_face_quad_data,
   const MeshFaceDofToQuadPlus & plus_mesh_face_quad_data,
   const ElementFaceDofToQuadMinus & minus_face_quad_data,
   const ElementFaceDofToQuadPlus & plus_face_quad_data,
   Adv & adv,
   BCType & boundary_field,
   const InputDofsMinus & dofs_in_minus,
   const InputDofsPlus & dofs_in_plus,
   OutputDofsMinus & dofs_out_minus,
   OutputDofsPlus & dofs_out_plus )
{
   const auto face_info = face_mesh.GetGlobalFaceInfo( face_index );
   
   // Read the current element DOFs
   auto u_minus = ReadDofs( kernel_conf, minus_fe_space, face_info.minus_side(), dofs_in_minus );
   auto u_plus = ReadDofs( kernel_conf, plus_fe_space, face_info.plus_side(), dofs_in_plus );

   // Element contributions
   auto Bu_minus = InterpolateValues( kernel_conf, face_info.minus_side(), minus_face_quad_data, u_minus );
   auto Bu_plus = InterpolateValues( kernel_conf, face_info.plus_side(), plus_face_quad_data, u_plus );

   const auto cell = minus_fe_space.GetCell( face_info.minus_side().get_cell_index() );

   // Container to store values at all the quadrature points
   auto int_rule = GetFaceIntegrationRule( face_info.minus_side(), minus_face_integration_rule );
   auto DBu = MakeQuadraturePointValuesContainer( kernel_conf, int_rule );

   QuadraturePointLoop( kernel_conf, int_rule, [&] ( auto const & quad_index )
   {
      constexpr Integer Dim = FiniteElementSpaceMinus::Dim;
      static_assert(
         Dim == FiniteElementSpacePlus::Dim,
         "Dimension mismatch between MinusFiniteElementSpaceMinus and PlusFiniteElementSpacePlus finite element spaces"
      );

      using Mesh = typename FiniteElementSpaceMinus::mesh_type;
      using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
      using Jacobian = typename Mesh::cell_type::jacobian;
      
      PhysicalCoordinates X;
      Jacobian J_mesh;

      mesh::ComputePhysicalCoordinatesAndJacobian( cell, face_info.minus_side(), quad_index, minus_mesh_face_quad_data, X, J_mesh );

      Jacobian inv_J;
      const Real detJ = ComputeInverseAndDeterminant( J_mesh, inv_J );

      const Real weight = GetWeight( face_info.minus_side(), quad_index, minus_face_quad_data );

      Real D_Advection[ Dim ];
      adv( X, D_Advection );

      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu_minus );

      const Real neighbor_Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu_plus );

      const auto reference_normal = face_info.get_reference_normal();
      const auto physical_normal = ComputePhysicalNormal( inv_J, reference_normal );

      const Real dot = Dot( D_Advection, physical_normal );
      const Real upwind = ( dot > 0.0  ) ? ( dot * Bu_q ) : ( dot * neighbor_Bu_q );

      const Real Du_q = weight * detJ * upwind;

      WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q, DBu );
   });

   // Application of the test functions
   auto minus_BDBu = ApplyTestFunctions( kernel_conf, face_info.minus_side(), minus_face_quad_data, DBu );
   auto plus_BDBu = ApplyTestFunctions( kernel_conf, face_info.plus_side(), plus_face_quad_data, DBu );
   WriteAddDofs( kernel_conf, minus_fe_space, face_info.minus_side(), minus_BDBu, dofs_out_minus );
   WriteSubDofs( kernel_conf, plus_fe_space, face_info.plus_side(), plus_BDBu, dofs_out_plus );
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

         AdvectionFusedOperatorWithoutBC(
            kernel_conf,
            fe_space,
            IntegrationRule{},
            FaceIntegrationRulesTuple{},
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

         AdvectionFusedOperatorWithBC(
            kernel_conf,
            fe_space,
            IntegrationRule{},
            FaceIntegrationRulesTuple{},
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
 * @brief Explicit advection face operator containing only "face" contributions.
 * 
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam FaceIntegrationRule The type of the integration rule used by the face operator.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam FaceMesh The type of face mesh associated to the operator.
 * @tparam MeshFaceDofToQuad The type of the mesh face data structure needed to perform computation at quadrature points on each face.
 * @tparam ElementFaceDofToQuad The type of the finite element data structure needed to perform computation at quadrature points on each face.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam BCType The advected field on the boundary conditions described by a function.
 * @param fe_space The finite element space.
 * @param face_mesh The face mesh.
 * @param mesh_face_quad_data The mesh data at quadrature points needed to perform the computation on each face.
 * @param element_face_quad_data The finite element data at quadrature points need to perform the computation on each face.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @param boundary_field The function to evaluate the advected field on the boundary.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FaceIntegrationRules,
   typename FiniteElementSpace,
   typename FaceMesh,
   typename MeshFaceDofToQuad,
   typename ElementFaceDofToQuad,
   typename Adv,
   typename BCType >
void AdvectionExplicitFaceOperator(
   const FiniteElementSpace & fe_space,
   const FaceMesh & face_mesh,
   const MeshFaceDofToQuad & mesh_face_quad_data,
   const ElementFaceDofToQuad & element_face_quad_data,
   Adv adv,
   BCType & boundary_field,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   mesh::GlobalFaceIterator<KernelConfiguration>(
      face_mesh,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex face_index ) mutable
      {
         constexpr size_t required_shared_mem = required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         AdvectionLocalFaceOperator(
            kernel_conf,
            fe_space,
            face_mesh,
            FaceIntegrationRules{},
            face_index,
            mesh_face_quad_data,
            element_face_quad_data,
            adv,
            boundary_field,
            dofs_in,
            dofs_out );
      }
   );
}

/**
 * @brief Explicit advection face operator containing only "face" contributions.
 * 
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam FaceIntegrationRule The type of the integration rule used by the face operator.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam FaceMesh The type of face mesh associated to the operator.
 * @tparam MeshFaceDofToQuad The type of the mesh data structure needed to perform computation at quadrature points on each face.
 * @tparam ElementFaceDofToQuad The type of the finite element data structure needed to perform computation at quadrature points on each face.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam BCType The advected field on the boundary conditions described by a function.
 * @param fe_space_lhs The finite element space.
 * @param fe_space_rhs The finite element space.
 * @param face_mesh The face mesh associated to the operator.
 * @param mesh_face_quad_data_lhs The mesh data at quadrature points needed to perform the computation on each face.
 * @param mesh_face_quad_data_rhs The mesh data at quadrature points needed to perform the computation on each face.
 * @param element_face_quad_data_lhs The finite element data at quadrature points need to perform the computation on each face.
 * @param element_face_quad_data_rhs The finite element data at quadrature points need to perform the computation on each face.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @param boundary_field The function to evaluate the advected field on the boundary.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FaceIntegrationRulesLHS,
   typename FaceIntegrationRulesRHS,
   typename FiniteElementSpaceLHS,
   typename FiniteElementSpaceRHS,
   typename FaceMesh,
   typename MeshFaceDofToQuadLHS,
   typename MeshFaceDofToQuadRHS,
   typename ElementFaceDofToQuadLHS,
   typename ElementFaceDofToQuadRHS,
   typename Adv,
   typename BCType,
   typename InputDofsLHS,
   typename InputDofsRHS,
   typename OutputDofsLHS,
   typename OutputDofsRHS>
void AdvectionExplicitNonconformingFaceOperator(
   const FiniteElementSpaceLHS & fe_space_lhs,
   const FiniteElementSpaceRHS & fe_space_rhs,
   const FaceMesh & face_mesh,
   const MeshFaceDofToQuadLHS & mesh_face_quad_data_lhs,
   const MeshFaceDofToQuadRHS & mesh_face_quad_data_rhs,
   const ElementFaceDofToQuadLHS & element_face_quad_data_lhs,
   const ElementFaceDofToQuadRHS & element_face_quad_data_rhs,
   Adv adv,
   BCType & boundary_field,
   const InputDofsLHS & dofs_in_lhs,
   const InputDofsRHS & dofs_in_rhs,
   OutputDofsLHS & dofs_out_lhs,
   OutputDofsRHS & dofs_out_rhs )
{
   mesh::GlobalFaceIterator<KernelConfiguration>(
      face_mesh,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex face_index ) mutable
      {
         constexpr size_t required_shared_mem = required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         AdvectionNonconformingLocalFaceOperator(
            kernel_conf,
            fe_space_lhs,
            fe_space_rhs,
            face_mesh,
            FaceIntegrationRulesLHS{},
            FaceIntegrationRulesRHS{},
            face_index,
            mesh_face_quad_data_lhs,
            mesh_face_quad_data_rhs,
            element_face_quad_data_lhs,
            element_face_quad_data_rhs,
            adv,
            boundary_field,
            dofs_in_lhs,
            dofs_in_rhs,
            dofs_out_lhs,
            dofs_out_rhs );
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
   typename FaceMesh,
   typename IntegrationRule,
   typename Adv,
   typename BCType >
class AdvectionFaceOperator
   : public MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >
{
   using base = MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >;
   FaceMesh face_meshes;
   Adv adv;
   BCType boundary_field;

   using input = StridedView< FiniteElementSpace::Dim + 1, const Real >;
   using output = StridedView< FiniteElementSpace::Dim + 1, Real >;

public:
   /**
    * @brief Construct a new AdvectionOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    * @param adv The function to evaluate the advection vector at physical coordinates.
    * @param boundary_field The function to evaluate the advected field on the boundary.
    */
   AdvectionFaceOperator(
      const FiniteElementSpace & finite_element_space,
      const FaceMesh & face_mesh,
      const IntegrationRule & int_rules,
      Adv && adv,
      BCType && boundary_field ) :
      base( finite_element_space, int_rules ),
      face_meshes( face_mesh ),
      adv( adv ),
      boundary_field( boundary_field )
   { }

   /**
    * @brief Apply the advection operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      static_assert(
         std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction >,
         "AdvectionOperator::operator() only supports L2Restriction" );
      mesh::ForEachFaceMesh(
         face_meshes,
         [&] ( const auto & face_mesh ) mutable
         {
            AdvectionExplicitFaceOperator< KernelPolicy, IntegrationRule, typename base::face_integration_rules >(
               this->finite_element_space,
               face_mesh,
               this->mesh_face_quad_data,
               this->element_face_quad_data,
               adv,
               boundary_field,
               dofs_in,
               dofs_out );
         }
      );
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
 * @brief Represent an advection finite element operrator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpaceLHS The type of the finite element space associated to the operator.
 * @tparam FiniteElementSpaceRHS The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam BCType The advected field on the boundary conditions described by a function.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpaceLHS,
   typename FiniteElementSpaceRHS,
   typename FaceMesh,
   typename IntegrationRule,
   typename Adv,
   typename BCType >
class AdvectionNonconformingFaceOperator
#ifdef GENDIL_USE_MFEM
: public mfem::Operator
#endif // GENDIL_USE_MFEM
{
   // Finite element space
   using fe_space_type_lhs = FiniteElementSpaceLHS;
   using fe_space_type_rhs = FiniteElementSpaceRHS;

   // Integration rules (this assumes the same integration rule for both sides)
   using integration_rule_type = IntegrationRule;
   using face_integration_rules = decltype( GetFaceIntegrationRules( IntegrationRule{} ) ); // This assumes hypercubes

   // Mesh quadrature data
   using mesh_type_lhs = typename fe_space_type_lhs::mesh_type;
   using mesh_type_rhs = typename fe_space_type_rhs::mesh_type;
   using mesh_quad_data_type_lhs = typename mesh_type_lhs::cell_type::template QuadData< IntegrationRule >;
   using mesh_quad_data_type_rhs = typename mesh_type_rhs::cell_type::template QuadData< IntegrationRule >;
   using mesh_face_quad_data_type_lhs =
      decltype(
         MakeMeshFaceQuadData< mesh_type_lhs >( face_integration_rules{} )
      );
   using mesh_face_quad_data_type_rhs =
      decltype(
         MakeMeshFaceQuadData< mesh_type_rhs >( face_integration_rules{} )
      );

   // Finite elements quadrature data
   using element_quad_data_type_lhs =
      decltype(
         MakeDofToQuad<
            typename fe_space_type_lhs::finite_element_type::shape_functions,
            integration_rule_type
         >()
      );
   using element_quad_data_type_rhs =
      decltype(
         MakeDofToQuad<
            typename fe_space_type_rhs::finite_element_type::shape_functions,
            integration_rule_type
         >()
      );
   using element_face_quad_data_type_lhs =
      decltype(
         MakeFaceDofToQuad<
            typename fe_space_type_lhs::finite_element_type::shape_functions,
            face_integration_rules
         >()
      );
   using element_face_quad_data_type_rhs =
      decltype(
         MakeFaceDofToQuad<
            typename fe_space_type_rhs::finite_element_type::shape_functions,
            face_integration_rules
         >()
      );

   const fe_space_type_lhs & finite_element_space_lhs;
   const fe_space_type_rhs & finite_element_space_rhs;

   mesh_quad_data_type_lhs mesh_quad_data_lhs;
   mesh_quad_data_type_rhs mesh_quad_data_rhs;
   mesh_face_quad_data_type_lhs mesh_face_quad_data_lhs;
   mesh_face_quad_data_type_rhs mesh_face_quad_data_rhs;

   element_quad_data_type_lhs element_quad_data_lhs;
   element_quad_data_type_rhs element_quad_data_rhs;
   element_face_quad_data_type_lhs element_face_quad_data_lhs;
   element_face_quad_data_type_rhs element_face_quad_data_rhs;

   const FaceMesh & face_meshes;
   Adv adv;
   BCType boundary_field;

   // using input = StridedView< FiniteElementSpace::Dim + 1, const Real >;
   // using output = StridedView< FiniteElementSpace::Dim + 1, Real >;

public:
   /**
    * @brief Construct a new AdvectionOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    * @param adv The function to evaluate the advection vector at physical coordinates.
    * @param boundary_field The function to evaluate the advected field on the boundary.
    */
   AdvectionNonconformingFaceOperator(
      const FiniteElementSpaceLHS & lhs_finite_element_space,
      const FiniteElementSpaceRHS & rhs_finite_element_space,
      const FaceMesh & face_mesh,
      const IntegrationRule & int_rules,
      Adv && adv,
      BCType && boundary_field ) :
#ifdef GENDIL_USE_MFEM
      Operator(
         lhs_finite_element_space.GetNumberOfFiniteElementDofs() + rhs_finite_element_space.GetNumberOfFiniteElementDofs(),
         lhs_finite_element_space.GetNumberOfFiniteElementDofs() + rhs_finite_element_space.GetNumberOfFiniteElementDofs() ),
#endif // GENDIL_USE_MFEM
      finite_element_space_lhs( lhs_finite_element_space ),
      finite_element_space_rhs( rhs_finite_element_space ),
      face_meshes( face_mesh ),
      adv( adv ),
      boundary_field( boundary_field )
   { }

   /**
    * @brief Apply the advection operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   template < typename input_lhs, typename input_rhs, typename output_lhs, typename output_rhs >
   void Apply(
      const input_lhs & dofs_in_lhs,
      const input_rhs & dofs_in_rhs,
      output_lhs & dofs_out_lhs,
      output_rhs & dofs_out_rhs ) const
   {
      static_assert(
         std::is_same_v< typename FiniteElementSpaceLHS::restriction_type, L2Restriction >,
         "AdvectionOperator::operator() only supports L2Restriction" );
      static_assert(
         std::is_same_v< typename FiniteElementSpaceRHS::restriction_type, L2Restriction >,
         "AdvectionOperator::operator() only supports L2Restriction" );
      mesh::ForEachFaceMesh(
         face_meshes,
         [&] ( const auto & face_mesh ) mutable
         {
            AdvectionExplicitNonconformingFaceOperator<
               KernelPolicy,
               IntegrationRule,
               face_integration_rules,
               face_integration_rules >(
                  finite_element_space_lhs,
                  finite_element_space_rhs,
                  face_mesh,
                  mesh_face_quad_data_lhs,
                  mesh_face_quad_data_rhs,
                  element_face_quad_data_lhs,
                  element_face_quad_data_rhs,
                  adv,
                  boundary_field,
                  dofs_in_lhs,
                  dofs_in_rhs,
                  dofs_out_lhs,
                  dofs_out_rhs );
         }
      );
   }

   void operator()( const Vector & dofs_vector_in, Vector & dofs_vector_out ) const
   {
      auto dofs_in_lhs = MakeReadOnlyEVectorView< KernelPolicy >( finite_element_space_lhs, dofs_vector_in );
      auto dofs_in_rhs = MakeReadOnlyEVectorView< KernelPolicy >( finite_element_space_rhs, dofs_vector_in );
      auto dofs_out_lhs = MakeWriteOnlyEVectorView< KernelPolicy >( finite_element_space_lhs, dofs_vector_out );
      auto dofs_out_rhs = MakeWriteOnlyEVectorView< KernelPolicy >( finite_element_space_rhs, dofs_vector_out );

      Apply( dofs_in_lhs, dofs_in_rhs, dofs_out_lhs, dofs_out_rhs );
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
      auto dofs_in_lhs = MakeReadOnlyEVectorView< KernelPolicy >( finite_element_space_lhs, dofs_vector_in );
      auto dofs_in_rhs = MakeReadOnlyEVectorView< KernelPolicy >( finite_element_space_rhs, dofs_vector_in );
      auto dofs_out_lhs = MakeWriteOnlyEVectorView< KernelPolicy >( finite_element_space_lhs, dofs_vector_out );
      auto dofs_out_rhs = MakeWriteOnlyEVectorView< KernelPolicy >( finite_element_space_rhs, dofs_vector_out );

      Apply( dofs_in_lhs, dofs_in_rhs, dofs_out_lhs, dofs_out_rhs );
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

/**
 * @brief Factory to build advection face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam FaceMesh The type of face mesh associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param face_mesh The face mesh associated to the operator.
 * @param int_rule The integration rule used by the operator. This is a volume integration rule,
 * the face integrations rules are deuced automatically.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @return auto The advection operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename FaceMesh,
   typename IntegrationRule,
   typename Adv >
auto MakeAdvectionFaceOperator(
   const FiniteElementSpace & finite_element_space,
   const FaceMesh & face_mesh,
   const IntegrationRule & int_rule,
   Adv && adv )
{
   return AdvectionFaceOperator<
             KernelPolicy,
             FiniteElementSpace,
             FaceMesh,
             IntegrationRule,
             Adv,
             Empty
          >( finite_element_space, face_mesh, int_rule, adv, Empty{} );
}

/**
 * @brief Factory to build advection face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpaceLHS The type of the finite element space associated to the operator.
 * @tparam FiniteElementSpaceRHS The type of the finite element space associated to the operator.
 * @tparam FaceMesh The type of face mesh associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @param finite_element_space_lhs The finite element space associated to the operator.
 * @param finite_element_space_rhs The finite element space associated to the operator.
 * @param face_mesh The face mesh associated to the operator.
 * @param int_rule The integration rule used by the operator. This is a volume integration rule,
 * the face integrations rules are deuced automatically.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @return auto The advection operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpaceLHS,
   typename FiniteElementSpaceRHS,
   typename FaceMesh,
   typename IntegrationRule,
   typename Adv >
auto MakeAdvectionFaceOperator(
   const FiniteElementSpaceLHS & finite_element_space_lhs,
   const FiniteElementSpaceRHS & finite_element_space_rhs,
   const FaceMesh & face_mesh,
   const IntegrationRule & int_rule,
   Adv && adv )
{
   return AdvectionNonconformingFaceOperator<
             KernelPolicy,
             FiniteElementSpaceLHS,
             FiniteElementSpaceRHS,
             FaceMesh,
             IntegrationRule,
             Adv,
             Empty
          >( finite_element_space_lhs, finite_element_space_rhs, face_mesh, int_rule, adv, Empty{} );
}

}

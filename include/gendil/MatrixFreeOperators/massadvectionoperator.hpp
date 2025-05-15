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
 * @brief Implementation of the "volume" contributions of the mass+advection operator at the element level.
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
 * @param adv The function to evaluate the advection vector at physical coordinates.
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
   typename Adv,
   typename Sigma >
GENDIL_HOST_DEVICE
void MassAdvectionElementOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   Adv & adv,
   Sigma & sigma,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   using Mesh = typename FiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   constexpr Integer Dim = FiniteElementSpace::Dim;

   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   auto Bu = InterpolateValues( kernel_conf, element_quad_data, u );

   const auto cell = fe_space.GetCell( element_index );

   // Container to store values at all the quadrature points
   auto Duq = MakeQuadraturePointValuesContainer( kernel_conf, IntegrationRule{} );
   auto DGuq = MakeQuadraturePointValuesContainer< Dim >( kernel_conf, IntegrationRule{} );

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

      const Real D_Mass = sigma( X );

      // Compute: weight * detJ * D_A * J^-T * Gu_q
      Real D_Advection[ Dim ];
      adv( X, D_Advection );

      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      const Real Du_q = D_Mass * Bu_q;

      ApplyMappingTranspose( inv_J_mesh, D_Advection );
      const Real w = - weight * detJ * Bu_q;
      Real DGu_q[Dim];
      for( size_t i = 0; i < Dim; i++)
      {
         DGu_q[i] = w * D_Advection[i];
      }

      const Real Du_q_w = weight * detJ * Du_q;

      WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q_w, Duq );
      WriteQuadratureLocalValues( kernel_conf, quad_index, DGu_q, DGuq );
   } );

   // Application of the test functions
   auto & GDBu = Bu; // use same memory;
   ApplyGradientTestFunctionsAtQPoints( kernel_conf, element_quad_data, DGuq, GDBu );
   GDBu += Duq;
   auto BGDBu = ApplyTestFunctions( kernel_conf, element_quad_data, GDBu );

   WriteDofs( kernel_conf, fe_space, element_index, BGDBu, dofs_out );
}

/**
 * @brief Implementation of the "face" contributions of the mass+advection operator at the element level.
 * 
 * @tparam FaceIntegrationRulesTuple List of the types of integration rules used on each face of the element.
 * @tparam KernelContext The type of the kernel context.
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
 * @param element_face_quad_data The finite element data at quadrature points need to perform the computation on each face.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename FaceIntegrationRulesTuple,
   typename KernelContext,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename MeshFaceDofToQuad,
   typename ElementQuadData,
   typename ElementFaceDofToQuad,
   typename Adv,
   typename Sigma >
GENDIL_HOST_DEVICE
void MassAdvectionFaceOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const MeshFaceDofToQuad & mesh_face_quad_data,
   const ElementQuadData & element_quad_data,
   const ElementFaceDofToQuad & element_face_quad_data,
   Adv & adv,
   Sigma & sigma,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   const auto cell = fe_space.GetCell( element_index );

   // Container to store values at all the quadrature points
   // decltype( u ) BfDBfu{};
   auto BfDBfu = MakeElementDoFValuesContainer( kernel_conf, fe_space );

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

         auto & Duq = Bu;
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

            // Compute: weight * detJ * D_A * J^-T * Gu_q
            Real D_Advection[ Dim ];
            adv( X, D_Advection );

            const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

            const Real neighbor_Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, neighbor_Bu );

            const auto reference_normal = GetReferenceNormal( face_info );
            const auto physical_normal = ComputePhysicalNormal( inv_J, reference_normal );

            const Real dot = Dot( D_Advection, physical_normal );
            const Real upwind = dot > 0.0 ? dot * Bu_q : dot * neighbor_Bu_q;

            const Real Du_q = weight * detJ * upwind;

            WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q, Duq );
         });

         // Application of the test functions
         ApplyAddTestFunctions( kernel_conf, std::get< face_index >( element_face_quad_data ), Duq, BfDBfu );
      }
   );
   WriteAddDofs( kernel_conf, fe_space, element_index, BfDBfu, dofs_out );
}

/**
 * @brief Explicit mass-advection operator containing the "volume" and "face" contributions.
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
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param fe_space The finite element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param mesh_face_quad_data The mesh data at quadrature points needed to perform the computation on each face.
 * @param element_quad_data The finite element data at quadrature points need to perform the computation.
 * @param element_face_quad_data The finite element data at quadrature points need to perform the computation on each face.
 * @param adv The function to evaluate the advection vector at physical coordinates.
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
   typename Adv,
   typename Sigma >
void MassAdvectionExplicitOperator(
   const FiniteElementSpace & fe_space,
   const MeshQuadData & mesh_quad_data,
   const MeshFaceDofToQuad & mesh_face_quad_data,
   const ElementQuadData & element_quad_data,
   const ElementFaceDofToQuad & element_face_quad_data,
   Adv adv,
   Sigma sigma,
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

         MassAdvectionElementOperator< IntegrationRule >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            element_quad_data,
            adv,
            sigma,
            dofs_in,
            dofs_out );
         MassAdvectionFaceOperator< FaceIntegrationRulesTuple >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            mesh_face_quad_data,
            element_quad_data,
            element_face_quad_data,
            adv,
            sigma,
            dofs_in,
            dofs_out );
      }
   );
}

/**
 * @brief Represent a mass-advection finite element operrator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Adv,
   typename Sigma >
class MassAdvectionOperator
   : public MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >
{
   using base = MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >;
   Adv adv;
   Sigma sigma;

   using input = StridedView< FiniteElementSpace::Dim + 1, const Real >;
   using output = StridedView< FiniteElementSpace::Dim + 1, Real >;

public:
   /**
    * @brief Construct a new MassAdvectionOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    * @param adv The function to evaluate the advection vector at physical coordinates.
    * @param sigma The function to evaluate the mass density at physical coordinates.
    */
   MassAdvectionOperator( const FiniteElementSpace & finite_element_space,
                          const IntegrationRule & int_rules,
                          Adv & adv,
                          Sigma & sigma ) :
      base( finite_element_space, int_rules ),
      adv( adv ),
      sigma( sigma )
   { }

   /**
    * @brief Apply the mass-advection operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      static_assert(
         std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction >,
         "MassAdvectionOperator::operator() only supports L2Restriction" );
      MassAdvectionExplicitOperator< KernelPolicy, typename base::integration_rule, typename base::face_integration_rules >
         ( this->finite_element_space,
           this->mesh_quad_data,
           this->mesh_face_quad_data,
           this->element_quad_data,
           this->element_face_quad_data,
           adv,
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
    * @brief Apply the mass-advection operator.
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
 * @brief Factory to build mass-advection operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 * @tparam Sigma The type of the function to evaluate the mass density at physical coordinates.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @param adv The function to evaluate the advection vector at physical coordinates.
 * @param sigma The function to evaluate the mass density at physical coordinates.
 * @return auto The mass-advection operator.
 */
template < typename FiniteElementSpace,
           typename IntegrationRule,
           typename Adv,
           typename Sigma >
auto MakeMassAdvectionOperator( const FiniteElementSpace & finite_element_space,
                                const IntegrationRule & int_rule,
                                Adv & adv,
                                Sigma & sigma )
{
   using KernelPolicy = SerialKernelConfiguration;

   return MassAdvectionOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule,
             Adv,
             Sigma
          >( finite_element_space, int_rule, adv, sigma );
}

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename Adv,
   typename Sigma >
auto MakeMassAdvectionOperator( const FiniteElementSpace & finite_element_space,
                                const IntegrationRule & int_rule,
                                Adv & adv,
                                Sigma & sigma )
{
   return MassAdvectionOperator<
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule,
             Adv,
             Sigma
          >( finite_element_space, int_rule, adv, sigma );
}

}

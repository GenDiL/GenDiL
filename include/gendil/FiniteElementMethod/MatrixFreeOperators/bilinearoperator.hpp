// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/NumericalIntegration/faceintegrationrules.hpp"
#include "gendil/Meshes/makefacequaddata.hpp"

namespace gendil {

/**
 * @brief Contains basic data for matrix-free bilinear DG finite element operators.
 * 
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 */
template < typename FiniteElementSpace, typename IntegrationRule >
class MatrixFreeBilinearFiniteElementOperator
#ifdef GENDIL_USE_MFEM
: public mfem::Operator
#endif // GENDIL_USE_MFEM
{
protected:
   using fe_space_type = FiniteElementSpace;
   using finite_element_type = typename fe_space_type::finite_element_type;

   using integration_rule = IntegrationRule;

   /**
    * @brief Deduce the face integration rules from the given "volume"
    * integration rule.
    * 
    */
   using face_integration_rules = decltype( GetFaceIntegrationRules( IntegrationRule{} ) );

   const FiniteElementSpace & finite_element_space;
   const IntegrationRule & int_rules;

   using Mesh = typename FiniteElementSpace::mesh_type;

   /**
    * @brief Data structure storing information to compute physical coordinates
    * and mesh jacobians at the provided integration rule quadrature points.
    * 
    */
   using MeshQuadData = typename Mesh::cell_type::template QuadData< IntegrationRule >;
   const MeshQuadData mesh_quad_data;

   /**
    * @brief Data structure storing information to compute field values
    * and field gradients at the provided integration rule quadrature points.
    * 
    */
   using ElementQuadData = decltype(
                              MakeDofToQuad<
                                 typename finite_element_type::shape_functions,
                                 integration_rule
                              >()
                           );
   const ElementQuadData element_quad_data;

   /**
    * @brief Data structure storing information to compute physical coordinates
    * and mesh jacobians at the face integration rule quadrature points.
    * 
    * @note This data is a std::tuple containing QuadData for each face.
    */
   using MeshFaceQuadData = decltype(
                               MakeMeshFaceQuadData< Mesh >( face_integration_rules{} )
                            );
   const MeshFaceQuadData mesh_face_quad_data;

   /**
    * @brief Data structure storing information to compute field values
    * and field gradients at the face integration rule quadrature points.
    * 
    * @note This data is a std::tuple containing a DofToQuad struct for each face.
    */
   using ElementFaceQuadData = decltype(
                           MakeFaceDofToQuad<
                              typename finite_element_type::shape_functions,
                              face_integration_rules
                           >()
                        );
   const ElementFaceQuadData element_face_quad_data;

public:
   /**
    * @brief Construct a new MatrixFreeBilinearFiniteElementOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    */
   MatrixFreeBilinearFiniteElementOperator( const FiniteElementSpace & finite_element_space,
                                            const IntegrationRule & int_rules ) :
#ifdef GENDIL_USE_MFEM
      Operator( finite_element_space.GetNumberOfFiniteElementDofs() ),
#endif // GENDIL_USE_MFEM
      finite_element_space( finite_element_space ),
      int_rules( int_rules ),
      mesh_quad_data{},
      element_quad_data{}
   { }
};

}

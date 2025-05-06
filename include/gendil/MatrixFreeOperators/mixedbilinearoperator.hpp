// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/NumericalIntegration/faceintegrationrules.hpp"
#include "gendil/Meshes/makefacequaddata.hpp"

namespace gendil {

/**
 * @brief Contains basic data for matrix-free mixed bilinear DG finite element operators.
 * 
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 */
template < typename TrialFiniteElementSpace, typename TestFiniteElementSpace, typename IntegrationRule >
class MatrixFreeMixedBilinearFiniteElementOperator
#ifdef GENDIL_USE_MFEM
: public mfem::Operator
#endif // GENDIL_USE_MFEM
{
protected:
   using trial_fe_space_type = TrialFiniteElementSpace;
   using test_fe_space_type = TestFiniteElementSpace;
   using trial_finite_element_type = typename trial_fe_space_type::finite_element_type;
   using test_finite_element_type = typename test_fe_space_type::finite_element_type;

   using integration_rule = IntegrationRule;

   const TrialFiniteElementSpace & trial_finite_element_space;
   const TestFiniteElementSpace & test_finite_element_space;
   const IntegrationRule & int_rules;

   using Mesh = typename TrialFiniteElementSpace::mesh_type;

   /**
    * @brief Data structure storing information to compute physical coordinates
    * and mesh jacobians at the provided integration rule quadrature points.
    * 
    */
   using MeshQuadData = typename Mesh::cell_type::template QuadData< IntegrationRule >;
   MeshQuadData mesh_quad_data;

   /**
    * @brief Data structure storing information to compute field values
    * and field gradients on the trial finite element space at the provided
    * integration rule quadrature points.
    * 
    */
   using TrialElementQuadData = decltype(
                              MakeDofToQuad<
                                 trial_finite_element_type,
                                 integration_rule
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
                                 test_finite_element_type,
                                 integration_rule
                              >()
                           );
   TestElementQuadData test_element_quad_data;

   // TODO: Add face data

public:
   /**
    * @brief Construct a new MixedMassOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    */
   MatrixFreeMixedBilinearFiniteElementOperator(
      const TrialFiniteElementSpace & trial_finite_element_space,
      const TestFiniteElementSpace & test_finite_element_space,
      const IntegrationRule & int_rules ) :
#ifdef GENDIL_USE_MFEM
      Operator( test_finite_element_space.GetNumberOfFiniteElementDofs(), trial_finite_element_space.GetNumberOfFiniteElementDofs() ),
#endif // GENDIL_USE_MFEM
      trial_finite_element_space( trial_finite_element_space ),
      test_finite_element_space( test_finite_element_space ),
      int_rules( int_rules ),
      mesh_quad_data{},
      trial_element_quad_data{},
      test_element_quad_data{}
   { }
};

}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file nodalsubspaceprojectionoperator.hpp
 * @brief Collocated nodal projection after integrating trailing tensor axes.
 *
 * This operator evaluates a scalar tensor-product trial field, integrates the
 * dimensions not present in the test space, and stores the retained values as
 * nodal test-space coefficients.  It does not compute a best approximation in
 * the L-infinity norm.
 */

#include "gendil/prelude.hpp"
#include "gendil/Utilities/utilities.hpp"
#include "gendil/Interfaces/interfaces.hpp"
#include "gendil/Meshes/meshes.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/NumericalIntegration/numericalintegration.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/kerneloperators.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil
{

namespace nodal_subspace_projection_detail
{

template < typename ShapeFunctions, typename PointSet, typename = void >
struct uses_nodal_point_set : std::false_type
{ };

template < typename ShapeFunctions, typename PointSet >
struct uses_nodal_point_set<
   ShapeFunctions,
   PointSet,
   std::void_t< typename std::remove_cvref_t< ShapeFunctions >::points > >
   : std::bool_constant<
        std::same_as<
           typename std::remove_cvref_t< ShapeFunctions >::points,
           std::remove_cvref_t< PointSet > > >
{ };

template < typename ShapeFunctions, typename IntegrationRule >
consteval bool HasCollocatedNodalTensorRule()
{
   using ShapeFunctionsType = std::remove_cvref_t< ShapeFunctions >;
   using IntegrationRuleType = std::remove_cvref_t< IntegrationRule >;

   if constexpr (
      !is_tensor_shape_functions_v< ShapeFunctionsType > ||
      !requires
      {
         typename ShapeFunctionsType::shape_functions_1d_tuple;
         typename IntegrationRuleType::points::points_1d_tuple;
      } )
   {
      return false;
   }
   else
   {
      using ShapeFunctionsTuple =
         typename ShapeFunctionsType::shape_functions_1d_tuple;
      using PointSetTuple =
         typename IntegrationRuleType::points::points_1d_tuple;
      if constexpr (
         std::tuple_size_v< ShapeFunctionsTuple > !=
         std::tuple_size_v< PointSetTuple > )
      {
         return false;
      }
      else
      {
         return []< size_t... I >( std::index_sequence< I... > ) consteval
         {
            return
               ( uses_nodal_point_set<
                    std::tuple_element_t< I, ShapeFunctionsTuple >,
                    std::tuple_element_t< I, PointSetTuple > >::value && ... );
         }( std::make_index_sequence<
               std::tuple_size_v< ShapeFunctionsTuple > >{} );
      }
   }
}

template < typename TrialIntegrationRule, typename TestIntegrationRule >
consteval bool TestRuleMatchesRetainedTrialPrefix()
{
   using TrialPointSetTuple = typename std::remove_cvref_t<
      TrialIntegrationRule >::points::points_1d_tuple;
   using TestPointSetTuple = typename std::remove_cvref_t<
      TestIntegrationRule >::points::points_1d_tuple;

   if constexpr (
      std::tuple_size_v< TestPointSetTuple > >
      std::tuple_size_v< TrialPointSetTuple > )
   {
      return false;
   }
   else
   {
      return []< size_t... I >( std::index_sequence< I... > ) consteval
      {
         return
            ( std::same_as<
                 std::tuple_element_t< I, TestPointSetTuple >,
                 std::tuple_element_t< I, TrialPointSetTuple > > && ... );
      }( std::make_index_sequence<
            std::tuple_size_v< TestPointSetTuple > >{} );
   }
}

template <
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename TrialIntegrationRule,
   typename TestIntegrationRule >
consteval bool ValidateCompatibility()
{
   using TrialShapeFunctions = typename std::remove_cvref_t<
      TrialFiniteElementSpace >::finite_element_type::shape_functions;
   using TestShapeFunctions = typename std::remove_cvref_t<
      TestFiniteElementSpace >::finite_element_type::shape_functions;

   static_assert(
      is_tensor_shape_functions_v< TrialShapeFunctions >,
      "NodalSubspaceProjection requires scalar tensor-product trial shape functions." );
   static_assert(
      is_tensor_shape_functions_v< TestShapeFunctions >,
      "NodalSubspaceProjection requires scalar tensor-product test shape functions." );

   if constexpr (
      is_tensor_shape_functions_v< TrialShapeFunctions > &&
      is_tensor_shape_functions_v< TestShapeFunctions > )
   {
      static_assert(
         std::same_as<
            typename TestShapeFunctions::dof_shape,
            typename std::remove_cvref_t<
               TestIntegrationRule >::points::num_points_tensor >,
         "NodalSubspaceProjection requires the test DoF tensor extents to "
         "match the test quadrature tensor extents." );
      static_assert(
         HasCollocatedNodalTensorRule<
            TestShapeFunctions,
            TestIntegrationRule >(),
         "NodalSubspaceProjection requires test quadrature points to coincide "
         "axis by axis with the test finite-element interpolation nodes." );
      static_assert(
         TestRuleMatchesRetainedTrialPrefix<
            TrialIntegrationRule,
            TestIntegrationRule >(),
         "NodalSubspaceProjection requires the test quadrature rule to match "
         "the retained prefix of the trial quadrature rule." );
   }

   return true;
}

} // namespace nodal_subspace_projection_detail

// TODO: Generalize
/**
 * @brief Integrate trailing tensor dimensions into collocated nodal test DoFs.
 * 
 * @tparam TrialIntegrationRule The type of the integration rule used by the element operator.
 * @tparam TestIntegrationRule The type of the integration rule used by the element operator.
 * @tparam KernelContext Contextual information for the kernel.
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam TrialElementQuadData Trial-space interpolation and quadrature data.
 * @tparam TestElementQuadData Test-space interpolation and quadrature data.
 * @param kernel_conf Execution context and scratch allocator.
 * @param trial_fe_space Higher-dimensional trial finite-element space.
 * @param test_fe_space Retained nodal test finite-element space.
 * @param trial_element_index Trial element being reduced.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param trial_element_quad_data Trial interpolation and quadrature data.
 * @param test_element_quad_data Collocated test interpolation and quadrature data.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 *
 * @note The first TestFiniteElementSpace::Dim trial quadrature axes are
 * retained. All remaining axes are integrated.
 */
template <
   typename TrialIntegrationRule,
   typename TestIntegrationRule,
   typename KernelContext,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename MeshQuadData,
   typename TrialElementQuadData,
   typename TestElementQuadData >
GENDIL_HOST_DEVICE
void NodalSubspaceProjectionElementOperator(
   KernelContext & kernel_conf,
   const TrialFiniteElementSpace & trial_fe_space,
   const TestFiniteElementSpace & test_fe_space,
   const GlobalIndex trial_element_index,
   const MeshQuadData & mesh_quad_data,
   const TrialElementQuadData & trial_element_quad_data,
   const TestElementQuadData & test_element_quad_data,
   const element_tensor_view_t< TrialFiniteElementSpace, const Real > & dofs_in,
   element_tensor_view_t< TestFiniteElementSpace, Real > & dofs_out )
{
   static_assert(
      nodal_subspace_projection_detail::ValidateCompatibility<
         TrialFiniteElementSpace,
         TestFiniteElementSpace,
         TrialIntegrationRule,
         TestIntegrationRule >() );

   using Mesh = typename TrialFiniteElementSpace::mesh_type;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian = typename Mesh::cell_type::jacobian;

   static_assert(
      TrialFiniteElementSpace::Dim >= TestFiniteElementSpace::Dim,
      "NodalSubspaceProjection requires the trial dimension to be at least "
      "the test dimension."
   );
   constexpr Integer TestDim = TestFiniteElementSpace::Dim;

   auto u = ReadDofs( kernel_conf, trial_fe_space, trial_element_index, dofs_in );

   auto Bu = InterpolateValues( kernel_conf, trial_element_quad_data, u );

   const auto cell = mesh::GetCell( trial_fe_space, trial_element_index );

   // Container to store values at all the quadrature points
   auto DBu = MakeQuadraturePointValuesContainer( kernel_conf, TrialIntegrationRule{} );

   // Application of the QFunction
   QuadraturePointLoop< TrialIntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      PhysicalCoordinates X;
      Jacobian J_mesh;

      cell.GetValuesAndJacobian( quad_index, mesh_quad_data, X, J_mesh );

      constexpr Integer n_tuple_elements = std::tuple_size< Jacobian >{};
      using sub_indices = make_index_sequence_without< n_tuple_elements, 0 >;

      const Real detJ = Determinant( get_subtuple( J_mesh, sub_indices{} ) );

      auto sub_quad_index = quad_index.template Sub< 0, TestDim >();

      const Real test_weight = GetWeight( sub_quad_index, test_element_quad_data );
      const Real trial_weight = GetWeight( quad_index, trial_element_quad_data );
      const Real weight = trial_weight / test_weight;

      const Real Bu_q = ReadQuadratureLocalValues( kernel_conf, quad_index, Bu );

      const Real Du_q_w = weight * detJ * Bu_q;

      // WriteAddQuadratureLocalValues( kernel_conf, sub_quad_index, Du_q_w, DBu );
      WriteQuadratureLocalValues( kernel_conf, quad_index, Du_q_w, DBu );
   } );

   // Collocation identifies the retained quadrature values with test DoFs.
   auto DBu_low =
      AggregateDimensions< TrialIntegrationRule, TestIntegrationRule >(
         kernel_conf,
         DBu,
         std::make_index_sequence< TestDim >{} );

   // Assumes tensor mesh
   GlobalIndex test_element_index = trial_fe_space.template GetStructuredSubIndex< 0 >( trial_element_index );
   WriteAddDofs( kernel_conf, test_fe_space, test_element_index, DBu_low, dofs_out );
}

/**
 * @brief Apply nodal subspace projection to a process-local DG field.
 * 
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam TrialIntegrationRule Higher-dimensional trial quadrature rule.
 * @tparam TestIntegrationRule Collocated quadrature rule for retained axes.
 * @tparam TrialFiniteElementSpace Higher-dimensional scalar tensor space.
 * @tparam TestFiniteElementSpace Collocated scalar tensor test space.
 * @tparam MeshQuadData The type of the mesh data structure needed to perform computation at quadrature points.
 * @tparam TrialElementQuadData Trial-space interpolation and quadrature data.
 * @tparam TestElementQuadData Test-space interpolation and quadrature data.
 * @param trial_fe_space Higher-dimensional trial finite-element space.
 * @param test_fe_space Retained nodal test finite-element space.
 * @param mesh_quad_data The mesh data at quadrature points needed to perform the computation.
 * @param trial_element_quad_data The trial finite element data at quadrature points need to perform the computation.
 * @param test_element_quad_data The test finite element data at quadrature points need to perform the computation.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename TrialIntegrationRule,
   typename TestIntegrationRule,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename MeshQuadData,
   typename TrialElementQuadData,
   typename TestElementQuadData >
void NodalSubspaceProjectionOperator(
   const TrialFiniteElementSpace & trial_fe_space,
   const TestFiniteElementSpace & test_fe_space,
   const MeshQuadData & mesh_quad_data,
   const TrialElementQuadData & trial_element_quad_data,
   const TestElementQuadData & test_element_quad_data,
   const element_tensor_view_t< TrialFiniteElementSpace, const Real > & dofs_in,
   element_tensor_view_t< TestFiniteElementSpace, Real > & dofs_out )
{
   GENDIL_REQUIRE_UNBATCHED_OPERATOR( KernelConfiguration );

   mesh::CellIterator< KernelConfiguration >(
      trial_fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem =
            Max(
               required_shared_memory_v< KernelConfiguration, TrialIntegrationRule >, // Interpolation
               required_shared_memory_v< KernelConfiguration, TestIntegrationRule >, // Interpolation
               Product( typename TestIntegrationRule::points::num_points_tensor{} ) // Accumulation at quadrature point
            );
         using Context =
            KernelContext< KernelConfiguration, required_shared_mem >;
         GENDIL_SHARED Real _shared_mem[Context::shared_memory_block_size];

         Context kernel_conf( _shared_mem );

         NodalSubspaceProjectionElementOperator<
            TrialIntegrationRule,
            TestIntegrationRule >(
            kernel_conf,
            trial_fe_space,
            test_fe_space,
            element_index,
            mesh_quad_data,
            trial_element_quad_data,
            test_element_quad_data,
            dofs_in,
            dofs_out );
      }
   );
}

/**
 * @brief Integrate trailing tensor axes into a collocated nodal subspace.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam TrialIntegrationRule Higher-dimensional trial quadrature rule.
 * @tparam TestIntegrationRule Collocated quadrature rule for retained axes.
 */
template <
   typename KernelPolicy,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename TrialIntegrationRule,
   typename TestIntegrationRule >
class NodalSubspaceProjection
#ifdef GENDIL_USE_MFEM
: public mfem::Operator
#endif // GENDIL_USE_MFEM
{
   static_assert(
      nodal_subspace_projection_detail::ValidateCompatibility<
         TrialFiniteElementSpace,
         TestFiniteElementSpace,
         TrialIntegrationRule,
         TestIntegrationRule >() );

protected:
   using trial_fe_space_type = TrialFiniteElementSpace;
   using test_fe_space_type = TestFiniteElementSpace;
   using trial_finite_element_type = typename trial_fe_space_type::finite_element_type;
   using test_finite_element_type = typename test_fe_space_type::finite_element_type;

   using trial_integration_rule = TrialIntegrationRule;
   using test_integration_rule = TestIntegrationRule; // TODO: Should this be automatically deduced?

   const trial_fe_space_type & trial_finite_element_space;
   const test_fe_space_type & test_finite_element_space;
   const trial_integration_rule & trial_int_rule;
   const test_integration_rule & test_int_rule;

   using Mesh = typename TrialFiniteElementSpace::mesh_type;

   /**
    * @brief Data structure storing information to compute physical coordinates
    * and mesh jacobians at the provided integration rule quadrature points.
    * 
    */
   using MeshQuadData = typename Mesh::cell_type::template QuadData< TrialIntegrationRule >;
   MeshQuadData mesh_quad_data;

   /**
    * @brief Data structure storing information to compute field values
    * and field gradients on the trial finite element space at the provided
    * integration rule quadrature points.
    * 
    */
   using TrialElementQuadData = decltype(
                              MakeDofToQuad<
                                 typename trial_finite_element_type::shape_functions,
                                 trial_integration_rule
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
                                 typename test_finite_element_type::shape_functions,
                                 test_integration_rule
                              >()
                           );
   TestElementQuadData test_element_quad_data;

   using input = element_tensor_view_t< TrialFiniteElementSpace, const Real >;
   using output = element_tensor_view_t< TestFiniteElementSpace, Real >;

public:
   /**
    * @brief Construct a nodal subspace projection operator.
    * 
    * @param trial_finite_element_space The trial finite element space associated to the operator.
    * @param test_finite_element_space The test finite element space associated to the operator.
    * @param trial_int_rule The integration rule used by the operator.
    * @param test_int_rule The integration rule used by the operator.
    */
   NodalSubspaceProjection(
      const TrialFiniteElementSpace & trial_finite_element_space,
      const TestFiniteElementSpace & test_finite_element_space,
      const TrialIntegrationRule & trial_int_rule,
      const TestIntegrationRule & test_int_rule ) :
#ifdef GENDIL_USE_MFEM
         mfem::Operator( GetAlgebraicDofExtent( test_finite_element_space ), GetAlgebraicDofExtent( trial_finite_element_space ) ),
#endif
         trial_finite_element_space( trial_finite_element_space ),
         test_finite_element_space( test_finite_element_space ),
         trial_int_rule( trial_int_rule ),
         test_int_rule( test_int_rule ),
         mesh_quad_data{},
         trial_element_quad_data{},
         test_element_quad_data{}
   { }

   /**
    * @brief Apply the nodal subspace projection operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      // TODO: dofs_out = 0.0
      NodalSubspaceProjectionOperator<
         KernelPolicy,
         trial_integration_rule,
         test_integration_rule >(
         this->trial_finite_element_space,
         this->test_finite_element_space,
         this->mesh_quad_data,
         this->trial_element_quad_data,
         this->test_element_quad_data,
         dofs_in,
         dofs_out );
   }

   void operator()( const Vector & dofs_vector_in, Vector & dofs_vector_out ) const
   {
      dofs_vector_out = 0.0;
      auto dofs_in = MakeReadOnlyElementTensorView< KernelPolicy >( this->trial_finite_element_space, dofs_vector_in );
      auto dofs_out = MakeReadWriteElementTensorView< KernelPolicy >( this->test_finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }

   #ifdef GENDIL_USE_MFEM
   /**
    * @brief Apply the nodal subspace projection operator.
    * 
    * @param dofs_vector_in The input degrees of freedom.
    * @param dofs_vector_out The output degrees of freedom.
    */
   void Mult( const mfem::Vector & dofs_vector_in,
              mfem::Vector & dofs_vector_out ) const override
   {
      dofs_vector_out = 0.0;
      auto dofs_in = MakeReadOnlyElementTensorView< KernelPolicy >( this->trial_finite_element_space, dofs_vector_in );
      auto dofs_out = MakeReadWriteElementTensorView< KernelPolicy >( this->test_finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }
   #endif // GENDIL_USE_MFEM
};

/**
 * @brief Build a nodal subspace projection with an explicit kernel policy.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam TrialIntegrationRule The type of the integration rule used by the operator on the trial space.
 * @tparam TestIntegrationRule The type of the integration rule used by the operator on the test space.
 * @param trial_finite_element_space The trial finite element space associated to the operator.
 * @param test_finite_element_space The test finite element space associated to the operator.
 * @param trial_int_rule The integration rule used by the operator on the trial space.
 * @param test_int_rule The integration rule used by the operator on the test space.
 * @return The nodal subspace projection operator.
 */
template <
   typename KernelPolicy,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename TrialIntegrationRule,
   typename TestIntegrationRule >
auto MakeNodalSubspaceProjection(
   const TrialFiniteElementSpace & trial_finite_element_space,
   const TestFiniteElementSpace & test_finite_element_space,
   const TrialIntegrationRule & trial_int_rule,
   const TestIntegrationRule & test_int_rule )
{
   return NodalSubspaceProjection<
      KernelPolicy,
      TrialFiniteElementSpace,
      TestFiniteElementSpace,
      TrialIntegrationRule,
      TestIntegrationRule >(
         trial_finite_element_space,
         test_finite_element_space,
         trial_int_rule,
         test_int_rule );
}

/**
 * @brief Build a serial nodal subspace projection operator.
 * 
 * @tparam TrialFiniteElementSpace The trial finite element space associated to the operator.
 * @tparam TestFiniteElementSpace The test finite element space associated to the operator.
 * @tparam TrialIntegrationRule The type of the integration rule used by the operator on the trial space.
 * @tparam TestIntegrationRule The type of the integration rule used by the operator on the test space.
 * @param trial_finite_element_space The trial finite element space associated to the operator.
 * @param test_finite_element_space The test finite element space associated to the operator.
 * @param trial_int_rule The integration rule used by the operator on the trial space.
 * @param test_int_rule The integration rule used by the operator on the test space.
 * @return The nodal subspace projection operator.
 */
template <
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename TrialIntegrationRule,
   typename TestIntegrationRule >
auto MakeNodalSubspaceProjection(
   const TrialFiniteElementSpace & trial_finite_element_space,
   const TestFiniteElementSpace & test_finite_element_space,
   const TrialIntegrationRule & trial_int_rule,
   const TestIntegrationRule & test_int_rule )
{
   using KernelPolicy = SerialKernelConfiguration;

   return NodalSubspaceProjection<
      KernelPolicy,
      TrialFiniteElementSpace,
      TestFiniteElementSpace,
      TrialIntegrationRule,
      TestIntegrationRule >(
         trial_finite_element_space,
         test_finite_element_space,
         trial_int_rule,
         test_int_rule );
}

} // namespace gendil

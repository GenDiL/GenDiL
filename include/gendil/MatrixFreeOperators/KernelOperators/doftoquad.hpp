// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"
#include "gendil/Utilities/MathHelperFunctions/sqrt.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/lagrangeshapefunctions.hpp"

namespace gendil {

/**
 * @brief Structure storing the quadrature weights, the values of the shape functions
 * at quadrature points, and the values.
 * 
 * @tparam ShapeFunctions The shape functions.
 * @tparam IntegrationRule The integration rule.
 */
template < typename ShapeFunctions, typename IntegrationRule >
struct DofToQuad
{
   using shape_functions = ShapeFunctions;
   using integration_rule = IntegrationRule;
   static constexpr Integer num_dofs = ShapeFunctions::num_dofs;
   static constexpr Integer num_quads = IntegrationRule::GetNumPoints();
   Real weights_data[ num_quads ];
   Real values_data[ num_dofs ][ num_quads ];
   Real gradients_data[ num_dofs ][ num_quads ];
   Real grad_q[ num_quads ][ num_quads ];

   GENDIL_HOST_DEVICE
   constexpr DofToQuad()
   {
      for ( Integer quad = 0; quad < num_quads; quad++ )
      {
         weights_data[ quad ] = IntegrationRule::GetWeight( quad );
         for ( Integer dof = 0; dof < num_dofs; dof++ )
         {
            // TODO: Use points instead of coord.
            values_data[ dof ][ quad ] = ShapeFunctions::ComputeValue( dof, IntegrationRule::GetCoord( quad ) );
            gradients_data[ dof ][ quad ] = ShapeFunctions::ComputeGradientValue( dof, IntegrationRule::GetCoord( quad ) );
         }
      }

      // TODO: maybe use projection instead? that is, grad_q = G * inv(M) * B * diag(w)
      using gl = LagrangeShapeFunctions< IntegrationRule >;
      for ( Integer i = 0; i < num_quads; ++i )
      {
         for ( Integer j = 0; j < num_quads; ++j )
         {
            grad_q[ i ][ j ] = gl::ComputeGradientValue( j, IntegrationRule::GetCoord( i ) );
         }
      }
   }

   constexpr Real values( LocalIndex q, LocalIndex d ) const
   {
      return values_data[ d ][ q ];
   }

   constexpr Real gradients( LocalIndex q, LocalIndex d ) const
   {
      return gradients_data[ d ][ q ];
   }

   /// @brief G*B' gradient operation from quadrature points to quadrature points
   constexpr Real quad_gradients( LocalIndex i, LocalIndex j ) const
   {
      return grad_q[ i ][ j ];
   }

   constexpr Real weights( LocalIndex q ) const
   {
      return weights_data[ q ];
   }
};

/**
 * @brief Creates a tuple of DofToQuad objects corresponding to the 1D shape functions
 * evaluated at the 1D quadrature rule.
 * 
 * @tparam FiniteElement The type of the finite element.
 * @tparam IntRule The type of the integration rule.
 * @return A tuple of DofToQuad for each dimension of the finite element.
 * 
 * @note Currently assumes tensor finite element and tensor integration rule.
 */
// TODO: type check that IntTule is a tuple IntRule?
template < typename FiniteElement, typename IntRule > 
auto MakeDofToQuad( );

template < typename FiniteElement, typename IntRule, size_t... Is > 
auto MakeDofToQuad( std::index_sequence< Is... > )
{
   return std::make_tuple(
             DofToQuad<
                std::tuple_element_t< Is, typename FiniteElement::shape_functions::shape_functions_1d_tuple >,
                std::tuple_element_t< Is, typename IntRule::points::points_1d_tuple > >{}... );
}

// FIXME: IntRules assumes a specific struct for the IntRule: std::tuple<IntRules...>
template < typename FiniteElement, typename IntRule > 
auto MakeDofToQuad( )
{
   static_assert(
      FiniteElement::space_dim == IntRule::space_dim,
      "Finite Element and Integration Rule have different space dimensions." );

   // FIXME: Assumes tensor product points
   return MakeDofToQuad< FiniteElement, IntRule >( std::make_index_sequence< std::tuple_size_v< typename IntRule::points::points_1d_tuple > >{} );
}

/**
 * @brief Returns DofToQuad objects for each face in a tuple.
 * 
 * @tparam FiniteElement The type of the finite element.
 * @tparam FaceIntRulesTuple A tuple containing an integration rule for each face.
 * @return A tuple storing a DofToQuad objects for each face.
 * 
 * @note For tensor finite elements with tensor integration rules this returns a tuple of tuple of 1D DofToQuad.
 */
template < typename FiniteElement, typename FaceIntRulesTuple > 
auto MakeFaceDofToQuad( );

template < typename FiniteElement, typename FaceIntRulesTuple, size_t... Is > 
auto MakeFaceDofToQuad( std::index_sequence< Is... > )
{
   return std::make_tuple( MakeDofToQuad< FiniteElement, std::tuple_element_t< Is, FaceIntRulesTuple > >()... );
}

template < typename FiniteElement, typename FaceIntRulesTuple > 
auto MakeFaceDofToQuad( )
{
   return MakeFaceDofToQuad< FiniteElement, FaceIntRulesTuple >( std::make_index_sequence< std::tuple_size_v< FaceIntRulesTuple > >{} );
}

// Mesh Face quadrature data
template < typename CellType, typename FaceIntRulesTuple, size_t... Is > 
auto MakeMeshFaceQuadData( std::index_sequence< Is... > )
{
   return std::make_tuple( typename CellType::template QuadData< std::tuple_element_t< Is, FaceIntRulesTuple > >{}... );
}

template < typename CellType, typename FaceIntRulesTuple >
auto MakeMeshFaceQuadData()
{
   return MakeFaceDofToQuad< CellType, FaceIntRulesTuple >( std::make_index_sequence< std::tuple_size_v< FaceIntRulesTuple > >{} );
}

// Assumes 1D QuadData
template < typename QuadData >
struct is_face_interpolation
{
   static constexpr bool value = QuadData::num_quads == 1;
};

template < typename... QuadData >
struct is_face_interpolation< std::tuple< QuadData... > >
{
   static constexpr bool value = ( is_face_interpolation< QuadData >::value || ... );
};

template < typename QuadData >
static constexpr bool is_face_interpolation_v = is_face_interpolation< QuadData >::value;

}

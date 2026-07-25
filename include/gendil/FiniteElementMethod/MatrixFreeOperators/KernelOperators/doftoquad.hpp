// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <tuple>
#include <type_traits>

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"
#include "gendil/Utilities/MathHelperFunctions/sqrt.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/shapefunctions.hpp"
#include "gendil/Utilities/tensorproductdata.hpp"

namespace gendil {

struct CachedDofToQuadPolicy
{
};

struct ComputedDofToQuadPolicy
{
};

template < typename KernelConfiguration, typename = void >
struct cell_dof_to_quad_policy
{
   using type = CachedDofToQuadPolicy;
};

template < typename KernelConfiguration >
struct cell_dof_to_quad_policy<
   KernelConfiguration,
   std::void_t< typename KernelConfiguration::cell_dof_to_quad_policy > >
{
   using type = typename KernelConfiguration::cell_dof_to_quad_policy;
};

template < typename KernelConfiguration >
using cell_dof_to_quad_policy_t =
   typename cell_dof_to_quad_policy<
      std::remove_cvref_t< KernelConfiguration > >::type;

template < typename KernelConfiguration, typename = void >
struct face_dof_to_quad_policy
{
   using type = CachedDofToQuadPolicy;
};

template < typename KernelConfiguration >
struct face_dof_to_quad_policy<
   KernelConfiguration,
   std::void_t< typename KernelConfiguration::face_dof_to_quad_policy > >
{
   using type = typename KernelConfiguration::face_dof_to_quad_policy;
};

template < typename KernelConfiguration >
using face_dof_to_quad_policy_t =
   typename face_dof_to_quad_policy<
      std::remove_cvref_t< KernelConfiguration > >::type;

/**
 * @brief Concept for "DoF → quadrature" mapping containers.
 *
 * A type `DTQ` models DofToQuadMapping if it provides:
 *  - Associated types:
 *      - `using shape_functions;`
 *      - `using integration_rule;`
 *  - Static constexpr integers:
 *      - `DTQ::num_dofs`
 *      - `DTQ::num_quads`
 *    (must be usable in constant-expression contexts)
 *  - Accessors (all const-qualified):
 *      - `Real values(LocalIndex q, LocalIndex d) const;`
 *      - `Real gradients(LocalIndex q, LocalIndex d) const;`
 *      - `Real quad_gradients(LocalIndex i, LocalIndex j) const;`   // G*B'
 *      - `Real weights(LocalIndex q) const;`
 *
 * This is satisfied by @c CachedDofToQuad, @c ComputedDofToQuad,
 * and @c NonconformingDofToQuad.
 */
template <typename DTQ>
concept DofToQuadMapping =
   requires // associated types
   {
      typename DTQ::shape_functions;
      typename DTQ::integration_rule;
   }
   &&
   requires // static constexpr counts (and prove constant-expression-ness)
   {
      { DTQ::num_dofs }  -> std::convertible_to<Integer>;
      { DTQ::num_quads } -> std::convertible_to<Integer>;
      // Must be usable where non-type template params are required:
      std::integral_constant<Integer, DTQ::num_dofs>{};
      std::integral_constant<Integer, DTQ::num_quads>{};
   }
   &&
   requires (const DTQ& a, LocalIndex q, LocalIndex d, LocalIndex i, LocalIndex j)
   {
      { a.values(q, d) }         -> std::convertible_to<Real>;
      { a.gradients(q, d) }      -> std::convertible_to<Real>;
      { a.quad_gradients(i, j) } -> std::convertible_to<Real>;
      { a.weights(q) }           -> std::convertible_to<Real>;
   };

template<DofToQuadMapping... Maps>
using TensorProductDofToQuad =
   TensorProductData<Maps...>;

namespace dof_to_quad_detail
{

template<class T>
struct QuadratureSignature;

template<class T>
struct QuadratureLeafSignature
{
   using type = std::remove_cvref_t<T>;
};

template<class T>
   requires requires
   {
      typename std::remove_cvref_t<T>::points;
   }
struct QuadratureLeafSignature<T>
{
   using type = typename std::remove_cvref_t<T>::points;
};

template<class T>
   requires (
      !requires
      {
         typename std::remove_cvref_t<T>::points;
      } &&
      requires
      {
         typename std::remove_cvref_t<T>::integration_rule;
      })
struct QuadratureLeafSignature<T>
{
   using type = typename std::remove_cvref_t<T>::integration_rule;
};

template<class T>
struct QuadratureSignature : QuadratureLeafSignature<T>
{
};

template<class... Entries>
struct QuadratureSignature<TensorProductData<Entries...>>
{
   using type =
      std::tuple<typename QuadratureSignature<Entries>::type...>;
};

template<class T>
using quadrature_signature_t =
   typename QuadratureSignature<std::remove_cvref_t<T>>::type;

} // namespace dof_to_quad_detail

template<class... ScalarMaps>
struct VectorDofToQuad
{
   static_assert(
      sizeof...(ScalarMaps) > 0,
      "VectorDofToQuad requires at least one component.");

   using components_type = std::tuple<ScalarMaps...>;
   using first_component_type =
      std::tuple_element_t<0, components_type>;

   static_assert(
      (std::is_same_v<
         dof_to_quad_detail::quadrature_signature_t<first_component_type>,
         dof_to_quad_detail::quadrature_signature_t<ScalarMaps>> && ...),
      "VectorDofToQuad requires all components to use the same quadrature "
      "points.");

   components_type components;
};

template<class... ScalarMaps>
GENDIL_HOST_DEVICE
constexpr auto MakeVectorDofToQuad(ScalarMaps&&... maps)
{
   using Data = VectorDofToQuad<std::remove_cvref_t<ScalarMaps>...>;
   using Tuple = typename Data::components_type;
   return Data{
      Tuple{std::forward<ScalarMaps>(maps)...}};
}

template<size_t I, class... ScalarMaps>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetVectorComponent(
   VectorDofToQuad<ScalarMaps...>& data)
{
   return std::get<I>(data.components);
}

template<size_t I, class... ScalarMaps>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetVectorComponent(
   const VectorDofToQuad<ScalarMaps...>& data)
{
   return std::get<I>(data.components);
}

template<size_t I, class... ScalarMaps>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetVectorComponent(
   VectorDofToQuad<ScalarMaps...>&& data)
{
   return std::get<I>(std::move(data.components));
}

template<size_t I, class... ScalarMaps>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetVectorComponent(
   const VectorDofToQuad<ScalarMaps...>&& data)
{
   return std::get<I>(std::move(data.components));
}

template<class T>
struct is_vector_dof_to_quad : std::false_type
{
};

template<class... ScalarMaps>
struct is_vector_dof_to_quad<VectorDofToQuad<ScalarMaps...>>
   : std::true_type
{
};

template<class T>
inline constexpr bool is_vector_dof_to_quad_v =
   is_vector_dof_to_quad<std::remove_cvref_t<T>>::value;

template<size_t I, class QData>
using vector_component_t =
   std::tuple_element_t<
      I,
      typename std::remove_cvref_t<QData>::components_type>;

template<class QData>
inline constexpr size_t vector_component_count_v =
   std::tuple_size_v<
      typename std::remove_cvref_t<QData>::components_type>;

/**
 * @brief Structure storing the quadrature weights, the values of the shape functions
 * at quadrature points, and the values.
 *
 * @tparam ShapeFunctions The shape functions.
 * @tparam Points The points and weights of the integration rule.
 */
template < typename ShapeFunctions, typename Points >
struct CachedDofToQuad
{
   using shape_functions = ShapeFunctions;
   using integration_rule = Points; // !FIXME
   using points = Points;
   static constexpr Integer num_dofs = ShapeFunctions::num_dofs;
   static constexpr Integer num_quads = points::GetNumPoints();
   Real weights_data[ num_quads ];
   Real values_data[ num_dofs ][ num_quads ];
   Real gradients_data[ num_dofs ][ num_quads ];
   Real grad_q[ num_quads ][ num_quads ];

   GENDIL_HOST_DEVICE
   constexpr CachedDofToQuad()
   {
      for ( Integer quad = 0; quad < num_quads; quad++ )
      {
         weights_data[ quad ] = points::GetWeight( quad );
         for ( Integer dof = 0; dof < num_dofs; dof++ )
         {
            // TODO: Use points instead of coord.
            values_data[ dof ][ quad ] = ShapeFunctions::ComputeValue( dof, points::GetCoord( quad ) );
            gradients_data[ dof ][ quad ] = ShapeFunctions::ComputeGradientValue( dof, points::GetCoord( quad ) );
         }
      }

      // TODO: maybe use projection instead? that is, grad_q = G * inv(M) * B * diag(w)
      using gl = LagrangeShapeFunctions< points >;
      for ( Integer i = 0; i < num_quads; ++i )
      {
         for ( Integer j = 0; j < num_quads; ++j )
         {
            grad_q[ i ][ j ] = gl::ComputeGradientValue( j, points::GetCoord( i ) );
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
 * @brief Computed/on-demand DoF-to-quadrature map.
 *
 * This representation preserves the @c DofToQuadMapping accessor
 * convention but stores no value, gradient, or quadrature-gradient
 * tables. It is intended as an execution/resource policy for kernels
 * where cached map payloads dominate compile-time or launch resources.
 */
template < typename ShapeFunctions, typename Points >
struct ComputedDofToQuad
{
   using shape_functions = ShapeFunctions;
   using integration_rule = Points; // !FIXME
   using points = Points;
   static constexpr Integer num_dofs = ShapeFunctions::num_dofs;
   static constexpr Integer num_quads = points::GetNumPoints();

   constexpr Real values( LocalIndex q, LocalIndex d ) const
   {
      return ShapeFunctions::ComputeValue( d, points::GetCoord( q ) );
   }

   constexpr Real gradients( LocalIndex q, LocalIndex d ) const
   {
      return ShapeFunctions::ComputeGradientValue( d, points::GetCoord( q ) );
   }

   /// @brief G*B' gradient operation from quadrature points to quadrature points.
   constexpr Real quad_gradients( LocalIndex i, LocalIndex j ) const
   {
      using gl = LagrangeShapeFunctions< points >;
      return gl::ComputeGradientValue( j, points::GetCoord( i ) );
   }

   constexpr Real weights( LocalIndex q ) const
   {
      return points::GetWeight( q );
   }
};

/**
 * @brief Creates strongly typed tensor or vector DoF-to-quadrature data.
 * 
 * @tparam FiniteElement The type of the finite element.
 * @tparam IntRule The type of the integration rule.
 * @return TensorProductDofToQuad for scalar tensor elements, or
 * VectorDofToQuad for vector elements.
 * 
 * @note Currently assumes tensor finite element and tensor integration rule.
 */
// TODO: type check that IntTule is a tuple IntRule?
template < typename ShapeFunctions, typename IntRule > 
auto MakeDofToQuad( );

template < typename Policy, typename ShapeFunctions, typename IntRule >
auto MakeDofToQuadForPolicy( );

template < typename Policy, typename ShapeFunctions, typename IntRule, size_t... Is >
auto MakeTensorDofToQuadForPolicy( std::index_sequence< Is... > )
{
   return MakeTensorProductData(
      MakeDofToQuadForPolicy<
         Policy,
         std::tuple_element_t<
            Is,
            typename ShapeFunctions::shape_functions_1d_tuple>,
         std::tuple_element_t<
            Is,
            typename IntRule::points::points_1d_tuple>>()...);
}

template < typename Policy, typename ShapeFunctions, typename IntRule, size_t... Is >
auto MakeVectorDofToQuadForPolicy( std::index_sequence< Is... > )
{
   return MakeVectorDofToQuad(
      MakeDofToQuadForPolicy<
         Policy,
         std::tuple_element_t< Is, typename ShapeFunctions::scalar_shape_functions_tuple >,
         IntRule>()... );
}

// FIXME: IntRules assumes a specific struct for the IntRule: std::tuple<IntRules...>
template < typename Policy, typename ShapeFunctions, typename IntRule >
auto MakeDofToQuadForPolicy( )
{
   if constexpr ( requires { IntRule::space_dim; } )
   {
      static_assert(
         ShapeFunctions::dim == IntRule::space_dim,
         "Shape functions and Integration Rule have different space dimensions." );
   }

   if constexpr ( is_tensor_shape_functions_v< ShapeFunctions > )
   {
      return MakeTensorDofToQuadForPolicy< Policy, ShapeFunctions, IntRule >(
         std::make_index_sequence< std::tuple_size_v< typename IntRule::points::points_1d_tuple > >{} );
   }
   else if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      return MakeVectorDofToQuadForPolicy< Policy, ShapeFunctions, IntRule >(
         std::make_index_sequence< ShapeFunctions::vector_dim >{} );
   }
   else
   {
      if constexpr ( std::is_same_v< Policy, ComputedDofToQuadPolicy > )
      {
         return ComputedDofToQuad< ShapeFunctions, IntRule >{};
      }
      else
      {
         return CachedDofToQuad< ShapeFunctions, IntRule >{};
      }
   }
}

template < typename ShapeFunctions, typename IntRule >
auto MakeDofToQuad( )
{
   return MakeDofToQuadForPolicy<
      CachedDofToQuadPolicy,
      ShapeFunctions,
      IntRule >();
}

template < typename KernelConfiguration, typename ShapeFunctions, typename IntRule >
auto MakeDofToQuad( )
{
   using policy = cell_dof_to_quad_policy_t< KernelConfiguration >;
   static_assert(
      std::is_same_v< policy, CachedDofToQuadPolicy >,
      "Computed cell DofToQuad maps are not implemented in this milestone." );
   return MakeDofToQuadForPolicy<
      CachedDofToQuadPolicy,
      ShapeFunctions,
      IntRule >();
}

/**
 * @brief Returns DofToQuad objects for each face in a tuple.
 * 
 * @tparam ShapeFunctions The shape functions used in the finite element.
 * @tparam FaceIntRulesTuple A tuple containing an integration rule for each face.
 * @return A tuple storing a DofToQuad objects for each face.
 * 
 * @note The outer tuple remains indexed by local face. Each face entry is a
 * TensorProductDofToQuad or VectorDofToQuad.
 */
template < typename ShapeFunctions, typename FaceIntRulesTuple > 
auto MakeFaceDofToQuad( );

template < typename Policy, typename ShapeFunctions, typename FaceIntRulesTuple, size_t... Is >
auto MakeFaceDofToQuadForPolicy( std::index_sequence< Is... > )
{
   return std::make_tuple(
      MakeDofToQuadForPolicy<
         Policy,
         ShapeFunctions,
         std::tuple_element_t< Is, FaceIntRulesTuple > >()... );
}

template < typename Policy, typename ShapeFunctions, typename FaceIntRulesTuple >
auto MakeFaceDofToQuadForPolicy( )
{
   return MakeFaceDofToQuadForPolicy<
      Policy,
      ShapeFunctions,
      FaceIntRulesTuple >(
         std::make_index_sequence<
            std::tuple_size_v< FaceIntRulesTuple > >{} );
}

template < typename ShapeFunctions, typename FaceIntRulesTuple > 
auto MakeFaceDofToQuad( )
{
   return MakeFaceDofToQuadForPolicy<
      CachedDofToQuadPolicy,
      ShapeFunctions,
      FaceIntRulesTuple >();
}

template < typename KernelConfiguration, typename ShapeFunctions, typename FaceIntRulesTuple >
auto MakeFaceDofToQuad( )
{
   return MakeFaceDofToQuadForPolicy<
      face_dof_to_quad_policy_t< KernelConfiguration >,
      ShapeFunctions,
      FaceIntRulesTuple >();
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


/**
 * @brief Structure storing the quadrature weights, the values of the shape functions
 * at quadrature points, and the values.
 * 
 * @tparam ShapeFunctions The shape functions.
 * @tparam IntegrationRule The integration rule.
 */
template < typename ShapeFunctions, typename IntegrationRule, typename Face, Integer DimIndex >
struct NonconformingDofToQuad
{
   using shape_functions = ShapeFunctions;
   using integration_rule = IntegrationRule;
   using face_type = Face;
   static constexpr Integer num_dofs = ShapeFunctions::num_dofs;
   static constexpr Integer num_quads = IntegrationRule::GetNumPoints();
   Real weights_data[ num_quads ];
   Face face;

   GENDIL_HOST_DEVICE
   constexpr NonconformingDofToQuad( const Face & face ) : face( face )
   {
      for ( Integer quad = 0; quad < num_quads; quad++ )
      {
         weights_data[ quad ] = IntegrationRule::GetWeight( quad );
      }
   }

   constexpr Real values( LocalIndex q, LocalIndex d ) const
   {
      return ShapeFunctions::ComputeValue( d, face.template MapReferenceToFaceCoordinates1d<DimIndex>( IntegrationRule::GetCoord( q ) ) );
   }

   constexpr Real gradients( LocalIndex q, LocalIndex d ) const
   {
      return ShapeFunctions::ComputeGradientValue( d, face.template MapReferenceToFaceCoordinates1d<DimIndex>( IntegrationRule::GetCoord( q ) ) );
   }

   /// @brief G*B' gradient operation from quadrature points to quadrature points
   constexpr Real quad_gradients( LocalIndex i, LocalIndex j ) const
   {
      using gl = LagrangeShapeFunctions< IntegrationRule >;
      return gl::ComputeGradientValue( j, face.template MapReferenceToFaceCoordinates1d<DimIndex>( IntegrationRule::GetCoord( i ) ) );
   }

   constexpr Real weights( LocalIndex q ) const
   {
      return weights_data[ q ];
   }
};

template <
   CellFaceView Face,
   typename ShapeFunctions,
   typename IntegrationRule,
   Integer DimIndex>
GENDIL_HOST_DEVICE
auto MakeNonconformingDofToQuadData(
   const Face & face,
   const CachedDofToQuad< ShapeFunctions, IntegrationRule > & dtq,
   std::integral_constant<Integer, DimIndex>)
{
   return NonconformingDofToQuad< ShapeFunctions, IntegrationRule, Face, DimIndex >( face );
}

template <
   CellFaceView Face,
   typename ... DofToQuads,
   Integer ... Is >
GENDIL_HOST_DEVICE
auto MakeNonconformingDofToQuadData(
   const Face& face,
   const TensorProductData<DofToQuads...>& dtq,
   std::index_sequence<Is...>)
{
   return MakeTensorProductData(
      MakeNonconformingDofToQuadData(
         face,
         GetTensorProductEntry<Is>(dtq),
         std::integral_constant<Integer, Is>{})...);
}

template <
   CellFaceView Face,
   typename ... DofToQuads >
GENDIL_HOST_DEVICE
auto MakeNonconformingDofToQuadData(
   const Face& face,
   const TensorProductData<DofToQuads...>& dtq)
{
   return MakeNonconformingDofToQuadData(
      face,
      dtq,
      std::make_index_sequence<sizeof...(DofToQuads)>{});
}

// Assumes 1D QuadData
template < typename QuadData >
struct is_face_interpolation
{
   static constexpr bool value = QuadData::num_quads == 1;
};

template < typename... QuadData >
struct is_face_interpolation<
   TensorProductData<QuadData...>>
{
   static constexpr bool value = ( is_face_interpolation< QuadData >::value || ... );
};

template < typename... ScalarMaps >
struct is_face_interpolation<VectorDofToQuad<ScalarMaps...>>
{
   static constexpr bool value =
      (is_face_interpolation<ScalarMaps>::value || ...);
};

template < typename QuadData >
static constexpr bool is_face_interpolation_v = is_face_interpolation< QuadData >::value;

}

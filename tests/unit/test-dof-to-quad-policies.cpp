// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>
#include <tuple>
#include <type_traits>

using namespace gendil;

namespace
{

template < typename BaseKernelPolicy >
struct KernelPolicyWithComputedFaceDofToQuad : public BaseKernelPolicy
{
   using face_dof_to_quad_policy = ComputedDofToQuadPolicy;
};

template < typename T >
struct is_tuple : std::false_type
{
};

template < typename... Ts >
struct is_tuple< std::tuple< Ts... > > : std::true_type
{
};

template < typename T >
inline constexpr bool is_tuple_v =
   is_tuple< std::remove_cvref_t< T > >::value;

template < typename T >
struct all_cached_dof_to_quad : std::false_type
{
};

template < typename ShapeFunctions, typename Points >
struct all_cached_dof_to_quad<
   CachedDofToQuad< ShapeFunctions, Points > > : std::true_type
{
};

template < typename... Ts >
struct all_cached_dof_to_quad< std::tuple< Ts... > >
   : std::bool_constant< ( all_cached_dof_to_quad< Ts >::value && ... ) >
{
};

template < typename T >
inline constexpr bool all_cached_dof_to_quad_v =
   all_cached_dof_to_quad< std::remove_cvref_t< T > >::value;

template < typename T >
struct all_computed_dof_to_quad : std::false_type
{
};

template < typename ShapeFunctions, typename Points >
struct all_computed_dof_to_quad<
   ComputedDofToQuad< ShapeFunctions, Points > > : std::true_type
{
};

template < typename... Ts >
struct all_computed_dof_to_quad< std::tuple< Ts... > >
   : std::bool_constant< ( all_computed_dof_to_quad< Ts >::value && ... ) >
{
};

template < typename T >
inline constexpr bool all_computed_dof_to_quad_v =
   all_computed_dof_to_quad< std::remove_cvref_t< T > >::value;

bool CheckClose(
   const Real value,
   const Real reference,
   const char * label,
   const Real tolerance = 1e-14 )
{
   const Real diff = std::abs( value - reference );
   const Real scale =
      std::max( Real{ 1.0 }, std::abs( reference ) );
   if ( diff > tolerance * scale )
   {
      std::cout
         << "FAILED: " << label
         << " value=" << value
         << " reference=" << reference
         << " diff=" << diff << '\n';
      return false;
   }
   return true;
}

template < DofToQuadMapping CachedMap, DofToQuadMapping ComputedMap >
bool CompareDofToQuadMap(
   const CachedMap & cached,
   const ComputedMap & computed,
   const char * label )
{
   static_assert( CachedMap::num_dofs == ComputedMap::num_dofs );
   static_assert( CachedMap::num_quads == ComputedMap::num_quads );

   bool success = true;
   for ( LocalIndex q = 0; q < CachedMap::num_quads; ++q )
   {
      success =
         CheckClose( computed.weights( q ), cached.weights( q ), label ) &&
         success;
      for ( LocalIndex d = 0; d < CachedMap::num_dofs; ++d )
      {
         success =
            CheckClose( computed.values( q, d ), cached.values( q, d ), label ) &&
            success;
         success =
            CheckClose(
               computed.gradients( q, d ),
               cached.gradients( q, d ),
               label ) &&
            success;
      }
   }

   for ( LocalIndex i = 0; i < CachedMap::num_quads; ++i )
   {
      for ( LocalIndex j = 0; j < CachedMap::num_quads; ++j )
      {
         success =
            CheckClose(
               computed.quad_gradients( i, j ),
               cached.quad_gradients( i, j ),
               label ) &&
            success;
      }
   }

   return success;
}

template < typename Cached, typename Computed >
bool CompareDofToQuadData(
   const Cached & cached,
   const Computed & computed,
   const char * label );

template <
   typename CachedTuple,
   typename ComputedTuple,
   size_t... Is >
bool CompareDofToQuadTuple(
   const CachedTuple & cached,
   const ComputedTuple & computed,
   const char * label,
   std::index_sequence< Is... > )
{
   return (
      CompareDofToQuadData(
         std::get< Is >( cached ),
         std::get< Is >( computed ),
         label ) && ... );
}

template < typename Cached, typename Computed >
bool CompareDofToQuadData(
   const Cached & cached,
   const Computed & computed,
   const char * label )
{
   if constexpr ( is_tuple_v< Cached > )
   {
      static_assert( is_tuple_v< Computed > );
      static_assert(
         std::tuple_size_v< std::remove_cvref_t< Cached > > ==
         std::tuple_size_v< std::remove_cvref_t< Computed > > );
      return CompareDofToQuadTuple(
         cached,
         computed,
         label,
         std::make_index_sequence<
            std::tuple_size_v< std::remove_cvref_t< Cached > > >{} );
   }
   else
   {
      return CompareDofToQuadMap( cached, computed, label );
   }
}

template < typename VectorType >
bool VectorFinite( const VectorType & values )
{
   for ( Integer i = 0; i < values.Size(); ++i )
   {
      if ( !std::isfinite( values[ i ] ) )
      {
         return false;
      }
   }
   return true;
}

template < typename VectorType >
bool VectorsClose(
   const VectorType & values,
   const VectorType & reference,
   const Real tolerance )
{
   Real diff_norm2 = 0.0;
   Real ref_norm2 = 0.0;
   for ( Integer i = 0; i < values.Size(); ++i )
   {
      const Real diff = values[ i ] - reference[ i ];
      diff_norm2 += diff * diff;
      ref_norm2 += reference[ i ] * reference[ i ];
   }

   const Real diff_norm = std::sqrt( diff_norm2 );
   const Real ref_norm = std::sqrt( ref_norm2 );
   const Real scale =
      ref_norm > Real{ 1.0 } ? ref_norm : Real{ 1.0 };
   return diff_norm <= tolerance * scale || diff_norm <= tolerance;
}

template < Integer Dim >
struct TestAdvectionVelocity
{
   GENDIL_HOST_DEVICE
   void operator()(
      const std::array< Real, Dim > & X,
      Real ( & velocity )[ Dim ] ) const
   {
      for ( Integer d = 0; d < Dim; ++d )
      {
         velocity[ d ] =
            0.5 +
            0.125 * static_cast< Real >( d + 1 ) +
            0.25 * X[ d ];
      }
   }
};

template < typename KernelPolicy, typename FiniteElementSpace, typename FaceMeshes, typename Rule >
Vector ApplyExplicitGlobalFaceAdvection(
   const FiniteElementSpace & fe_space,
   const FaceMeshes & face_meshes,
   const Rule & rule,
   const Vector & x,
   const Vector & baseline )
{
   constexpr Integer Dim = FiniteElementSpace::Dim;
   using face_integration_rules =
      decltype( GetFaceIntegrationRules( Rule{} ) );
   using mesh_type = typename FiniteElementSpace::mesh_type;
   using shape_functions =
      typename FiniteElementSpace::finite_element_type::shape_functions;

   Vector y = baseline;
   auto dofs_in =
      MakeReadOnlyElementTensorView< KernelPolicy >( fe_space, x );
   auto dofs_out =
      MakeWriteOnlyElementTensorView< KernelPolicy >( fe_space, y );
   auto mesh_face_quad_data =
      MakeMeshFaceQuadData< mesh_type >( face_integration_rules{} );
   auto element_face_quad_data =
      MakeFaceDofToQuad<
         KernelPolicy,
         shape_functions,
         face_integration_rules >();
   auto adv = TestAdvectionVelocity< Dim >{};
   auto boundary_field = Empty{};

   mesh::ForEachFaceMesh(
      face_meshes,
      [&] ( const auto & face_mesh ) mutable
      {
         AdvectionExplicitFaceOperator<
            KernelPolicy,
            Rule,
            face_integration_rules >(
               fe_space,
               face_mesh,
               mesh_face_quad_data,
               element_face_quad_data,
               adv,
               boundary_field,
               dofs_in,
               dofs_out );
      } );

   return y;
}

bool TestGlobalFaceAdvectionEquivalence()
{
   using CachedKernel = HostKernelConfiguration;
   using ComputedKernel =
      KernelPolicyWithComputedFaceDofToQuad< HostKernelConfiguration >;

   constexpr Integer order = 2;
   constexpr Integer num_quad_1d = order + 2;
   std::array< GlobalIndex, 2 > extents{ 3, 4 };

   auto mesh = Cartesian2DMesh( 1.0, extents[ 0 ], extents[ 1 ] );
   auto face_meshes =
      make_cartesian_interior_face_connectivity< 2 >( extents );
   auto finite_element =
      MakeLegendreFiniteElement( FiniteElementOrders< order, order >{} );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
   auto rule =
      MakeIntegrationRule(
         IntegrationRuleNumPoints<
            num_quad_1d,
            num_quad_1d >{} );

   const GlobalIndex num_dofs =
      fe_space.GetNumberOfFiniteElementDofs();
   Vector x(
      static_cast< Integer >( num_dofs ),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.25 +
            0.03125 * static_cast< Real >( i ) +
            0.0078125 * static_cast< Real >( ( i * 13 ) % 17 );
      } );
   Vector baseline(
      static_cast< Integer >( num_dofs ),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return -0.125 +
            0.00390625 * static_cast< Real >( ( i * 7 ) % 19 );
      } );

   const auto cached =
      ApplyExplicitGlobalFaceAdvection< CachedKernel >(
         fe_space,
         face_meshes,
         rule,
         x,
         baseline );
   const auto computed =
      ApplyExplicitGlobalFaceAdvection< ComputedKernel >(
         fe_space,
         face_meshes,
         rule,
         x,
         baseline );

   if ( !VectorFinite( cached ) || !VectorFinite( computed ) )
   {
      std::cout << "FAILED: global-face advection produced non-finite values.\n";
      return false;
   }

   if ( !VectorsClose( computed, cached, 1e-12 ) )
   {
      std::cout
         << "FAILED: computed face DofToQuad global-face advection output "
         << "does not match cached output.\n";
      return false;
   }

   return true;
}

} // namespace

int main()
{
   using Shape1DOrder0 = GaussLegendreShapeFunctions< 0 >;
   using Points1DOrder0 = GaussLegendrePoints< 2 >;
   using Shape1DOrder3 = GaussLegendreShapeFunctions< 3 >;
   using Points1DOrder3 = GaussLegendrePoints< 5 >;

   static_assert(
      DofToQuadMapping<
         CachedDofToQuad< Shape1DOrder3, Points1DOrder3 > > );
   static_assert(
      DofToQuadMapping<
         ComputedDofToQuad< Shape1DOrder3, Points1DOrder3 > > );

   using TensorShape3D =
      TensorShapeFunctions<
         GaussLegendreShapeFunctions< 2 >,
         GaussLegendreShapeFunctions< 2 >,
         GaussLegendreShapeFunctions< 2 > >;
   using TensorRule3D =
      decltype(
         MakeIntegrationRule(
            IntegrationRuleNumPoints< 4, 4, 4 >{} ) );
   using FaceRules3D =
      decltype( GetFaceIntegrationRules( TensorRule3D{} ) );
   using TensorShape4D =
      TensorShapeFunctions<
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 > >;
   using TensorRule4D =
      decltype(
         MakeIntegrationRule(
            IntegrationRuleNumPoints< 5, 5, 5, 5 >{} ) );
   using FaceRules4D =
      decltype( GetFaceIntegrationRules( TensorRule4D{} ) );

   using ComputedFaceKernel =
      KernelPolicyWithComputedFaceDofToQuad< HostKernelConfiguration >;

   using DefaultCellMap =
      decltype( MakeDofToQuad< TensorShape3D, TensorRule3D >() );
   using PolicyCellMap =
      decltype(
         MakeDofToQuad<
            ComputedFaceKernel,
            TensorShape3D,
            TensorRule3D >() );
   using DefaultFaceMap =
      decltype( MakeFaceDofToQuad< TensorShape3D, FaceRules3D >() );
   using ComputedFaceMap =
      decltype(
         MakeFaceDofToQuad<
            ComputedFaceKernel,
            TensorShape3D,
            FaceRules3D >() );

   static_assert( all_cached_dof_to_quad_v< DefaultCellMap > );
   static_assert( all_cached_dof_to_quad_v< PolicyCellMap > );
   static_assert( all_cached_dof_to_quad_v< DefaultFaceMap > );
   static_assert( all_computed_dof_to_quad_v< ComputedFaceMap > );
   static_assert(
      std::is_same_v<
         face_dof_to_quad_policy_t< HostKernelConfiguration >,
         CachedDofToQuadPolicy > );
   static_assert(
      std::is_same_v<
         face_dof_to_quad_policy_t< ComputedFaceKernel >,
         ComputedDofToQuadPolicy > );
   static_assert(
      std::is_same_v<
         cell_dof_to_quad_policy_t< ComputedFaceKernel >,
         CachedDofToQuadPolicy > );

   using TensorShape5DP3 =
      TensorShapeFunctions<
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 > >;
   using TensorRule5DP3 =
      decltype(
         MakeIntegrationRule(
            IntegrationRuleNumPoints< 5, 5, 5, 5, 5 >{} ) );
   using FaceRules5DP3 =
      decltype( GetFaceIntegrationRules( TensorRule5DP3{} ) );
   using TensorShape6DP3 =
      TensorShapeFunctions<
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 > >;
   using TensorRule6DP3 =
      decltype(
         MakeIntegrationRule(
            IntegrationRuleNumPoints< 5, 5, 5, 5, 5, 5 >{} ) );
   using FaceRules6DP3 =
      decltype( GetFaceIntegrationRules( TensorRule6DP3{} ) );

   static_assert(
      sizeof(
         decltype( MakeFaceDofToQuadForPolicy<
            ComputedDofToQuadPolicy,
            TensorShape4D,
            FaceRules4D >() ) ) <
      sizeof(
         decltype( MakeFaceDofToQuad<
            TensorShape4D,
            FaceRules4D >() ) ) / 8 );
   static_assert(
      sizeof(
         decltype( MakeFaceDofToQuadForPolicy<
            ComputedDofToQuadPolicy,
            TensorShape5DP3,
            FaceRules5DP3 >() ) ) <
      sizeof(
         decltype( MakeFaceDofToQuad<
            TensorShape5DP3,
            FaceRules5DP3 >() ) ) / 8 );
   static_assert(
      sizeof(
         decltype( MakeFaceDofToQuadForPolicy<
            ComputedDofToQuadPolicy,
            TensorShape6DP3,
            FaceRules6DP3 >() ) ) <
      sizeof(
         decltype( MakeFaceDofToQuad<
            TensorShape6DP3,
            FaceRules6DP3 >() ) ) / 8 );

   bool success = true;
   success =
      CompareDofToQuadMap(
         CachedDofToQuad< Shape1DOrder0, Points1DOrder0 >{},
         ComputedDofToQuad< Shape1DOrder0, Points1DOrder0 >{},
         "1D p=0" ) &&
      success;
   success =
      CompareDofToQuadMap(
         CachedDofToQuad< Shape1DOrder3, Points1DOrder3 >{},
         ComputedDofToQuad< Shape1DOrder3, Points1DOrder3 >{},
         "1D p=3" ) &&
      success;
   success =
      CompareDofToQuadData(
         MakeFaceDofToQuad< TensorShape3D, FaceRules3D >(),
         MakeFaceDofToQuadForPolicy<
            ComputedDofToQuadPolicy,
            TensorShape3D,
            FaceRules3D >(),
         "3D face map" ) &&
      success;
   success =
      CompareDofToQuadData(
         MakeFaceDofToQuad< TensorShape4D, FaceRules4D >(),
         MakeFaceDofToQuadForPolicy<
            ComputedDofToQuadPolicy,
            TensorShape4D,
            FaceRules4D >(),
         "4D face map" ) &&
      success;
   success = TestGlobalFaceAdvectionEquivalence() && success;

   return success ? 0 : 1;
}

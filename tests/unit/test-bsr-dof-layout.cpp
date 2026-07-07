// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

using namespace gendil;

namespace
{

using ScalarFE0 = GLFiniteElement< 1, 1 >;
using ScalarFE1 = GLFiniteElement< 2, 1 >;
using ScalarShape0 = typename ScalarFE0::shape_functions;
using ScalarDofShape0 = finite_element_dof_shape_t< ScalarShape0 >;
using VectorFE = decltype( MakeVectorFiniteElement( ScalarFE0{}, ScalarFE1{} ) );
using VectorShape = typename VectorFE::shape_functions;
using VectorSpace = FiniteElementSpace< Cartesian2DMesh, VectorFE, L2Restriction >;
using KernelCtx = KernelContext< SerialKernelConfiguration, 1 >;
using Component0Tag = std::integral_constant< size_t, 0 >;
using Component1Tag = std::integral_constant< size_t, 1 >;
using Component0DofShape = component_dof_shape_t< VectorShape, 0 >;
using Component1DofShape = component_dof_shape_t< VectorShape, 1 >;
using AnisotropicComponent0FE = GLFiniteElement< 1, 1, 1 >;
using AnisotropicComponent1FE = GLFiniteElement< 3, 2, 2 >;
using AnisotropicVectorFE = decltype(
   MakeVectorFiniteElement( AnisotropicComponent0FE{}, AnisotropicComponent1FE{} ) );
using AnisotropicVectorShape = typename AnisotropicVectorFE::shape_functions;
using AnisotropicComponent1DofShape =
   component_dof_shape_t< AnisotropicVectorShape, 1 >;

struct UnknownRestriction {};

template < typename Restriction, typename = void >
struct HasRestrictionTraits : std::false_type {};

template < typename Restriction >
struct HasRestrictionTraits<
   Restriction,
   std::void_t<
      decltype( restriction_traits< Restriction >::is_direct_index_map ),
      decltype( restriction_traits< Restriction >::is_injective ) > >
   : std::true_type {};

static_assert( LocalDofCount< ScalarShape0 >() == 4 );
static_assert(
   FlattenLocalDof< ScalarShape0 >( std::array< GlobalIndex, 2 >{ 1, 1 } ) == 3 );

static_assert( LocalDofCount< VectorShape >() == 10 );
static_assert(
   ComponentLocalDofOffset< VectorShape >( std::integral_constant< size_t, 0 >{} ) == 0 );
static_assert(
   ComponentLocalDofOffset< VectorShape >( std::integral_constant< size_t, 1 >{} ) == 4 );
static_assert(
   FlattenLocalDof< VectorShape >(
      std::integral_constant< size_t, 0 >{},
      std::array< GlobalIndex, 2 >{ 1, 1 } ) == 3 );
static_assert(
   FlattenLocalDof< VectorShape >(
      std::integral_constant< size_t, 1 >{},
      std::array< GlobalIndex, 2 >{ 2, 1 } ) == 9 );

using ZeroVectorType = std::remove_cvref_t<
   decltype( MakeZeroElementVector(
      std::declval< const KernelCtx & >(),
      std::declval< const VectorSpace & >() ) ) >;
using ExpectedVectorDofsType = std::remove_cvref_t<
   decltype( MakeVectorDofs(
      std::declval< const KernelCtx & >(),
      typename VectorShape::dof_shape{},
      std::make_index_sequence< VectorShape::vector_dim >{} ) ) >;

static_assert(
   std::is_same_v< ZeroVectorType, ExpectedVectorDofsType >,
   "MakeZeroElementVector must return the same tuple-of-component container as MakeVectorDofs." );

static_assert( HasRestrictionTraits< L2Restriction >::value );
static_assert( HasRestrictionTraits< H1Restriction >::value );
static_assert( HasRestrictionTraits< VectorH1Restriction< 2 > >::value );
static_assert(
   !HasRestrictionTraits< UnknownRestriction >::value,
   "Unknown restrictions must fail closed until they provide explicit traits." );

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool TestRestrictionTraits()
{
   bool success = true;

   success = Check(
      restriction_traits< L2Restriction >::is_direct_index_map,
      "L2Restriction should be a direct index map." ) && success;
   success = Check(
      restriction_traits< L2Restriction >::is_injective,
      "L2Restriction should be injective." ) && success;
   success = Check(
      restriction_traits< H1Restriction >::is_direct_index_map,
      "H1Restriction should be a direct index map." ) && success;
   success = Check(
      !restriction_traits< H1Restriction >::is_injective,
      "H1Restriction should not be injective." ) && success;
   success = Check(
      restriction_traits< VectorH1Restriction< 2 > >::is_direct_index_map,
      "VectorH1Restriction should preserve direct-index view support." ) && success;
   success = Check(
      !restriction_traits< VectorH1Restriction< 2 > >::is_injective,
      "VectorH1Restriction should not be injective." ) && success;
   success = Check(
      restriction_supports_contiguous_element_view_v< L2Restriction >,
      "L2Restriction should keep the contiguous element-view fast path." ) && success;
   success = Check(
      !restriction_supports_contiguous_element_view_v< H1Restriction >,
      "H1Restriction should use canonical mapped element views." ) && success;

   return success;
}

bool TestScalarFlattenLocalDof()
{
   bool success = true;

   const std::array< GlobalIndex, 2 > i00{ 0, 0 };
   const std::array< GlobalIndex, 2 > i10{ 1, 0 };
   const std::array< GlobalIndex, 2 > i01{ 0, 1 };
   const std::array< GlobalIndex, 2 > i11{ 1, 1 };

   success = Check(
      FlattenLocalDof< ScalarShape0 >( i00 ) == FlattenMultiIndex< ScalarDofShape0 >( i00 ),
      "Scalar local index (0,0) does not match FlattenMultiIndex." ) && success;
   success = Check(
      FlattenLocalDof< ScalarShape0 >( i10 ) == FlattenMultiIndex< ScalarDofShape0 >( i10 ),
      "Scalar local index (1,0) does not match FlattenMultiIndex." ) && success;
   success = Check(
      FlattenLocalDof< ScalarShape0 >( i01 ) == FlattenMultiIndex< ScalarDofShape0 >( i01 ),
      "Scalar local index (0,1) does not match FlattenMultiIndex." ) && success;
   success = Check(
      FlattenLocalDof< ScalarShape0 >( i11 ) == FlattenMultiIndex< ScalarDofShape0 >( i11 ),
      "Scalar local index (1,1) does not match FlattenMultiIndex." ) && success;

   success = Check(
      FlattenLocalDof< ScalarShape0 >( i11 ) == 3,
      "Scalar local index does not follow FIFO ordering." ) && success;

   return success;
}

bool TestVectorFlattenLocalDof()
{
   bool success = true;

   constexpr Component0Tag c0{};
   constexpr Component1Tag c1{};
   const std::array< GlobalIndex, 2 > c0_i11{ 1, 1 };
   const std::array< GlobalIndex, 2 > c1_i00{ 0, 0 };
   const std::array< GlobalIndex, 2 > c1_i21{ 2, 1 };

   const GlobalIndex c0_offset = ComponentLocalDofOffset< VectorShape >( c0 );
   const GlobalIndex c1_offset = ComponentLocalDofOffset< VectorShape >( c1 );

   success = Check( c0_offset == 0, "Unexpected vector component 0 local offset." ) && success;
   success = Check( c1_offset == Product( Component0DofShape{} ), "Unexpected vector component 1 local offset." ) && success;

   success = Check(
      FlattenLocalDof< VectorShape >( c0, c0_i11 ) ==
         c0_offset + FlattenMultiIndex< Component0DofShape >( c0_i11 ),
      "Vector component 0 local index is not component-major." ) && success;
   success = Check(
      FlattenLocalDof< VectorShape >( c1, c1_i00 ) ==
         c1_offset + FlattenMultiIndex< Component1DofShape >( c1_i00 ),
      "Vector component 1 first local index is not component-major." ) && success;
   success = Check(
      FlattenLocalDof< VectorShape >( c1, c1_i21 ) ==
         c1_offset + FlattenMultiIndex< Component1DofShape >( c1_i21 ),
      "Vector component 1 last local index is not component-major." ) && success;

   success = Check(
      FlattenLocalDof< VectorShape >( c1, c1_i21 ) == 9,
      "Vector component-major local index for unequal component shapes is wrong." ) && success;

   return success;
}

bool TestScalarGlobalDofIndex()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto scalar_space = MakeFiniteElementSpace( mesh, ScalarFE0{} );
   auto shifted_space =
      MakeFiniteElementSpace( mesh, ScalarFE0{}, L2Restriction{ 7 } );

   bool success = true;
   const GlobalIndex element_index = 1;
   const std::array< GlobalIndex, 2 > i11{ 1, 1 };
   const GlobalIndex element_dofs = Product( ScalarDofShape0{} );
   const GlobalIndex local_id = FlattenLocalDof( scalar_space, i11 );
   const GlobalIndex zero_based_index =
      element_index * element_dofs + local_id;

   success = Check(
      ScalarLocalDofCount( scalar_space ) == element_dofs,
      "Unexpected scalar local DoF count." ) && success;
   success = Check(
      ScalarGlobalTopologyDofCount( scalar_space ) ==
         scalar_space.GetNumberOfFiniteElements() * element_dofs,
      "Unexpected scalar L2 global topology DoF count." ) && success;
   success = Check(
      scalar_space.GetNumberOfFiniteElementDofs() ==
         scalar_space.GetNumberOfFiniteElements() * element_dofs,
      "Unexpected scalar L2 final finite-element DoF count." ) && success;

   success = Check(
      GlobalDofIndex(
         scalar_space,
         element_index,
         i11 ) == zero_based_index,
      "Unexpected scalar L2 global DoF index." ) && success;
   success = Check(
      ZeroBasedElementToGlobalDofIndex(
         scalar_space,
         element_index,
         local_id ) == zero_based_index,
      "Scalar L2 zero-based topology index is wrong." ) && success;
   success = Check(
      GlobalDofIndex(
         scalar_space,
         element_index,
         local_id ) ==
         GlobalDofIndex(
            scalar_space,
            element_index,
            i11 ),
      "Scalar L2 scalar-flat GlobalDofIndex disagrees with tensor-index mapping." ) && success;
   success = Check(
      ScalarElementDofOrdinalToGlobalDofIndex(
         scalar_space,
         zero_based_index ) == zero_based_index,
      "Scalar L2 ordinal mapping is wrong." ) && success;

   success = Check(
      ScalarGlobalTopologyDofCount( shifted_space ) ==
         scalar_space.GetNumberOfFiniteElements() * element_dofs,
      "Shifted scalar L2 global topology count should not include the FE-space shift." ) && success;
   success = Check(
      ZeroBasedElementToGlobalDofIndex(
         shifted_space,
         element_index,
         local_id ) == zero_based_index,
      "Shifted scalar L2 zero-based topology index should not include the FE-space shift." ) && success;
   success = Check(
      GlobalDofIndex(
         shifted_space,
         element_index,
         local_id ) == shifted_space.restriction.shift + zero_based_index,
      "Shifted scalar L2 final global index should include the FE-space shift." ) && success;
   success = Check(
      ScalarElementDofOrdinalToGlobalDofIndex(
         shifted_space,
         zero_based_index ) == shifted_space.restriction.shift + zero_based_index,
      "Shifted scalar L2 ordinal mapping should include the FE-space shift." ) && success;

   std::vector< Real > data(
      shifted_space.restriction.shift +
      shifted_space.GetNumberOfFiniteElementDofs() );
   for ( GlobalIndex i = 0; i < data.size(); ++i )
   {
      data[i] = static_cast< Real >( i );
   }

   auto element_view =
      MakeScalarElementTensorView( shifted_space, data.data() );
   const GlobalIndex shifted_global_index =
      GlobalDofIndex( shifted_space, element_index, local_id );
   success = Check(
      element_view( i11[0], i11[1], element_index ) ==
         data[shifted_global_index],
      "Shifted scalar L2 ElementTensorView read disagrees with canonical mapping." ) && success;

   element_view( i11[0], i11[1], element_index ) = -3.0;
   success = Check(
      data[shifted_global_index] == -3.0,
      "Shifted scalar L2 ElementTensorView write disagrees with canonical mapping." ) && success;

   return success;
}

bool TestScalarH1GlobalDofIndex()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   const std::array< int, 8 > restriction_map{
      0, 1, 3, 4,
      1, 2, 4, 5
   };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction restriction{ restriction_indices, 6 };
   auto h1_space = MakeFiniteElementSpace( mesh, ScalarFE0{}, restriction );

   const std::array< GlobalIndex, 2 > i00{ 0, 0 };
   const std::array< GlobalIndex, 2 > i10{ 1, 0 };
   const std::array< GlobalIndex, 2 > i01{ 0, 1 };
   const std::array< GlobalIndex, 2 > i11{ 1, 1 };

   bool success = true;

   success = Check(
      ScalarLocalDofCount( h1_space ) == 4,
      "Unexpected scalar H1 local DoF count." ) && success;
   success = Check(
      ScalarGlobalTopologyDofCount( h1_space ) == 6,
      "Unexpected scalar H1 topology DoF count." ) && success;
   success = Check(
      h1_space.GetNumberOfFiniteElementDofs() == 6,
      "Unexpected scalar H1 final finite-element DoF count." ) && success;

   success = Check(
      FlattenLocalDof( h1_space, i00 ) == 0,
      "Q1 local ordering for (0,0) should be local id 0." ) && success;
   success = Check(
      FlattenLocalDof( h1_space, i10 ) == 1,
      "Q1 local ordering for (1,0) should be local id 1." ) && success;
   success = Check(
      FlattenLocalDof( h1_space, i01 ) == 2,
      "Q1 local ordering for (0,1) should be local id 2." ) && success;
   success = Check(
      FlattenLocalDof( h1_space, i11 ) == 3,
      "Q1 local ordering for (1,1) should be local id 3." ) && success;

   success = Check(
      GlobalDofIndex( h1_space, 0, i00 ) == 0,
      "H1 element 0 local id 0 should map to global DoF 0." ) && success;
   success = Check(
      GlobalDofIndex( h1_space, 0, i10 ) == 1,
      "H1 element 0 local id 1 should map to global DoF 1." ) && success;
   success = Check(
      GlobalDofIndex( h1_space, 0, i01 ) == 3,
      "H1 element 0 local id 2 should map to global DoF 3." ) && success;
   success = Check(
      GlobalDofIndex( h1_space, 0, i11 ) == 4,
      "H1 element 0 local id 3 should map to global DoF 4." ) && success;

   success = Check(
      GlobalDofIndex( h1_space, 1, i00 ) == 1,
      "H1 element 1 local id 0 should map to global DoF 1." ) && success;
   success = Check(
      GlobalDofIndex( h1_space, 1, i10 ) == 2,
      "H1 element 1 local id 1 should map to global DoF 2." ) && success;
   success = Check(
      GlobalDofIndex( h1_space, 1, i01 ) == 4,
      "H1 element 1 local id 2 should map to global DoF 4." ) && success;
   success = Check(
      GlobalDofIndex( h1_space, 1, i11 ) == 5,
      "H1 element 1 local id 3 should map to global DoF 5." ) && success;

   const GlobalIndex local00 = FlattenLocalDof( h1_space, i00 );
   const GlobalIndex local10 = FlattenLocalDof( h1_space, i10 );
   const GlobalIndex local01 = FlattenLocalDof( h1_space, i01 );
   const GlobalIndex local11 = FlattenLocalDof( h1_space, i11 );

   success = Check(
      GlobalDofIndex( h1_space, 0, local00 ) ==
         GlobalDofIndex( h1_space, 0, i00 ),
      "Scalar H1 scalar-flat mapping disagrees for element 0 local id 0." ) && success;
   success = Check(
      GlobalDofIndex( h1_space, 0, local10 ) ==
         GlobalDofIndex( h1_space, 0, i10 ),
      "Scalar H1 scalar-flat mapping disagrees for element 0 local id 1." ) && success;
   success = Check(
      GlobalDofIndex( h1_space, 1, local01 ) ==
         GlobalDofIndex( h1_space, 1, i01 ),
      "Scalar H1 scalar-flat mapping disagrees for element 1 local id 2." ) && success;
   success = Check(
      GlobalDofIndex( h1_space, 1, local11 ) ==
         GlobalDofIndex( h1_space, 1, i11 ),
      "Scalar H1 scalar-flat mapping disagrees for element 1 local id 3." ) && success;
   success = Check(
      ZeroBasedElementToGlobalDofIndex( h1_space, 0, local11 ) ==
         GlobalDofIndex( h1_space, 0, local11 ),
      "Scalar H1 zero-based topology index should already be the final global DoF." ) && success;

   const GlobalIndex element0_local01_ordinal =
      0 * ScalarLocalDofCount( h1_space ) +
      static_cast< GlobalIndex >( local01 );
   const GlobalIndex element1_local00_ordinal =
      1 * ScalarLocalDofCount( h1_space ) +
      static_cast< GlobalIndex >( local00 );

   success = Check(
      ScalarElementDofOrdinalToGlobalDofIndex(
         h1_space,
         element0_local01_ordinal ) ==
         GlobalDofIndex( h1_space, 0, local01 ),
      "Scalar H1 ordinal mapping disagrees for element 0 local id 2." ) && success;
   success = Check(
      ScalarElementDofOrdinalToGlobalDofIndex(
         h1_space,
         element1_local00_ordinal ) ==
         GlobalDofIndex( h1_space, 1, local00 ),
      "Scalar H1 ordinal mapping disagrees for element 1 local id 0." ) && success;

   std::vector< Real > data( h1_space.GetNumberOfFiniteElementDofs() );
   for ( GlobalIndex i = 0; i < data.size(); ++i )
   {
      data[i] = 10.0 + static_cast< Real >( i );
   }

   auto element_view =
      MakeScalarElementTensorView( h1_space, data.data() );
   success = Check(
      element_view( i01[0], i01[1], 0 ) ==
         data[ScalarElementDofOrdinalToGlobalDofIndex(
            h1_space,
            element0_local01_ordinal )],
      "Scalar H1 ElementTensorView read disagrees with canonical mapping." ) && success;
   success = Check(
      element_view( i00[0], i00[1], 1 ) ==
         data[ScalarElementDofOrdinalToGlobalDofIndex(
            h1_space,
            element1_local00_ordinal )],
      "Scalar H1 ElementTensorView read disagrees with shared-node mapping." ) && success;

   element_view( i10[0], i10[1], 0 ) = 42.0;
   success = Check(
      data[GlobalDofIndex( h1_space, 0, local10 )] == 42.0,
      "Scalar H1 ElementTensorView write disagrees with canonical mapping." ) && success;

   return success;
}

bool TestTensorProductH1ElementTensorView()
{
   Cartesian2DMesh mesh0( 1.0, 2, 1 );
   Cartesian2DMesh mesh1( 1.0, 1, 2 );

   const std::array< int, 8 > restriction_map0{
      0, 1, 3, 4,
      1, 2, 4, 5
   };
   const std::array< int, 8 > restriction_map1{
      0, 1, 2, 3,
      2, 3, 4, 5
   };

   HostDevicePointer< const int > restriction_indices0{};
   restriction_indices0.host_pointer = restriction_map0.data();
   HostDevicePointer< const int > restriction_indices1{};
   restriction_indices1.host_pointer = restriction_map1.data();

   H1Restriction restriction0{ restriction_indices0, 6 };
   H1Restriction restriction1{ restriction_indices1, 6 };
   auto factor_space0 =
      MakeFiniteElementSpace( mesh0, ScalarFE0{}, restriction0 );
   auto factor_space1 =
      MakeFiniteElementSpace( mesh1, ScalarFE0{}, restriction1 );
   auto product_mesh = MakeCartesianProductMesh( mesh0, mesh1 );
   auto product_restriction =
      MakeTensorProductRestriction( factor_space0, factor_space1 );
   using ProductRestriction =
      std::remove_cvref_t< decltype( product_restriction ) >;
   auto product_space =
      MakeFiniteElementSpace(
         product_mesh,
         GLFiniteElement< 1, 1, 1, 1 >{},
         product_restriction );

   static_assert( is_tensor_product_restriction_v< ProductRestriction > );
   static_assert(
      restriction_traits< ProductRestriction >::is_direct_index_map );
   static_assert(
      !restriction_traits< ProductRestriction >::is_injective );

   constexpr GlobalIndex e0 = 1;
   constexpr GlobalIndex e1 = 1;
   constexpr GlobalIndex ne0 = 2;
   constexpr GlobalIndex product_element = e0 + ne0 * e1;
   const std::array< GlobalIndex, 2 > factor0_indices{ 0, 1 };
   const std::array< GlobalIndex, 2 > factor1_indices{ 1, 0 };
   const std::array< GlobalIndex, 4 > product_indices{
      factor0_indices[0],
      factor0_indices[1],
      factor1_indices[0],
      factor1_indices[1]
   };

   const GlobalIndex local0 =
      FlattenLocalDof< ScalarShape0 >( factor0_indices );
   const GlobalIndex local1 =
      FlattenLocalDof< ScalarShape0 >( factor1_indices );
   const GlobalIndex g0 =
      GlobalDofIndex( factor_space0, e0, local0 );
   const GlobalIndex g1 =
      GlobalDofIndex( factor_space1, e1, local1 );
   const GlobalIndex n0 = ScalarGlobalTopologyDofCount( factor_space0 );
   const GlobalIndex expected_global = g0 + n0 * g1;

   bool success = true;
   success = Check(
      product_space.GetNumberOfFiniteElementDofs() ==
         ScalarGlobalTopologyDofCount( factor_space0 ) *
         ScalarGlobalTopologyDofCount( factor_space1 ),
      "Tensor-product H1 DoF count should be the product of factor topology counts." ) && success;
   success = Check(
      GlobalDofIndex(
         product_space,
         product_element,
         FlattenLocalDof( product_space, product_indices ) ) ==
         expected_global,
      "Tensor-product H1 scalar-flat mapping did not match factor recombination." ) && success;

   std::vector< Real > data( product_space.GetNumberOfFiniteElementDofs() );
   for ( GlobalIndex i = 0; i < data.size(); ++i )
   {
      data[i] = 100.0 + static_cast< Real >( i );
   }

   auto element_view =
      MakeScalarElementTensorView( product_space, data.data() );
   success = Check(
      element_view(
         product_indices[0],
         product_indices[1],
         product_indices[2],
         product_indices[3],
         product_element ) == data[expected_global],
      "Tensor-product H1 element view did not slice local DoF indices by factor." ) && success;

   element_view(
      product_indices[0],
      product_indices[1],
      product_indices[2],
      product_indices[3],
      product_element ) = -17.0;
   success = Check(
      data[expected_global] == -17.0,
      "Tensor-product H1 element view write did not hit the recombined factor DoF." ) && success;

   return success;
}

bool TestTensorProductThreeFactorElementTensorView()
{
   using Factor0FE = GLFiniteElement< 1 >;
   using Factor1FE = GLFiniteElement< 1, 2 >;
   using Factor2FE = GLFiniteElement< 2 >;
   using Factor0DofShape =
      finite_element_dof_shape_t< typename Factor0FE::shape_functions >;
   using Factor1DofShape =
      finite_element_dof_shape_t< typename Factor1FE::shape_functions >;
   using Factor2DofShape =
      finite_element_dof_shape_t< typename Factor2FE::shape_functions >;

   Cartesian1DMesh mesh0( 1.0, 3 );
   Cartesian2DMesh mesh1( 1.0, 2, 2 );
   Cartesian1DMesh mesh2( 1.0, 2 );

   auto factor_space0 =
      MakeFiniteElementSpace( mesh0, Factor0FE{}, L2Restriction{ 100 } );
   auto factor_space1 =
      MakeFiniteElementSpace( mesh1, Factor1FE{}, L2Restriction{ 200 } );
   auto factor_space2 =
      MakeFiniteElementSpace( mesh2, Factor2FE{}, L2Restriction{ 300 } );
   auto product_mesh = MakeCartesianProductMesh( mesh0, mesh1, mesh2 );
   auto product_restriction =
      MakeTensorProductRestriction(
         factor_space0,
         factor_space1,
         factor_space2 );
   auto product_space =
      MakeFiniteElementSpace(
         product_mesh,
         GLFiniteElement< 1, 1, 2, 2 >{},
         product_restriction );

   constexpr GlobalIndex e0 = 2;
   constexpr GlobalIndex e1 = 3;
   constexpr GlobalIndex e2 = 1;
   constexpr GlobalIndex ne0 = 3;
   constexpr GlobalIndex ne1 = 4;
   constexpr GlobalIndex product_element =
      e0 + ne0 * e1 + ne0 * ne1 * e2;
   const std::array< GlobalIndex, 1 > factor0_indices{ 1 };
   const std::array< GlobalIndex, 2 > factor1_indices{ 1, 2 };
   const std::array< GlobalIndex, 1 > factor2_indices{ 2 };
   const std::array< GlobalIndex, 4 > product_indices{
      factor0_indices[0],
      factor1_indices[0],
      factor1_indices[1],
      factor2_indices[0]
   };

   const GlobalIndex n0 = ScalarGlobalTopologyDofCount( factor_space0 );
   const GlobalIndex n1 = ScalarGlobalTopologyDofCount( factor_space1 );
   const GlobalIndex local0 =
      FlattenMultiIndex< Factor0DofShape >( factor0_indices );
   const GlobalIndex local1 =
      FlattenMultiIndex< Factor1DofShape >( factor1_indices );
   const GlobalIndex local2 =
      FlattenMultiIndex< Factor2DofShape >( factor2_indices );
   const GlobalIndex g0 =
      e0 * Product( Factor0DofShape{} ) + local0;
   const GlobalIndex g1 =
      e1 * Product( Factor1DofShape{} ) + local1;
   const GlobalIndex g2 =
      e2 * Product( Factor2DofShape{} ) + local2;
   const GlobalIndex expected_global = g0 + n0 * g1 + n0 * n1 * g2;

   bool success = true;
   success = Check(
      product_space.GetNumberOfFiniteElementDofs() == n0 * n1 *
         ScalarGlobalTopologyDofCount( factor_space2 ),
      "Three-factor tensor-product DoF count should ignore shifted factor offsets." ) && success;
   success = Check(
      expected_global + 1 ==
         static_cast< GlobalIndex >(
            product_space.GetNumberOfFiniteElementDofs() ),
      "Three-factor test should exercise the highest product topology index." ) && success;
   success = Check(
      GlobalDofIndex(
         product_space,
         product_element,
         FlattenLocalDof( product_space, product_indices ) ) ==
         expected_global,
      "Three-factor tensor-product scalar-flat mapping used the wrong factor stride." ) && success;

   std::vector< Real > data( product_space.GetNumberOfFiniteElementDofs() );
   for ( GlobalIndex i = 0; i < data.size(); ++i )
   {
      data[i] = static_cast< Real >( i );
   }

   auto element_view =
      MakeScalarElementTensorView( product_space, data.data() );
   success = Check(
      element_view(
         product_indices[0],
         product_indices[1],
         product_indices[2],
         product_indices[3],
         product_element ) == data[expected_global],
      "Three-factor tensor-product element view did not ignore factor L2 shifts." ) && success;

   element_view(
      product_indices[0],
      product_indices[1],
      product_indices[2],
      product_indices[3],
      product_element ) = -29.0;
   success = Check(
      data[expected_global] == -29.0,
      "Three-factor tensor-product element view write used the wrong product topology index." ) && success;

   return success;
}

bool TestVectorH1GlobalDofIndex()
{
   const Integer n = 2;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 1;
   FiniteElementOrders< order > orders;
   auto scalar_fe = MakeLobattoFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );

   const std::array< int, 4 > restriction_map{
      0, 1,
      1, 2
   };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction scalar_restriction{ restriction_indices, 3 };
   auto restriction = MakeVectorH1Restriction< 2 >( scalar_restriction );
   auto vector_h1_space =
      MakeFiniteElementSpace( mesh, vector_fe, restriction );

   using VectorH1Space =
      std::remove_cvref_t< decltype( vector_h1_space ) >;
   using VectorH1Shape =
      typename VectorH1Space::finite_element_type::shape_functions;
   constexpr Component0Tag c0{};
   constexpr Component1Tag c1{};
   const std::array< GlobalIndex, 1 > i0{ 0 };
   const std::array< GlobalIndex, 1 > i1{ 1 };

   static_assert(
      is_vector_h1_restriction_v< typename VectorH1Space::restriction_type >,
      "This test must use the explicit vector H1 restriction type." );
   static_assert(
      VectorH1Space::restriction_type::num_comp ==
         VectorH1Shape::vector_dim,
      "VectorH1Restriction<NComp> should match the vector FE component count." );

   bool success = true;

   success = Check(
      vector_h1_space.GetNumberOfFiniteElementDofs() == 6,
      "Vector H1 total true-DoF count should be component-major scalar_count * NComp." ) && success;
   success = Check(
      ScalarGlobalTopologyDofCount( vector_h1_space ) == 3,
      "Vector H1 scalar topology DoF count should match the underlying scalar restriction." ) && success;

   success = Check(
      FlattenLocalDof( vector_h1_space, c0, i0 ) == 0,
      "Vector H1 component 0 local node 0 should flatten to local id 0." ) && success;
   success = Check(
      FlattenLocalDof( vector_h1_space, c0, i1 ) == 1,
      "Vector H1 component 0 local node 1 should flatten to local id 1." ) && success;
   success = Check(
      FlattenLocalDof( vector_h1_space, c1, i0 ) == 2,
      "Vector H1 component 1 local node 0 should flatten after component 0." ) && success;
   success = Check(
      FlattenLocalDof( vector_h1_space, c1, i1 ) == 3,
      "Vector H1 component 1 local node 1 should flatten after component 0." ) && success;

   success = Check(
      GlobalDofIndex( vector_h1_space, c0, 0, i0 ) == 0,
      "Vector H1 component 0 element 0 left node should map to true DoF 0." ) && success;
   success = Check(
      GlobalDofIndex( vector_h1_space, c0, 0, i1 ) == 1,
      "Vector H1 component 0 element 0 right node should map to true DoF 1." ) && success;
   success = Check(
      GlobalDofIndex( vector_h1_space, c0, 1, i0 ) == 1,
      "Vector H1 component 0 shared node should map to true DoF 1 from element 1." ) && success;
   success = Check(
      GlobalDofIndex( vector_h1_space, c0, 1, i1 ) == 2,
      "Vector H1 component 0 element 1 right node should map to true DoF 2." ) && success;

   success = Check(
      GlobalDofIndex( vector_h1_space, c1, 0, i0 ) == 3,
      "Vector H1 component 1 true DoFs should begin after all component 0 DoFs." ) && success;
   success = Check(
      GlobalDofIndex( vector_h1_space, c1, 0, i1 ) == 4,
      "Vector H1 component 1 element 0 right node should map to true DoF 4." ) && success;
   success = Check(
      GlobalDofIndex( vector_h1_space, c1, 1, i0 ) == 4,
      "Vector H1 component 1 shared node should map to true DoF 4 from element 1." ) && success;
   success = Check(
      GlobalDofIndex( vector_h1_space, c1, 1, i1 ) == 5,
      "Vector H1 component 1 element 1 right node should map to true DoF 5." ) && success;

   success = Check(
      GlobalDofIndex( vector_h1_space, c0, 0, i1 ) ==
         GlobalDofIndex( vector_h1_space, c0, 1, i0 ),
      "Vector H1 shared node should be shared within component 0." ) && success;
   success = Check(
      GlobalDofIndex( vector_h1_space, c1, 0, i1 ) ==
         GlobalDofIndex( vector_h1_space, c1, 1, i0 ),
      "Vector H1 shared node should be shared within component 1." ) && success;
   success = Check(
      GlobalDofIndex( vector_h1_space, c0, 0, i0 ) !=
         GlobalDofIndex( vector_h1_space, c1, 0, i0 ),
      "Vector H1 components must not alias each other." ) && success;

   success = Check(
      GlobalDofIndex(
         vector_h1_space,
         c0,
         0,
         FlattenComponentLocalDof< VectorH1Shape >( c0, i1 ) ) ==
         GlobalDofIndex( vector_h1_space, c0, 0, i1 ),
      "Vector H1 component-flat wrapper changed component 0 behavior." ) && success;
   success = Check(
      GlobalDofIndex(
         vector_h1_space,
         c1,
         1,
         FlattenComponentLocalDof< VectorH1Shape >( c1, i0 ) ) ==
         GlobalDofIndex( vector_h1_space, c1, 1, i0 ),
      "Vector H1 component-flat wrapper changed component 1 behavior." ) && success;

   return success;
}

bool TestVectorGlobalDofIndex()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto vector_space = MakeFiniteElementSpace( mesh, VectorFE{} );

   constexpr Component0Tag c0{};
   constexpr Component1Tag c1{};
   using VectorDofShape = typename VectorShape::dof_shape;

   bool success = true;
   const GlobalIndex num_elements = vector_space.GetNumberOfFiniteElements();
   const GlobalIndex element_index = 1;
   const std::array< GlobalIndex, 2 > c0_i11{ 1, 1 };
   const std::array< GlobalIndex, 2 > c1_i21{ 2, 1 };

   const GlobalIndex c0_global_offset =
      VectorOffset( VectorDofShape{}, num_elements, std::make_index_sequence< 0 >{} );
   const GlobalIndex c1_global_offset =
      VectorOffset( VectorDofShape{}, num_elements, std::make_index_sequence< 1 >{} );
   const GlobalIndex c0_ndofs = Product( Component0DofShape{} );
   const GlobalIndex c1_ndofs = Product( Component1DofShape{} );

   success = Check(
      GlobalDofIndex(
         vector_space,
         c0,
         element_index,
         c0_i11 ) ==
         c0_global_offset + element_index * c0_ndofs + FlattenMultiIndex< Component0DofShape >( c0_i11 ),
      "Unexpected vector component 0 L2 global DoF index." ) && success;

   success = Check(
      GlobalDofIndex(
         vector_space,
         c1,
         element_index,
         c1_i21 ) ==
         c1_global_offset + element_index * c1_ndofs + FlattenMultiIndex< Component1DofShape >( c1_i21 ),
      "Unexpected vector component 1 L2 global DoF index." ) && success;

   success = Check(
      GlobalDofIndex(
         vector_space,
         c0,
         element_index,
         FlattenComponentLocalDof< VectorShape >( c0, c0_i11 ) ) ==
         GlobalDofIndex(
            vector_space,
            c0,
            element_index,
            c0_i11 ),
      "Vector L2 component-flat wrapper changed component 0 behavior." ) && success;
   success = Check(
      GlobalDofIndex(
         vector_space,
         c1,
         element_index,
         FlattenComponentLocalDof< VectorShape >( c1, c1_i21 ) ) ==
         GlobalDofIndex(
            vector_space,
            c1,
            element_index,
            c1_i21 ),
      "Vector L2 component-flat wrapper changed component 1 behavior." ) && success;

   std::vector< Real > data( vector_space.GetNumberOfFiniteElementDofs() );
   for ( GlobalIndex i = 0; i < data.size(); ++i )
   {
      data[i] = static_cast< Real >( i );
   }

   auto evector = MakeVectorElementTensorView( vector_space, data.data() );
   const GlobalIndex c0_global_index =
      GlobalDofIndex( vector_space, c0, element_index, c0_i11 );
   const GlobalIndex c1_global_index =
      GlobalDofIndex( vector_space, c1, element_index, c1_i21 );

   success = Check(
      std::get< 0 >( evector )( c0_i11[0], c0_i11[1], element_index ) == data[c0_global_index],
      "Vector GlobalDofIndex disagrees with component 0 ElementTensorView layout." ) && success;
   success = Check(
      std::get< 1 >( evector )( c1_i21[0], c1_i21[1], element_index ) == data[c1_global_index],
      "Vector GlobalDofIndex disagrees with component 1 ElementTensorView layout." ) && success;

   return success;
}

bool TestVectorTrialTraversalAndSetLocalDof()
{
   Real shared_data[1]{};
   KernelCtx kernel_context( shared_data );
   Cartesian2DMesh mesh( 1.0, 1, 1 );
   auto vector_space = MakeFiniteElementSpace( mesh, VectorFE{} );
   auto local_dofs = MakeZeroElementVector( kernel_context, vector_space );

   GlobalIndex count = 0;
   ForEachLocalTrialDof( kernel_context, vector_space, [&] ( const auto & dof )
   {
      const Real value = static_cast< Real >(
         1 + FlattenLocalDof(
            vector_space,
            typename std::remove_cvref_t< decltype(dof) >::component{},
            dof.indices ) );
      SetLocalDofOnOwnerThread( kernel_context, local_dofs, dof, value );
      ++count;
   });

   bool success = true;
   success = Check(
      count == LocalDofCount< VectorShape >(),
      "Vector trial DoF traversal did not visit every local DoF." ) && success;
   success = Check(
      std::get< 0 >( local_dofs )( 1, 1 ) == 4.0,
      "SetLocalDofOnOwnerThread did not set vector component 0 with a compile-time tag." ) && success;
   success = Check(
      std::get< 1 >( local_dofs )( 2, 1 ) == 10.0,
      "SetLocalDofOnOwnerThread did not set vector component 1 with a compile-time tag." ) && success;

   return success;
}

bool TestDescriptorThreadRegisterSplit()
{
   bool success = true;
   bool found_mixed_index = false;
   LocalIndex count = 0;

   ForEachLocalDofWithShapes<
      0,
      false,
      std::index_sequence< 2 >,
      std::index_sequence< 3 > >( [&] ( const auto & dof )
   {
      using Descriptor = std::remove_cvref_t< decltype(dof) >;
      static_assert( Descriptor::thread_dim == 1 );
      static_assert( Descriptor::register_dim == 1 );

      success = Check(
         dof.indices[0] == dof.thread_indices[0],
         "Descriptor full index did not begin with the threaded index." ) && success;
      success = Check(
         dof.indices[1] == dof.register_indices[0],
         "Descriptor full index did not append the register index." ) && success;

      if ( dof.thread_indices[0] == 1 && dof.register_indices[0] == 2 )
      {
         found_mixed_index = true;
         success = Check(
            dof.indices == std::array< GlobalIndex, 2 >{ 1, 2 },
            "Descriptor full index for {thread=1, register=2} is wrong." ) && success;
      }

      ++count;
   });

   success = Check(
      count == 6,
      "Split descriptor traversal did not visit every explicit thread/register pair." ) && success;
   success = Check(
      found_mixed_index,
      "Split descriptor traversal did not exercise a nontrivial threaded/register pair." ) && success;

   return success;
}

bool TestAnisotropicDescriptorOrientationMapping()
{
   Cartesian3DMesh mesh( 1.0, 1, 1, 1 );
   auto vector_space = MakeFiniteElementSpace( mesh, AnisotropicVectorFE{} );

   constexpr Component1Tag c1{};
   const auto reference_dof = MakeLocalDofDescriptor(
      c1,
      std::true_type{},
      std::index_sequence< 4 >{},
      std::index_sequence< 3, 3 >{},
      std::array< GlobalIndex, 1 >{ 1 },
      std::array< GlobalIndex, 2 >{ 2, 0 } );

   const Permutation< 3 > orientation{ { 2, -1, 3 } };
   const auto native_dof =
      OrientReferenceDofToNative( vector_space, reference_dof, orientation );

   const std::array< GlobalIndex, 3 > expected_native_indices{ 3, 2, 0 };
   const auto dof_sizes = to_array( AnisotropicComponent1DofShape{} );
   const auto oriented_layout = MakeOrientedLayout( dof_sizes, orientation );
   const auto legacy_native_indices =
      ReferenceToNativeIndex( reference_dof.indices, dof_sizes, orientation );

   bool success = true;
   success = Check(
      native_dof.indices == expected_native_indices,
      "Anisotropic vector descriptor orientation did not map to the expected native index." ) && success;
   success = Check(
      native_dof.thread_indices == std::array< GlobalIndex, 1 >{ 3 },
      "Anisotropic vector descriptor orientation did not preserve the thread/register split." ) && success;
   success = Check(
      native_dof.register_indices == std::array< GlobalIndex, 2 >{ 2, 0 },
      "Anisotropic vector descriptor orientation did not preserve native register indices." ) && success;
   success = Check(
      oriented_layout.Offset(
         native_dof.indices[0],
         native_dof.indices[1],
         native_dof.indices[2] ) ==
         FlattenMultiIndex< AnisotropicComponent1DofShape >( reference_dof.indices ),
      "Anisotropic vector native index does not match the oriented-layout reference offset." ) && success;
   success = Check(
      legacy_native_indices != native_dof.indices,
      "Anisotropic orientation smoke check no longer exercises the vector-specific layout path." ) && success;

   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestRestrictionTraits() && success;
   success = TestScalarFlattenLocalDof() && success;
   success = TestVectorFlattenLocalDof() && success;
   success = TestScalarGlobalDofIndex() && success;
   success = TestScalarH1GlobalDofIndex() && success;
   success = TestTensorProductH1ElementTensorView() && success;
   success = TestTensorProductThreeFactorElementTensorView() && success;
   success = TestVectorH1GlobalDofIndex() && success;
   success = TestVectorGlobalDofIndex() && success;
   success = TestVectorTrialTraversalAndSetLocalDof() && success;
   success = TestDescriptorThreadRegisterSplit() && success;
   success = TestAnisotropicDescriptorOrientationMapping() && success;

   return success ? 0 : 1;
}

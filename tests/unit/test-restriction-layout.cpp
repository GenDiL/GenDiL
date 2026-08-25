// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictions.hpp"

#include <array>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace restriction_layout_test {

template < size_t EntryCount, bool ReferenceView >
struct SyntheticRestriction
{
   using dof_shape_type = std::index_sequence< 2 >;
   static constexpr gendil::Integer tensor_dim = 1;
   static constexpr bool supports_reference_view = ReferenceView;

   gendil::GlobalIndex shift;
};

template < size_t EntryCount, bool ReferenceView >
gendil::GlobalIndex GetNumberOfLocalDofs(
   const SyntheticRestriction< EntryCount, ReferenceView > & )
{
   return 4;
}

template < size_t EntryCount, bool ReferenceView >
gendil::GlobalIndex GetNumberOfGlobalDofs(
   const SyntheticRestriction< EntryCount, ReferenceView > & )
{
   return 4;
}

template < size_t EntryCount, bool ReferenceView >
gendil::GlobalIndex GetAlgebraicDofExtent(
   const SyntheticRestriction< EntryCount, ReferenceView > & restriction )
{
   return restriction.shift + 4;
}

template < size_t EntryCount, bool ReferenceView, typename Visitor >
void ForEachRestrictionEntry(
   const SyntheticRestriction< EntryCount, ReferenceView > & restriction,
   const gendil::GlobalIndex element,
   const std::array< gendil::GlobalIndex, 1 > & local_dof,
   Visitor && visitor )
{
   for ( size_t entry = 0; entry < EntryCount; ++entry )
   {
      std::forward< Visitor >( visitor )(
         restriction.shift + element * 2 + local_dof[0] + entry,
         gendil::RestrictionUnitWeight{} );
   }
}

using ThirdPartyRestriction = SyntheticRestriction< 1, true >;
using MultiEntryRestriction = SyntheticRestriction< 2, true >;
using NoReferenceViewRestriction = SyntheticRestriction< 1, false >;

template < typename Restriction >
struct FactorSpace
{
   using restriction_type = Restriction;

   Restriction restriction;
   gendil::GlobalIndex num_elements;

   constexpr gendil::GlobalIndex GetNumberOfFiniteElements() const
   {
      return num_elements;
   }
};

template < typename Component0, typename Component1 >
struct ThirdPartyVectorRestriction
{
   static constexpr gendil::Integer num_components = 2;

   std::tuple< Component0, Component1 > components;
   gendil::GlobalIndex num_local_dofs;
   gendil::GlobalIndex num_global_dofs;
   gendil::GlobalIndex algebraic_dof_extent;
};

template < size_t Component, typename Component0, typename Component1 >
constexpr decltype(auto) GetComponentRestriction(
   const ThirdPartyVectorRestriction< Component0, Component1 > & restriction )
{
   return std::get< Component >( restriction.components );
}

template <
   size_t Component,
   gendil::Integer Dim,
   typename Component0,
   typename Component1,
   typename Visitor >
constexpr void ForEachRestrictionEntry(
   const ThirdPartyVectorRestriction< Component0, Component1 > & restriction,
   const gendil::GlobalIndex element,
   const gendil::LocalComponentDoFIndex< Component, Dim > & local_dof,
   Visitor && visitor )
{
   ForEachRestrictionEntry(
      GetComponentRestriction< Component >( restriction ),
      element,
      local_dof.local_dof,
      std::forward< Visitor >( visitor ) );
}

template < typename Component0, typename Component1 >
constexpr gendil::GlobalIndex GetNumberOfLocalDofs(
   const ThirdPartyVectorRestriction< Component0, Component1 > & restriction )
{
   return restriction.num_local_dofs;
}

template < typename Component0, typename Component1 >
constexpr gendil::GlobalIndex GetNumberOfGlobalDofs(
   const ThirdPartyVectorRestriction< Component0, Component1 > & restriction )
{
   return restriction.num_global_dofs;
}

template < typename Component0, typename Component1 >
constexpr gendil::GlobalIndex GetAlgebraicDofExtent(
   const ThirdPartyVectorRestriction< Component0, Component1 > & restriction )
{
   return restriction.algebraic_dof_extent;
}

struct UnknownCardinalityRestriction
{
   using dof_shape_type = std::index_sequence< 2 >;
   static constexpr gendil::Integer tensor_dim = 1;
};

inline gendil::GlobalIndex GetNumberOfLocalDofs(
   const UnknownCardinalityRestriction & )
{
   return 2;
}

inline gendil::GlobalIndex GetNumberOfGlobalDofs(
   const UnknownCardinalityRestriction & )
{
   return 2;
}

inline gendil::GlobalIndex GetAlgebraicDofExtent(
   const UnknownCardinalityRestriction & )
{
   return 2;
}

template < typename Visitor >
void ForEachRestrictionEntry(
   const UnknownCardinalityRestriction &,
   gendil::GlobalIndex,
   const std::array< gendil::GlobalIndex, 1 > & local_dof,
   Visitor && visitor )
{
   std::forward< Visitor >( visitor )(
      local_dof[0],
      gendil::RestrictionUnitWeight{} );
}

} // namespace restriction_layout_test

namespace gendil {

template < size_t EntryCount, bool ReferenceView >
inline constexpr size_t static_restriction_entry_count_v<
   restriction_layout_test::SyntheticRestriction<
      EntryCount,
      ReferenceView > > = EntryCount;

template < size_t EntryCount >
struct restriction_supports_element_reference_view<
   restriction_layout_test::SyntheticRestriction<
      EntryCount,
      true > > : std::true_type { };

template < bool ReferenceView >
inline constexpr bool restriction_may_share_global_dofs_v<
   restriction_layout_test::SyntheticRestriction<
      1,
      ReferenceView > > = false;

} // namespace gendil

namespace {

using namespace gendil;
using restriction_layout_test::MultiEntryRestriction;
using restriction_layout_test::NoReferenceViewRestriction;
using restriction_layout_test::ThirdPartyRestriction;
using restriction_layout_test::UnknownCardinalityRestriction;

template < typename Restriction >
concept HasRestrictionLayout =
   requires( const Restriction & restriction )
   {
      MakeRestrictionLayout( restriction );
   };

template < typename Restriction >
concept HasRestrictionElementView =
   requires( const Restriction & restriction, Real * data )
   {
      MakeRestrictionElementView( restriction, data );
   };

template < typename Layout, typename... Indices >
concept HasLayoutOffset =
   requires( const Layout & layout, Indices... indices )
   {
      { layout.Offset( indices... ) } -> std::same_as< GlobalIndex >;
   };

using ScalarShape = std::index_sequence< 2, 3 >;
using ScalarRestriction = ContiguousL2Restriction< ScalarShape >;
using VectorFactorRestriction = VectorRestriction< ScalarRestriction >;
using ScalarLayout = RestrictionLayout< ScalarRestriction >;

static_assert( HasRestrictionLayout< ScalarRestriction > );
static_assert( HasLayoutOffset<
   ScalarLayout,
   GlobalIndex,
   GlobalIndex,
   GlobalIndex > );
static_assert( !HasLayoutOffset<
   ScalarLayout,
   GlobalIndex,
   GlobalIndex > );
static_assert( HasRestrictionLayout< ThirdPartyRestriction > );
static_assert( !HasRestrictionLayout< MultiEntryRestriction > );
static_assert( !HasRestrictionLayout< UnknownCardinalityRestriction > );
static_assert( HasRestrictionLayout< NoReferenceViewRestriction > );
static_assert( HasRestrictionElementView< ThirdPartyRestriction > );
static_assert( !HasRestrictionElementView< NoReferenceViewRestriction > );
static_assert( ElementwiseIndependentRestriction< ThirdPartyRestriction > );

template < typename... FactorRestrictions >
concept CanFormTensorProductRestriction = requires
{
   typename TensorProductRestriction< FactorRestrictions... >;
};

static_assert( CanFormTensorProductRestriction< ScalarRestriction > );
static_assert( CanFormTensorProductRestriction< ThirdPartyRestriction > );
static_assert( !CanFormTensorProductRestriction< MultiEntryRestriction > );
static_assert( !CanFormTensorProductRestriction<
   UnknownCardinalityRestriction > );
static_assert( !CanFormTensorProductRestriction<
   NoReferenceViewRestriction > );
static_assert( !CanFormTensorProductRestriction< VectorFactorRestriction > );

using VectorComponent0 =
   ContiguousL2Restriction< std::index_sequence< 2 > >;
using VectorComponent1 =
   ContiguousL2Restriction< std::index_sequence< 3 > >;
using ThirdPartyVector = restriction_layout_test::ThirdPartyVectorRestriction<
   VectorComponent0,
   VectorComponent1 >;
static_assert( VectorElementDoFRestriction< ThirdPartyVector > );

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool TestScalarLayouts()
{
   bool success = true;

   const ScalarRestriction l2{ 5, 12, 17 };
   const auto l2_layout = MakeRestrictionLayout( l2 );
   for ( GlobalIndex element = 0; element < 2; ++element )
   {
      for ( GlobalIndex j = 0; j < 3; ++j )
      {
         for ( GlobalIndex i = 0; i < 2; ++i )
         {
            success = Check(
               l2_layout.Offset( i, j, element ) ==
                  GetGlobalDofIndex(
                     l2,
                     element,
                     std::array< GlobalIndex, 2 >{ i, j } ),
               "Contiguous L2 RestrictionLayout offset mismatch." ) && success;
         }
      }
   }

   const std::array< int, 4 > map{ 0, 1, 1, 2 };
   HostDevicePointer< const int > map_pointer{};
   map_pointer.host_pointer = map.data();
   const IndirectH1Restriction< std::index_sequence< 2 > > h1{
      map_pointer,
      4,
      4,
      3,
      7 };
   const auto h1_layout = MakeRestrictionLayout( h1 );
   success = Check(
      h1_layout.Offset( 1, 0 ) ==
         GetGlobalDofIndex(
            h1,
            0,
            std::array< GlobalIndex, 1 >{ 1 } ),
      "Indirect H1 RestrictionLayout offset mismatch." ) && success;
   success = Check(
      h1_layout.Offset( 0, 1 ) == h1_layout.Offset( 1, 0 ),
      "Indirect H1 RestrictionLayout did not preserve shared aliases." ) && success;

   std::array< Real, 16 > data{};
   auto h1_view = MakeRestrictionElementView( h1, data.data() );
   h1_view( 1, 0 ) = 7.0;
   success = Check(
      h1_view( 0, 1 ) == 7.0,
      "Restriction-backed H1 view did not preserve reference aliasing." ) && success;

   return success;
}

bool TestTensorProductRestrictionView()
{
   using Factor0Shape = std::index_sequence< 2 >;
   using Factor1Shape = std::index_sequence< 2, 3 >;
   using Factor0Restriction = ContiguousL2Restriction< Factor0Shape >;
   using Factor1Restriction = IndirectH1Restriction< Factor1Shape >;
   using ProductRestriction = TensorProductRestriction<
      Factor0Restriction,
      Factor1Restriction >;

   const std::array< int, 6 > map{ 0, 1, 2, 3, 4, 5 };
   HostDevicePointer< const int > map_pointer{};
   map_pointer.host_pointer = map.data();
   const ProductRestriction restriction{
      std::tuple{
         Factor0Restriction{ 0, 4, 4 },
         Factor1Restriction{ map_pointer, 0, 6, 6, 6 } },
      std::array< GlobalIndex, 2 >{ 1, 2 },
      std::array< GlobalIndex, 2 >{ 1, 4 },
      24,
      24,
      24 };

   const auto layout = MakeRestrictionLayout( restriction );
   bool success = true;
   for ( GlobalIndex element = 0; element < 2; ++element )
   {
      for ( GlobalIndex k = 0; k < 3; ++k )
      {
         for ( GlobalIndex j = 0; j < 2; ++j )
         {
            for ( GlobalIndex i = 0; i < 2; ++i )
            {
               success = Check(
                  layout.Offset( i, j, k, element ) ==
                     GetGlobalDofIndex(
                        restriction,
                        element,
                        std::array< GlobalIndex, 3 >{ i, j, k } ),
                  "Anisotropic tensor-product RestrictionLayout mismatch." ) && success;
            }
         }
      }
   }
   return success;
}

bool TestAlgebraicTensorProductCoordinates()
{
   using Shape = std::index_sequence< 2 >;
   using L2 = ContiguousL2Restriction< Shape >;
   using H1 = IndirectH1Restriction< Shape >;
   using ProductRestriction = TensorProductRestriction< L2, H1 >;

   const std::array< int, 2 > map{ 0, 1 };
   HostDevicePointer< const int > map_pointer{};
   map_pointer.host_pointer = map.data();
   const L2 l2{ 2, 2, 5 };
   const H1 h1{ map_pointer, 3, 2, 2, 6 };
   const ProductRestriction restriction{
      std::tuple{ l2, h1 },
      std::array< GlobalIndex, 2 >{ 1, 1 },
      std::array< GlobalIndex, 2 >{ 1, 5 },
      4,
      4,
      30 };
   ValidateElementDoFRestriction( restriction );

   bool success = true;
   success = Check(
      GetNumberOfGlobalDofs( restriction ) == 4 &&
         GetAlgebraicDofExtent( restriction ) == 30,
      "Tensor product did not distinguish logical count from algebraic extent." ) && success;
   success = Check(
      GetGlobalDofIndex(
         restriction,
         0,
         std::array< GlobalIndex, 2 >{ 1, 0 } ) == 18,
      "Tensor product did not preserve shifted and offset factor coordinates." ) && success;
   success = Check(
      GetGlobalDofIndex(
         restriction,
         0,
         std::array< GlobalIndex, 2 >{ 0, 1 } ) == 22,
      "Tensor product used logical counts instead of algebraic factor strides." ) && success;
   return success;
}

bool TestAssociativeTensorProductCoordinates()
{
   using Shape0 = std::index_sequence< 2 >;
   using Shape1 = std::index_sequence< 3 >;
   using R0 = ContiguousL2Restriction< Shape0 >;
   using R1 = ContiguousL2Restriction< Shape1 >;
   using Product01 = TensorProductRestriction< R0, R1 >;
   using NestedProduct = TensorProductRestriction< Product01, R0 >;
   using DirectProduct = TensorProductRestriction< R0, R1, R0 >;

   const R0 r0{ 0, 2, 2 };
   const R1 r1{ 0, 3, 3 };
   const Product01 product01{
      std::tuple{ r0, r1 },
      std::array< GlobalIndex, 2 >{ 1, 1 },
      std::array< GlobalIndex, 2 >{ 1, 2 },
      6,
      6,
      6 };
   const NestedProduct nested{
      std::tuple{ product01, r0 },
      std::array< GlobalIndex, 2 >{ 1, 1 },
      std::array< GlobalIndex, 2 >{ 1, 6 },
      12,
      12,
      12 };
   const DirectProduct direct{
      std::tuple{ r0, r1, r0 },
      std::array< GlobalIndex, 3 >{ 1, 1, 1 },
      std::array< GlobalIndex, 3 >{ 1, 2, 6 },
      12,
      12,
      12 };
   ValidateElementDoFRestriction( nested );
   ValidateElementDoFRestriction( direct );

   bool success = true;
   for ( GlobalIndex k = 0; k < 2; ++k )
   {
      for ( GlobalIndex j = 0; j < 3; ++j )
      {
         for ( GlobalIndex i = 0; i < 2; ++i )
         {
            const std::array< GlobalIndex, 3 > local{ i, j, k };
            success = Check(
               GetGlobalDofIndex( nested, 0, local ) ==
                  GetGlobalDofIndex( direct, 0, local ),
               "Nested tensor-product coordinates are not associative." ) && success;
         }
      }
   }
   return success;
}

bool TestThirdPartyTensorProductFactor()
{
   using L2 = ContiguousL2Restriction< std::index_sequence< 2 > >;
   using ProductRestriction =
      TensorProductRestriction< ThirdPartyRestriction, L2 >;
   const ProductRestriction restriction{
      std::tuple{ ThirdPartyRestriction{ 3 }, L2{ 0, 2, 2 } },
      std::array< GlobalIndex, 2 >{ 1, 2 },
      std::array< GlobalIndex, 2 >{ 1, 7 },
      8,
      8,
      14 };
   return Check(
      GetGlobalDofIndex(
         restriction,
         1,
         std::array< GlobalIndex, 2 >{ 1, 1 } ) == 13,
      "Third-party tensor-product factor did not use semantic addressing." );
}

bool TestVectorTensorProductFactory()
{
   using A0 = ContiguousL2Restriction< std::index_sequence< 2 > >;
   using A1 = ContiguousL2Restriction< std::index_sequence< 3 > >;
   using B0 = ContiguousL2Restriction< std::index_sequence< 2 > >;
   using B1 = ContiguousL2Restriction< std::index_sequence< 2 > >;
   using VectorA = VectorRestriction< A0, A1 >;
   using VectorB = VectorRestriction< B0, B1 >;

   const VectorA vector_a{
      std::tuple{ A0{ 0, 2, 5 }, A1{ 2, 3, 5 } },
      5,
      5,
      5 };
   const VectorB vector_b{
      std::tuple{ B0{ 0, 2, 4 }, B1{ 2, 2, 4 } },
      4,
      4,
      4 };
   const restriction_layout_test::FactorSpace< VectorA > space_a{
      vector_a,
      1 };
   const restriction_layout_test::FactorSpace< VectorB > space_b{
      vector_b,
      1 };

   bool success = true;
   const auto single_vector_product =
      MakeTensorProductRestriction( space_a );
   static_assert(
      VectorElementDoFRestriction< decltype( single_vector_product ) > );
   success = Check(
      GetNumberOfLocalDofs( single_vector_product ) == 5 &&
         GetNumberOfGlobalDofs( single_vector_product ) == 5 &&
         GetAlgebraicDofExtent( single_vector_product ) == 5 &&
         GetGlobalDofIndex(
            single_vector_product,
            0,
            LocalComponentDoFIndex< 1, 1 >{ { 2 } } ) == 4,
      "A single vector tensor-product factor did not preserve its completed component coordinates." ) && success;

   const auto product = MakeTensorProductRestriction( space_a, space_b );
   static_assert( VectorElementDoFRestriction< decltype( product ) > );
   static_assert( decltype( product )::num_components == 2 );

   success = Check(
      GetNumberOfLocalDofs( product ) == 10 &&
         GetNumberOfGlobalDofs( product ) == 10 &&
         GetAlgebraicDofExtent( product ) == 20,
      "Vector tensor-product dimensions are incorrect." ) && success;
   success = Check(
      GetGlobalDofIndex(
         product,
         0,
         LocalComponentDoFIndex< 0, 2 >{ { 1, 1 } } ) == 6,
      "Vector tensor product did not delegate component zero exactly." ) && success;
   success = Check(
      GetGlobalDofIndex(
         product,
         0,
         LocalComponentDoFIndex< 1, 2 >{ { 2, 1 } } ) == 19,
      "Vector tensor product compacted final component coordinates." ) && success;

   const auto reversed = MakeTensorProductRestriction( space_b, space_a );
   success = Check(
      GetGlobalDofIndex(
         reversed,
         0,
         LocalComponentDoFIndex< 1, 2 >{ { 1, 2 } } ) == 19,
      "Vector tensor-product factor order is incorrect." ) && success;

   using Scalar = ContiguousL2Restriction< std::index_sequence< 2 > >;
   const restriction_layout_test::FactorSpace< Scalar > scalar_space{
      Scalar{ 0, 2, 2 },
      1 };
   const auto scalar_product =
      MakeTensorProductRestriction( scalar_space );
   static_assert( TensorElementDoFRestriction< decltype( scalar_product ) > );
   static_assert( !VectorElementDoFRestriction< decltype( scalar_product ) > );
   success = Check(
      GetNumberOfLocalDofs( scalar_product ) == 2 &&
         GetNumberOfGlobalDofs( scalar_product ) == 2 &&
         GetAlgebraicDofExtent( scalar_product ) == 2 &&
         GetGlobalDofIndex(
            scalar_product,
            0,
            std::array< GlobalIndex, 1 >{ 1 } ) == 1,
      "The all-scalar tensor-product factory path changed its type or coordinates." ) && success;
   const auto broadcast = MakeTensorProductRestriction(
      scalar_space,
      space_a,
      scalar_space );
   static_assert( decltype( broadcast )::num_components == 2 );
   success = Check(
      GetNumberOfLocalDofs( broadcast ) == 20 &&
         GetNumberOfGlobalDofs( broadcast ) == 20 &&
         GetAlgebraicDofExtent( broadcast ) == 20,
      "Scalar factors were not broadcast into every vector component." ) && success;
   success = Check(
      GetGlobalDofIndex(
         broadcast,
         0,
         LocalComponentDoFIndex< 1, 3 >{ { 1, 2, 1 } } ) == 19,
      "Broadcast scalar/vector/scalar tensor-product coordinate is wrong." ) && success;

   const auto interleaved = MakeTensorProductRestriction(
      scalar_space,
      space_a,
      scalar_space,
      space_b,
      scalar_space );
   success = Check(
      GetAlgebraicDofExtent( interleaved ) == 160 &&
         GetGlobalDofIndex(
            interleaved,
            0,
            LocalComponentDoFIndex< 1, 5 >{
               { 1, 2, 1, 1, 1 } } ) == 159,
      "Scalar factors before, between, and after vector factors were not broadcast consistently." ) && success;

   const ThirdPartyVector third_party_vector{
      std::tuple{ A0{ 0, 2, 5 }, A1{ 2, 3, 5 } },
      5,
      5,
      5 };
   const restriction_layout_test::FactorSpace< ThirdPartyVector >
      third_party_space{ third_party_vector, 1 };
   const auto third_party_product = MakeTensorProductRestriction(
      third_party_space,
      scalar_space );
   success = Check(
      GetGlobalDofIndex(
         third_party_product,
         0,
         LocalComponentDoFIndex< 1, 2 >{ { 2, 1 } } ) == 9,
      "Third-party vector tensor-product factor required registration or lost its coordinates." ) && success;

   using HeterogeneousRankComponent =
      ContiguousL2Restriction< std::index_sequence< 1, 3 > >;
   using HeterogeneousRankVector =
      VectorRestriction< A0, HeterogeneousRankComponent >;
   const HeterogeneousRankVector heterogeneous_rank_vector{
      std::tuple{
         A0{ 0, 2, 5 },
         HeterogeneousRankComponent{ 2, 3, 5 } },
      5,
      5,
      5 };
   const restriction_layout_test::FactorSpace< HeterogeneousRankVector >
      heterogeneous_rank_space{ heterogeneous_rank_vector, 1 };
   const auto heterogeneous_rank_product = MakeTensorProductRestriction(
      heterogeneous_rank_space,
      scalar_space );
   static_assert(
      std::remove_cvref_t< decltype( GetComponentRestriction< 0 >(
         heterogeneous_rank_product ) ) >::tensor_dim == 2 );
   static_assert(
      std::remove_cvref_t< decltype( GetComponentRestriction< 1 >(
         heterogeneous_rank_product ) ) >::tensor_dim == 3 );
   success = Check(
      GetGlobalDofIndex(
         heterogeneous_rank_product,
         0,
         LocalComponentDoFIndex< 1, 3 >{ { 0, 2, 1 } } ) == 9,
      "Vector tensor-product factory forced heterogeneous component ranks into a common shape." ) && success;

   return success;
}

bool TestThirdPartyLayout()
{
   const ThirdPartyRestriction restriction{ 3 };
   const auto layout = MakeRestrictionLayout( restriction );
   std::array< Real, 7 > data{};
   auto view = MakeRestrictionElementView( restriction, data.data() );
   view( 1, 1 ) = 9.0;
   return Check(
      layout.Offset( 1, 1 ) ==
         GetGlobalDofIndex(
            restriction,
            1,
            std::array< GlobalIndex, 1 >{ 1 } ) &&
         data[6] == 9.0,
      "Third-party restriction did not use the default semantic layout." );
}

} // namespace

int main()
{
   bool success = true;
   success = TestScalarLayouts() && success;
   success = TestTensorProductRestrictionView() && success;
   success = TestAlgebraicTensorProductCoordinates() && success;
   success = TestAssociativeTensorProductCoordinates() && success;
   success = TestThirdPartyTensorProductFactor() && success;
   success = TestVectorTensorProductFactory() && success;
   success = TestThirdPartyLayout() && success;
   return success ? 0 : 1;
}

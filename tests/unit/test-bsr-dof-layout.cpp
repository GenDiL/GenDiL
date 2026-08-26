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

namespace third_party_restriction_test
{

template<class DofShape>
struct ScalarRestriction
{
   using dof_shape_type = DofShape;
   static constexpr Integer tensor_dim = DofShape::size();
   GlobalIndex num_local_dofs;
   GlobalIndex num_global_dofs;
   GlobalIndex algebraic_dof_extent;
};

template<class DofShape>
GlobalIndex GetNumberOfLocalDofs(const ScalarRestriction<DofShape>& r)
{
   return r.num_local_dofs;
}

template<class DofShape>
GlobalIndex GetNumberOfGlobalDofs(const ScalarRestriction<DofShape>& r)
{
   return r.num_global_dofs;
}

template<class DofShape>
GlobalIndex GetAlgebraicDofExtent(const ScalarRestriction<DofShape>& r)
{
   return r.algebraic_dof_extent;
}

template<class DofShape, class Visitor>
void ForEachRestrictionEntry(
   const ScalarRestriction<DofShape>&,
   GlobalIndex element,
   const std::array<
      GlobalIndex,
      ScalarRestriction<DofShape>::tensor_dim>& local_dof,
   Visitor&& visitor)
{
   std::forward<Visitor>(visitor)(
      element * Product(DofShape{}) + FlattenMultiIndex<DofShape>(local_dof),
      RestrictionUnitWeight{});
}

struct Specification {};

template<class Mesh, class FiniteElement>
auto MakeElementDoFRestriction(
   const Mesh& mesh,
   const FiniteElement&,
   const Specification&)
{
   using Shape = finite_element_dof_shape_t<
      typename FiniteElement::shape_functions>;
   const GlobalIndex count =
      static_cast<GlobalIndex>(mesh.GetNumberOfCells()) * Product(Shape{});
   return ScalarRestriction<Shape>{count, count, count};
}

// Deliberately visible through ADL for an already completed type. The
// contextual specification concept must still exclude the completed object.
template<class Mesh, class FiniteElement, class DofShape>
auto MakeElementDoFRestriction(
   const Mesh&,
   const FiniteElement&,
   const ScalarRestriction<DofShape>& restriction)
{
   return restriction;
}

template<class Child0, class Child1>
struct VectorRestriction
{
   static constexpr Integer num_components = 2;
   Child0 child0;
   Child1 child1;
   GlobalIndex num_local_dofs;
   GlobalIndex num_global_dofs;
   GlobalIndex algebraic_dof_extent;
};

template<size_t Component, class Child0, class Child1>
decltype(auto) GetComponentRestriction(
   const VectorRestriction<Child0, Child1>& restriction)
{
   static_assert(Component < 2);
   if constexpr (Component == 0)
   {
      return (restriction.child0);
   }
   else
   {
      return (restriction.child1);
   }
}

template<class Child0, class Child1>
GlobalIndex GetNumberOfLocalDofs(
   const VectorRestriction<Child0, Child1>& r)
{
   return r.num_local_dofs;
}

template<class Child0, class Child1>
GlobalIndex GetNumberOfGlobalDofs(
   const VectorRestriction<Child0, Child1>& r)
{
   return r.num_global_dofs;
}

template<class Child0, class Child1>
GlobalIndex GetAlgebraicDofExtent(
   const VectorRestriction<Child0, Child1>& r)
{
   return r.algebraic_dof_extent;
}

template<size_t Component, Integer Dim, class Child0, class Child1, class Visitor>
void ForEachRestrictionEntry(
   const VectorRestriction<Child0, Child1>& restriction,
   GlobalIndex element,
   const LocalComponentDoFIndex<Component, Dim>& local_dof,
   Visitor&& visitor)
{
   ForEachRestrictionEntry(
      GetComponentRestriction<Component>(restriction),
      element,
      local_dof.local_dof,
      std::forward<Visitor>(visitor));
}

} // namespace third_party_restriction_test

namespace multi_entry_restriction_test
{

struct Restriction
{
   using dof_shape_type = std::index_sequence< 1 >;
   static constexpr Integer tensor_dim = 1;
};

GlobalIndex GetNumberOfLocalDofs( const Restriction & )
{
   return 1;
}

GlobalIndex GetNumberOfGlobalDofs( const Restriction & )
{
   return 2;
}

GlobalIndex GetAlgebraicDofExtent( const Restriction & )
{
   return 2;
}

template < typename Visitor >
void ForEachRestrictionEntry(
   const Restriction &,
   GlobalIndex,
   const std::array< GlobalIndex, 1 > &,
   Visitor && visitor )
{
   visitor( 0, RestrictionUnitWeight{} );
   visitor( 1, RestrictionUnitWeight{} );
}

} // namespace multi_entry_restriction_test

namespace gendil
{

template <>
inline constexpr size_t static_restriction_entry_count_v<
   multi_entry_restriction_test::Restriction > = 2;

} // namespace gendil

namespace
{

template < typename Object, typename LocalDofIndex >
concept HasGetGlobalDofIndex =
   requires(
      const Object & object,
      const LocalDofIndex & local_dof )
   {
      {
         GetGlobalDofIndex( object, GlobalIndex{}, local_dof )
      } -> std::same_as< GlobalIndex >;
   };

template < typename Space, size_t Component, typename LocalDofIndex >
concept HasComponentGetGlobalDofIndex =
   requires(
      const Space & space,
      const LocalDofIndex & local_dof )
   {
      {
         GetGlobalDofIndex(
            space,
            std::integral_constant< size_t, Component >{},
            GlobalIndex{},
            local_dof )
      } -> std::same_as< GlobalIndex >;
   };

using ScalarFE0 = GLFiniteElement< 1, 1 >;
using ScalarFE1 = GLFiniteElement< 2, 1 >;
using ScalarShape0 = typename ScalarFE0::shape_functions;
using ScalarDofShape0 = finite_element_dof_shape_t< ScalarShape0 >;
using VectorFE = decltype( MakeVectorFiniteElement( ScalarFE0{}, ScalarFE1{} ) );
using VectorShape = typename VectorFE::shape_functions;
using VectorRestrictionType = VectorL2Restriction<
   ContiguousL2Restriction<
      component_dof_shape_t< VectorShape, 0 > >,
   ContiguousL2Restriction<
      component_dof_shape_t< VectorShape, 1 > > >;
using VectorSpace =
   FiniteElementSpace< Cartesian2DMesh, VectorFE, VectorRestrictionType >;
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

using ThirdPartyScalarRestriction =
   third_party_restriction_test::ScalarRestriction<std::index_sequence<2>>;
using CompatibleThirdPartyScalarRestriction =
   third_party_restriction_test::ScalarRestriction< ScalarDofShape0 >;
using ThirdPartyVectorRestriction =
   third_party_restriction_test::VectorRestriction<
      ThirdPartyScalarRestriction,
      ThirdPartyScalarRestriction>;

static_assert(TensorElementDoFRestriction<ThirdPartyScalarRestriction>);
static_assert(
   VectorRestrictionComponent< ThirdPartyVectorRestriction, 0 > );
static_assert(
   VectorRestrictionComponent< ThirdPartyVectorRestriction, 1 > );
static_assert(VectorElementDoFRestriction<ThirdPartyVectorRestriction>);
static_assert(
   CompatibleElementDoFRestrictionFor<
      CompatibleThirdPartyScalarRestriction,
      ScalarFE0 > );
static_assert(
   !CompatibleElementDoFRestrictionFor<
      CompatibleThirdPartyScalarRestriction,
      ScalarFE1 > );
static_assert(
   CompatibleElementDoFRestrictionFor<
      VectorRestrictionType,
      VectorFE > );
static_assert(
   !CompatibleElementDoFRestrictionFor<
      ContiguousL2Restriction< ScalarDofShape0 >,
      VectorFE > );
static_assert(
   RestrictionSpecificationFor<
      ContiguousL2RestrictionSpecification,
      Cartesian1DMesh,
      GLFiniteElement<1>>);
static_assert(!ElementDoFRestriction<ContiguousL2RestrictionSpecification>);
static_assert(
   RestrictionSpecificationFor<
      third_party_restriction_test::Specification,
      Cartesian1DMesh,
      GLFiniteElement<1>>);
static_assert(
   sizeof(LocalComponentDoFIndex<1, 3>) ==
      sizeof(std::array<GlobalIndex, 3>));
static_assert(std::is_trivially_copyable_v<LocalComponentDoFIndex<1, 3>>);
static_assert(
   !RestrictionSpecificationFor<
      ThirdPartyScalarRestriction,
      Cartesian1DMesh,
      GLFiniteElement<1>>);

struct UnknownRestriction {};

using ScalarContiguousRestriction =
   ContiguousL2Restriction< ScalarDofShape0 >;
using ScalarIndirectRestriction =
   IndirectH1Restriction< ScalarDofShape0 >;
using ScalarContiguousSpace = FiniteElementSpace<
   Cartesian2DMesh,
   ScalarFE0,
   ScalarContiguousRestriction >;
static_assert(
   std::same_as<
      finite_element_space_shape_functions_t< ScalarContiguousSpace >,
      ScalarShape0 > );
static_assert(
   std::same_as<
      finite_element_space_shape_functions_t<
         const ScalarContiguousSpace & >,
      ScalarShape0 > );
static_assert(
   std::same_as<
      finite_element_space_shape_functions_t< VectorSpace >,
      VectorShape > );
static_assert(
   std::same_as<
      finite_element_space_shape_functions_t< const VectorSpace >,
      VectorShape > );
static_assert(
   !is_vector_finite_element_space_v< ScalarContiguousSpace > );
static_assert(
   !is_vector_finite_element_space_v< ScalarContiguousSpace & > );
static_assert(
   is_vector_finite_element_space_v< VectorSpace > );
static_assert(
   is_vector_finite_element_space_v< const VectorSpace & > );
static_assert( num_comp_v< ScalarContiguousSpace > == 1 );
static_assert( num_comp_v< const VectorSpace & > == 2 );
using ScalarLocalDofDescriptor = LocalDofDescriptor<
   0,
   false,
   ScalarDofShape0,
   std::index_sequence<>,
   2,
   0 >;
using InvalidScalarComponentDofDescriptor = LocalDofDescriptor<
   1,
   false,
   ScalarDofShape0,
   std::index_sequence<>,
   2,
   0 >;
using VectorLocalDofDescriptor = LocalDofDescriptor<
   1,
   true,
   Component1DofShape,
   std::index_sequence<>,
   2,
   0 >;
using VectorIndirectRestriction = VectorRestriction<
   ScalarIndirectRestriction,
   ScalarIndirectRestriction >;
using AnisotropicMultiIndexShape = std::index_sequence< 2, 3, 4 >;

static_assert(
   ElementDoFRestrictionFor<
      ScalarContiguousRestriction,
      std::array< GlobalIndex, 2 > > );
static_assert(
   !ElementDoFRestrictionFor<
      ScalarContiguousRestriction,
      std::array< GlobalIndex, 1 > > );
static_assert(
   HasGetGlobalDofIndex<
      ScalarContiguousRestriction,
      std::array< GlobalIndex, 2 > > );
static_assert(
   !HasGetGlobalDofIndex<
      ScalarContiguousRestriction,
      std::array< GlobalIndex, 1 > > );
static_assert(
   HasGetGlobalDofIndex<
      ScalarIndirectRestriction,
      std::array< GlobalIndex, 2 > > );
static_assert(
   HasGetGlobalDofIndex<
      ScalarContiguousSpace,
      std::array< GlobalIndex, 2 > > );
static_assert(
   HasGetGlobalDofIndex<
      ScalarContiguousSpace,
      GlobalIndex > );
static_assert(
   HasComponentGetGlobalDofIndex<
      ScalarContiguousSpace,
      0,
      std::array< GlobalIndex, 2 > > );
static_assert(
   HasComponentGetGlobalDofIndex<
      ScalarContiguousSpace,
      0,
      GlobalIndex > );
static_assert(
   !HasComponentGetGlobalDofIndex<
      ScalarContiguousSpace,
      1,
      std::array< GlobalIndex, 2 > > );
static_assert(
   !HasComponentGetGlobalDofIndex<
      ScalarContiguousSpace,
      1,
      GlobalIndex > );
static_assert(
   !HasGetGlobalDofIndex<
      VectorSpace,
      std::array< GlobalIndex, 2 > > );
static_assert(
   !HasGetGlobalDofIndex<
      VectorSpace,
      GlobalIndex > );
static_assert(
   HasComponentGetGlobalDofIndex<
      VectorSpace,
      0,
      std::array< GlobalIndex, 2 > > );
static_assert(
   HasComponentGetGlobalDofIndex<
      VectorSpace,
      1,
      GlobalIndex > );
static_assert(
   HasGetGlobalDofIndex<
      ScalarContiguousSpace,
      ScalarLocalDofDescriptor > );
static_assert(
   !HasGetGlobalDofIndex<
      ScalarContiguousSpace,
      InvalidScalarComponentDofDescriptor > );
static_assert(
   !HasGetGlobalDofIndex<
      ScalarContiguousSpace,
      VectorLocalDofDescriptor > );
static_assert(
   HasGetGlobalDofIndex<
      VectorSpace,
      VectorLocalDofDescriptor > );
static_assert(
   !HasGetGlobalDofIndex<
      VectorSpace,
      ScalarLocalDofDescriptor > );
static_assert(
   HasGetGlobalDofIndex<
      VectorIndirectRestriction,
      LocalComponentDoFIndex< 0, 2 > > );
static_assert(
   !HasGetGlobalDofIndex<
      ThirdPartyScalarRestriction,
      std::array< GlobalIndex, 1 > > );
static_assert(
   ElementDoFRestrictionFor<
      multi_entry_restriction_test::Restriction,
      std::array< GlobalIndex, 1 > > );
static_assert(
   !HasGetGlobalDofIndex<
      multi_entry_restriction_test::Restriction,
      std::array< GlobalIndex, 1 > > );
static_assert(
   std::same_as<
      decltype(UnflattenMultiIndex< AnisotropicMultiIndexShape >( 0 )),
      std::array< GlobalIndex, 3 > > );
static_assert(
   FlattenMultiIndex< AnisotropicMultiIndexShape >(
      std::array< GlobalIndex, 3 >{ 1, 2, 3 } ) == 23 );
static_assert(
   UnflattenMultiIndex< AnisotropicMultiIndexShape >( 23 ) ==
      std::array< GlobalIndex, 3 >{ 1, 2, 3 } );
static_assert(
   details::DofIndexIsInBounds< AnisotropicMultiIndexShape >(
      std::array< GlobalIndex, 3 >{ 1, 2, 3 } ) );
static_assert(
   !details::DofIndexIsInBounds< AnisotropicMultiIndexShape >(
      std::array< GlobalIndex, 3 >{ 2, 0, 0 } ) );

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

static_assert( TensorElementDoFRestriction< ScalarContiguousRestriction > );
static_assert( TensorElementDoFRestriction< ScalarIndirectRestriction > );
static_assert( VectorElementDoFRestriction< VectorIndirectRestriction > );
static_assert( !ElementDoFRestriction< UnknownRestriction > );

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
      !restriction_may_share_global_dofs_v< ScalarContiguousRestriction >,
      "ContiguousL2Restriction should be non-sharing." ) && success;
   success = Check(
      restriction_may_share_global_dofs_v< ScalarIndirectRestriction >,
      "IndirectH1Restriction should conservatively allow sharing." ) && success;
   success = Check(
      !restriction_may_share_global_dofs_v< VectorRestrictionType >,
      "Validated vector L2 direct sums should be non-sharing." ) && success;
   success = Check(
      static_restriction_entry_count_v< ScalarContiguousRestriction > == 1 &&
      static_restriction_entry_count_v< ScalarIndirectRestriction > == 1,
      "Current scalar restrictions should have one static row entry." ) && success;
   success = Check(
      restriction_supports_element_reference_view_v< ScalarContiguousRestriction >,
      "ContiguousL2Restriction should support restriction-backed reference views." ) && success;
   success = Check(
      restriction_supports_element_reference_view_v< ScalarIndirectRestriction >,
      "IndirectH1Restriction should support restriction-backed reference views." ) && success;

   return success;
}

bool TestThirdPartyContextualCompletion()
{
   Cartesian1DMesh mesh(1.0, 1);
   auto space = MakeFiniteElementSpace(
      mesh,
      GLFiniteElement<1>{},
      third_party_restriction_test::Specification{});
   using StoredRestriction =
      typename std::remove_cvref_t<decltype(space)>::restriction_type;
   static_assert(
      std::is_same_v<StoredRestriction, ThirdPartyScalarRestriction>);
   const ThirdPartyScalarRestriction completed{2, 2, 2};
   auto completed_space = MakeFiniteElementSpace(
      mesh,
      GLFiniteElement<1>{},
      completed);
   using StoredCompletedRestriction =
      typename std::remove_cvref_t<decltype(completed_space)>::restriction_type;
   static_assert(
      std::is_same_v<StoredCompletedRestriction, ThirdPartyScalarRestriction>);
   return Check(
      GetNumberOfLocalDofs(space) == 2 &&
         GetNumberOfGlobalDofs(space) == 2 &&
         GetAlgebraicDofExtent(space) == 2 &&
         GetNumberOfLocalDofs(completed_space) == 2 &&
         GetNumberOfGlobalDofs(completed_space) == 2 &&
         GetAlgebraicDofExtent(completed_space) == 2,
      "Third-party specification/completed overload selection did not preserve the structural restriction.");
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

bool TestAnisotropicMultiIndexRoundTrip()
{
   bool success = true;
   constexpr GlobalIndex num_indices =
      Product( AnisotropicMultiIndexShape{} );

   for ( GlobalIndex ordinal = 0; ordinal < num_indices; ++ordinal )
   {
      const auto index =
         UnflattenMultiIndex< AnisotropicMultiIndexShape >( ordinal );
      success = Check(
         FlattenMultiIndex< AnisotropicMultiIndexShape >( index ) == ordinal,
         "Anisotropic std::array multi-index did not round trip." ) && success;
   }

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
      MakeFiniteElementSpace( mesh, ScalarFE0{}, ContiguousL2RestrictionSpecification{ 7 } );
   const GlobalIndex minimum_shifted_extent =
      GetAlgebraicDofExtent( shifted_space );
   auto explicit_extent_space = MakeFiniteElementSpace(
      mesh,
      ScalarFE0{},
      ContiguousL2RestrictionSpecification{
         7,
         minimum_shifted_extent } );
   auto larger_extent_space = MakeFiniteElementSpace(
      mesh,
      ScalarFE0{},
      ContiguousL2RestrictionSpecification{
         7,
         minimum_shifted_extent + 3 } );

   bool success = true;
   const GlobalIndex element_index = 1;
   const std::array< GlobalIndex, 2 > i11{ 1, 1 };
   const GlobalIndex element_dofs = Product( ScalarDofShape0{} );
   const GlobalIndex local_id = FlattenLocalDof( scalar_space, i11 );
   const GlobalIndex zero_based_index =
      element_index * element_dofs + local_id;
   const ScalarLocalDofDescriptor scalar_descriptor{
      i11,
      std::array< GlobalIndex, 0 >{},
      i11 };

   success = Check(
      LocalDofCount( scalar_space ) == element_dofs,
      "Unexpected scalar local DoF count." ) && success;
   success = Check(
      GetNumberOfGlobalDofs( scalar_space ) ==
         scalar_space.GetNumberOfFiniteElements() * element_dofs,
      "Unexpected scalar L2 global DoF count." ) && success;
   success = Check(
      GetAlgebraicDofExtent( scalar_space ) ==
         scalar_space.GetNumberOfFiniteElements() * element_dofs,
      "Unexpected scalar L2 final finite-element DoF count." ) && success;
   success = Check(
      GetNumberOfLocalDofs( scalar_space ) ==
         GetNumberOfGlobalDofs( scalar_space ) &&
      GetNumberOfGlobalDofs( scalar_space ) ==
         GetAlgebraicDofExtent( scalar_space ),
      "Zero-shift scalar L2 should have equal local, logical-global, and algebraic extents." ) && success;
   success = Check(
      GetAlgebraicDofExtent( explicit_extent_space ) ==
         minimum_shifted_extent,
      "Explicit exact contiguous L2 algebraic extent was not retained." ) &&
      success;
   success = Check(
      GetAlgebraicDofExtent( larger_extent_space ) ==
         minimum_shifted_extent + 3,
      "Explicit larger contiguous L2 algebraic extent was not retained." ) &&
      success;

   success = Check(
      GetGlobalDofIndex(
         scalar_space,
         element_index,
         i11 ) == zero_based_index,
      "Unexpected scalar L2 global DoF index." ) && success;
   success = Check(
      GetGlobalDofIndex(
         scalar_space,
         element_index,
         scalar_descriptor ) ==
         GetGlobalDofIndex(
            scalar_space,
            element_index,
            i11 ),
      "Scalar descriptor lookup disagrees with scalar tensor-index lookup." ) &&
      success;
   success = Check(
      GetGlobalDofIndex(
         scalar_space,
         Component0Tag{},
         element_index,
         i11 ) ==
         GetGlobalDofIndex(
            scalar_space,
            element_index,
            i11 ),
      "Component-zero scalar lookup disagrees with scalar tensor-index lookup." ) &&
      success;
   success = Check(
      GetGlobalDofIndex(
         scalar_space,
         Component0Tag{},
         element_index,
         local_id ) ==
         GetGlobalDofIndex(
            scalar_space,
            element_index,
            local_id ),
      "Component-zero scalar lookup disagrees with scalar ordinal lookup." ) &&
      success;
   success = Check(
      GetGlobalDofIndex(
         scalar_space,
         element_index,
         local_id ) ==
         scalar_space.restriction.shift + zero_based_index,
      "Scalar L2 coordinate does not match its direct occurrence ordinal." ) &&
      success;
   success = Check(
      GetGlobalDofIndex(
         scalar_space,
         element_index,
         local_id ) ==
         GetGlobalDofIndex(
            scalar_space,
            element_index,
            i11 ),
      "Scalar L2 scalar-flat GetGlobalDofIndex disagrees with tensor-index mapping." ) && success;
   success = Check(
      GetGlobalDofIndex(
         scalar_space,
         zero_based_index / element_dofs,
         zero_based_index % element_dofs ) == zero_based_index,
      "Scalar L2 ordinal mapping is wrong." ) && success;

   success = Check(
      GetNumberOfGlobalDofs( shifted_space ) ==
         scalar_space.GetNumberOfFiniteElements() * element_dofs,
      "Shifted scalar L2 logical global count should not include the FE-space shift." ) && success;
   success = Check(
      GetNumberOfLocalDofs( shifted_space ) ==
         GetNumberOfGlobalDofs( shifted_space ) &&
      GetNumberOfGlobalDofs( shifted_space ) <
         GetAlgebraicDofExtent( shifted_space ) &&
      GetAlgebraicDofExtent( shifted_space ) ==
         shifted_space.restriction.shift +
            GetNumberOfGlobalDofs( shifted_space ),
      "Shifted scalar L2 should separate logical global DoFs from algebraic extent." ) && success;
   success = Check(
      GetGlobalDofIndex(
         shifted_space,
         element_index,
         local_id ) == shifted_space.restriction.shift + zero_based_index,
      "Shifted scalar L2 coordinate should apply its placement exactly once." ) &&
      success;
   success = Check(
      GetGlobalDofIndex(
         shifted_space,
         element_index,
         local_id ) == shifted_space.restriction.shift + zero_based_index,
      "Shifted scalar L2 final global index should include the FE-space shift." ) && success;
   success = Check(
      GetGlobalDofIndex(
         shifted_space,
         zero_based_index / element_dofs,
         zero_based_index % element_dofs ) ==
         shifted_space.restriction.shift + zero_based_index,
      "Shifted scalar L2 ordinal mapping should include the FE-space shift." ) && success;

   std::vector< Real > data( GetAlgebraicDofExtent( shifted_space ) );
   for ( GlobalIndex i = 0; i < data.size(); ++i )
   {
      data[i] = static_cast< Real >( i );
   }

   auto element_view =
      MakeScalarElementTensorView( shifted_space, data.data() );
   const GlobalIndex shifted_global_index =
      GetGlobalDofIndex( shifted_space, element_index, local_id );
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
   IndirectH1RestrictionSpecification restriction{ restriction_indices, 6 };
   auto h1_space = MakeFiniteElementSpace( mesh, ScalarFE0{}, restriction );

   const std::array< GlobalIndex, 2 > i00{ 0, 0 };
   const std::array< GlobalIndex, 2 > i10{ 1, 0 };
   const std::array< GlobalIndex, 2 > i01{ 0, 1 };
   const std::array< GlobalIndex, 2 > i11{ 1, 1 };

   bool success = true;

   success = Check(
      LocalDofCount( h1_space ) == 4,
      "Unexpected scalar H1 local DoF count." ) && success;
   success = Check(
      GetNumberOfGlobalDofs( h1_space ) == 6,
      "Unexpected scalar H1 global DoF count." ) && success;
   success = Check(
      GetAlgebraicDofExtent( h1_space ) == 6,
      "Unexpected scalar H1 final finite-element DoF count." ) && success;
   success = Check(
      GetNumberOfGlobalDofs( h1_space ) == 6 &&
         GetAlgebraicDofExtent( h1_space ) == 6,
      "Homogeneous indirect H1 logical count and algebraic extent should agree." ) && success;

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
      GetGlobalDofIndex( h1_space, 0, i00 ) == 0,
      "H1 element 0 local id 0 should map to global DoF 0." ) && success;
   success = Check(
      GetGlobalDofIndex( h1_space, 0, i10 ) == 1,
      "H1 element 0 local id 1 should map to global DoF 1." ) && success;
   success = Check(
      GetGlobalDofIndex( h1_space, 0, i01 ) == 3,
      "H1 element 0 local id 2 should map to global DoF 3." ) && success;
   success = Check(
      GetGlobalDofIndex( h1_space, 0, i11 ) == 4,
      "H1 element 0 local id 3 should map to global DoF 4." ) && success;

   success = Check(
      GetGlobalDofIndex( h1_space, 1, i00 ) == 1,
      "H1 element 1 local id 0 should map to global DoF 1." ) && success;
   success = Check(
      GetGlobalDofIndex( h1_space, 1, i10 ) == 2,
      "H1 element 1 local id 1 should map to global DoF 2." ) && success;
   success = Check(
      GetGlobalDofIndex( h1_space, 1, i01 ) == 4,
      "H1 element 1 local id 2 should map to global DoF 4." ) && success;
   success = Check(
      GetGlobalDofIndex( h1_space, 1, i11 ) == 5,
      "H1 element 1 local id 3 should map to global DoF 5." ) && success;

   const GlobalIndex local00 = FlattenLocalDof( h1_space, i00 );
   const GlobalIndex local10 = FlattenLocalDof( h1_space, i10 );
   const GlobalIndex local01 = FlattenLocalDof( h1_space, i01 );
   const GlobalIndex local11 = FlattenLocalDof( h1_space, i11 );

   success = Check(
      GetGlobalDofIndex( h1_space, 0, local00 ) ==
         GetGlobalDofIndex( h1_space, 0, i00 ),
      "Scalar H1 scalar-flat mapping disagrees for element 0 local id 0." ) && success;
   success = Check(
      GetGlobalDofIndex( h1_space, 0, local10 ) ==
         GetGlobalDofIndex( h1_space, 0, i10 ),
      "Scalar H1 scalar-flat mapping disagrees for element 0 local id 1." ) && success;
   success = Check(
      GetGlobalDofIndex( h1_space, 1, local01 ) ==
         GetGlobalDofIndex( h1_space, 1, i01 ),
      "Scalar H1 scalar-flat mapping disagrees for element 1 local id 2." ) && success;
   success = Check(
      GetGlobalDofIndex( h1_space, 1, local11 ) ==
         GetGlobalDofIndex( h1_space, 1, i11 ),
      "Scalar H1 scalar-flat mapping disagrees for element 1 local id 3." ) && success;
   const GlobalIndex element0_local01_ordinal =
      0 * LocalDofCount( h1_space ) +
      static_cast< GlobalIndex >( local01 );
   const GlobalIndex element1_local00_ordinal =
      1 * LocalDofCount( h1_space ) +
      static_cast< GlobalIndex >( local00 );

   success = Check(
      GetGlobalDofIndex(
         h1_space,
         element0_local01_ordinal / LocalDofCount( h1_space ),
         element0_local01_ordinal % LocalDofCount( h1_space ) ) ==
         GetGlobalDofIndex( h1_space, 0, local01 ),
      "Scalar H1 ordinal mapping disagrees for element 0 local id 2." ) && success;
   success = Check(
      GetGlobalDofIndex(
         h1_space,
         element1_local00_ordinal / LocalDofCount( h1_space ),
         element1_local00_ordinal % LocalDofCount( h1_space ) ) ==
         GetGlobalDofIndex( h1_space, 1, local00 ),
      "Scalar H1 ordinal mapping disagrees for element 1 local id 0." ) && success;

   std::vector< Real > data( GetAlgebraicDofExtent( h1_space ) );
   for ( GlobalIndex i = 0; i < data.size(); ++i )
   {
      data[i] = 10.0 + static_cast< Real >( i );
   }

   auto element_view =
      MakeScalarElementTensorView( h1_space, data.data() );
   success = Check(
      element_view( i01[0], i01[1], 0 ) ==
         data[GetGlobalDofIndex( h1_space, 0, local01 )],
      "Scalar H1 ElementTensorView read disagrees with canonical mapping." ) && success;
   success = Check(
      element_view( i00[0], i00[1], 1 ) ==
         data[GetGlobalDofIndex( h1_space, 1, local00 )],
      "Scalar H1 ElementTensorView read disagrees with shared-node mapping." ) && success;

   element_view( i10[0], i10[1], 0 ) = 42.0;
   success = Check(
      data[GetGlobalDofIndex( h1_space, 0, local10 )] == 42.0,
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

   IndirectH1RestrictionSpecification restriction0{ restriction_indices0, 6 };
   IndirectH1RestrictionSpecification restriction1{ restriction_indices1, 6 };
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

   static_assert( TensorElementDoFRestriction< ProductRestriction > );
   static_assert(
      static_restriction_entry_count_v< ProductRestriction > == 1 );
   static_assert(
      HasGetGlobalDofIndex<
         ProductRestriction,
         std::array< GlobalIndex, 4 > > );
   static_assert(
      HasGetGlobalDofIndex<
         std::remove_cvref_t< decltype( product_space ) >,
         std::array< GlobalIndex, 4 > > );
   static_assert(
      restriction_may_share_global_dofs_v< ProductRestriction > );

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
      GetGlobalDofIndex( factor_space0, e0, local0 );
   const GlobalIndex g1 =
      GetGlobalDofIndex( factor_space1, e1, local1 );
   const GlobalIndex n0 = GetNumberOfGlobalDofs( factor_space0 );
   const GlobalIndex expected_global = g0 + n0 * g1;

   bool success = true;
   success = Check(
      GetAlgebraicDofExtent( product_space ) ==
         GetNumberOfGlobalDofs( factor_space0 ) *
         GetNumberOfGlobalDofs( factor_space1 ),
      "Tensor-product H1 DoF count should be the product of factor topology counts." ) && success;
   success = Check(
      GetGlobalDofIndex(
         product_space,
         product_element,
         FlattenLocalDof( product_space, product_indices ) ) ==
         expected_global,
      "Tensor-product H1 scalar-flat mapping did not match factor recombination." ) && success;
   success = Check(
      GetGlobalDofIndex(
         product_space.restriction,
         product_element,
         product_indices ) == expected_global,
      "Tensor-product std::array visitation did not match factor recombination." ) && success;

   std::vector< Real > data( GetAlgebraicDofExtent( product_space ) );
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
      MakeFiniteElementSpace( mesh0, Factor0FE{}, ContiguousL2RestrictionSpecification{} );
   auto factor_space1 =
      MakeFiniteElementSpace( mesh1, Factor1FE{}, ContiguousL2RestrictionSpecification{} );
   auto factor_space2 =
      MakeFiniteElementSpace( mesh2, Factor2FE{}, ContiguousL2RestrictionSpecification{} );
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

   const GlobalIndex n0 = GetNumberOfGlobalDofs( factor_space0 );
   const GlobalIndex n1 = GetNumberOfGlobalDofs( factor_space1 );
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
      GetAlgebraicDofExtent( product_space ) == n0 * n1 *
         GetNumberOfGlobalDofs( factor_space2 ),
      "Three-factor tensor-product DoF count is wrong." ) && success;
   success = Check(
      expected_global + 1 ==
         static_cast< GlobalIndex >(
            GetAlgebraicDofExtent( product_space ) ),
      "Three-factor test should exercise the highest product topology index." ) && success;
   success = Check(
      GetGlobalDofIndex(
         product_space,
         product_element,
         FlattenLocalDof( product_space, product_indices ) ) ==
         expected_global,
      "Three-factor tensor-product scalar-flat mapping used the wrong factor stride." ) && success;

   std::vector< Real > data( GetAlgebraicDofExtent( product_space ) );
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
      "Three-factor tensor-product element view used the wrong algebraic factor coordinate." ) && success;

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

bool TestVectorTensorProductFiniteElementSpace()
{
   Cartesian1DMesh mesh0( 1.0, 1 );
   Cartesian1DMesh mesh1( 1.0, 1 );
   const auto factor0_finite_element = MakeVectorFiniteElement(
      GLFiniteElement< 1 >{},
      GLFiniteElement< 2 >{} );
   const auto factor1_finite_element = MakeVectorFiniteElement(
      GLFiniteElement< 0 >{},
      GLFiniteElement< 1 >{} );
   const auto factor0 = MakeFiniteElementSpace(
      mesh0,
      factor0_finite_element );
   const auto factor1 = MakeFiniteElementSpace(
      mesh1,
      factor1_finite_element );
   const auto product_mesh = MakeCartesianProductMesh( mesh0, mesh1 );
   const auto product_restriction =
      MakeTensorProductRestriction( factor0, factor1 );
   const auto product_finite_element = MakeVectorFiniteElement(
      GLFiniteElement< 1, 0 >{},
      GLFiniteElement< 2, 1 >{} );
   const auto product_space = MakeFiniteElementSpace(
      product_mesh,
      product_finite_element,
      product_restriction );

   using ProductRestriction =
      std::remove_cvref_t< decltype( product_restriction ) >;
   static_assert( VectorElementDoFRestriction< ProductRestriction > );
   static_assert( ProductRestriction::num_components == 2 );

   bool success = true;
   success = Check(
      GetNumberOfLocalDofs( product_space ) == 8 &&
         GetNumberOfGlobalDofs( product_space ) == 8 &&
         GetAlgebraicDofExtent( product_space ) == 15,
      "Vector tensor-product finite-element dimensions are incorrect." ) && success;
   for ( GlobalIndex j = 0; j < 2; ++j )
   {
      for ( GlobalIndex i = 0; i < 3; ++i )
      {
         const std::array< GlobalIndex, 2 > local{ i, j };
         const GlobalIndex expected = ( 2 + i ) + 5 * ( 1 + j );
         success = Check(
            GetGlobalDofIndex(
               product_space,
               std::integral_constant< size_t, 1 >{},
               0,
               local ) == expected,
            "Vector tensor-product finite-element coordinate was rebased or compacted." ) && success;
      }
   }
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
   IndirectH1RestrictionSpecification scalar_restriction{ restriction_indices, 3 };
   auto restriction = MakeVectorIndirectH1RestrictionSpecification< 2 >( scalar_restriction );
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
      VectorElementDoFRestriction<
         typename VectorH1Space::restriction_type > &&
      restriction_may_share_global_dofs_v<
         typename VectorH1Space::restriction_type >,
      "This test requires a sharing vector restriction." );
   static_assert(
      VectorH1Space::restriction_type::num_components ==
         VectorH1Shape::vector_dim,
      "VectorIndirectH1RestrictionSpecification<NComp> should match the vector FE component count." );

   bool success = true;

   success = Check(
      GetAlgebraicDofExtent( vector_h1_space ) == 6,
      "Vector H1 total true-DoF count should be component-major scalar_count * NComp." ) && success;
   success = Check(
      GetNumberOfGlobalDofs( vector_h1_space ) == 6 &&
         GetAlgebraicDofExtent( vector_h1_space ) == 6 &&
         GetNumberOfGlobalDofs(
            GetComponentRestriction<0>(vector_h1_space.restriction)) == 3 &&
         GetAlgebraicDofExtent(
            GetComponentRestriction<0>(vector_h1_space.restriction)) == 6 &&
         GetNumberOfGlobalDofs(
            GetComponentRestriction<1>(vector_h1_space.restriction)) == 3 &&
         GetAlgebraicDofExtent(
            GetComponentRestriction<1>(vector_h1_space.restriction)) == 6,
      "Vector H1 children should have component logical counts and a common vector algebraic extent." ) && success;
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
      GetGlobalDofIndex( vector_h1_space, c0, 0, i0 ) == 0,
      "Vector H1 component 0 element 0 left node should map to true DoF 0." ) && success;
   success = Check(
      GetGlobalDofIndex( vector_h1_space, c0, 0, i1 ) == 1,
      "Vector H1 component 0 element 0 right node should map to true DoF 1." ) && success;
   success = Check(
      GetGlobalDofIndex( vector_h1_space, c0, 1, i0 ) == 1,
      "Vector H1 component 0 shared node should map to true DoF 1 from element 1." ) && success;
   success = Check(
      GetGlobalDofIndex( vector_h1_space, c0, 1, i1 ) == 2,
      "Vector H1 component 0 element 1 right node should map to true DoF 2." ) && success;

   success = Check(
      GetGlobalDofIndex( vector_h1_space, c1, 0, i0 ) == 3,
      "Vector H1 component 1 true DoFs should begin after all component 0 DoFs." ) && success;
   success = Check(
      GetGlobalDofIndex( vector_h1_space, c1, 0, i1 ) == 4,
      "Vector H1 component 1 element 0 right node should map to true DoF 4." ) && success;
   success = Check(
      GetGlobalDofIndex( vector_h1_space, c1, 1, i0 ) == 4,
      "Vector H1 component 1 shared node should map to true DoF 4 from element 1." ) && success;
   success = Check(
      GetGlobalDofIndex( vector_h1_space, c1, 1, i1 ) == 5,
      "Vector H1 component 1 element 1 right node should map to true DoF 5." ) && success;

   success = Check(
      GetGlobalDofIndex( vector_h1_space, c0, 0, i1 ) ==
         GetGlobalDofIndex( vector_h1_space, c0, 1, i0 ),
      "Vector H1 shared node should be shared within component 0." ) && success;
   success = Check(
      GetGlobalDofIndex( vector_h1_space, c1, 0, i1 ) ==
         GetGlobalDofIndex( vector_h1_space, c1, 1, i0 ),
      "Vector H1 shared node should be shared within component 1." ) && success;
   success = Check(
      GetGlobalDofIndex( vector_h1_space, c0, 0, i0 ) !=
         GetGlobalDofIndex( vector_h1_space, c1, 0, i0 ),
      "Vector H1 components must not alias each other." ) && success;

   success = Check(
      GetGlobalDofIndex(
         vector_h1_space,
         c0,
         0,
         FlattenComponentLocalDof< VectorH1Shape >( c0, i1 ) ) ==
         GetGlobalDofIndex( vector_h1_space, c0, 0, i1 ),
      "Vector H1 component-flat wrapper changed component 0 behavior." ) && success;
   success = Check(
      GetGlobalDofIndex(
         vector_h1_space,
         c1,
         1,
         FlattenComponentLocalDof< VectorH1Shape >( c1, i0 ) ) ==
         GetGlobalDofIndex( vector_h1_space, c1, 1, i0 ),
      "Vector H1 component-flat wrapper changed component 1 behavior." ) && success;

   return success;
}

bool TestVectorGlobalDofIndex()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto vector_space = MakeFiniteElementSpace( mesh, VectorFE{} );

   constexpr Component0Tag c0{};
   constexpr Component1Tag c1{};
   bool success = true;
   const GlobalIndex num_elements = vector_space.GetNumberOfFiniteElements();
   const GlobalIndex element_index = 1;
   const std::array< GlobalIndex, 2 > c0_i11{ 1, 1 };
   const std::array< GlobalIndex, 2 > c1_i21{ 2, 1 };
   const VectorLocalDofDescriptor vector_descriptor{
      c1_i21,
      std::array< GlobalIndex, 0 >{},
      c1_i21 };

   const GlobalIndex c0_global_offset =
      CheckedVectorComponentOffset< VectorShape, 0 >( num_elements );
   const GlobalIndex c1_global_offset =
      CheckedVectorComponentOffset< VectorShape, 1 >( num_elements );
   const GlobalIndex c0_ndofs = Product( Component0DofShape{} );
   const GlobalIndex c1_ndofs = Product( Component1DofShape{} );

   success = Check(
      GetNumberOfGlobalDofs(vector_space) ==
         GetAlgebraicDofExtent(vector_space) &&
      GetNumberOfGlobalDofs(
         GetComponentRestriction<0>(vector_space.restriction)) ==
         num_elements * c0_ndofs &&
      GetNumberOfGlobalDofs(
         GetComponentRestriction<1>(vector_space.restriction)) ==
         num_elements * c1_ndofs &&
      GetAlgebraicDofExtent(
         GetComponentRestriction<0>(vector_space.restriction)) ==
         GetAlgebraicDofExtent(vector_space) &&
      GetAlgebraicDofExtent(
         GetComponentRestriction<1>(vector_space.restriction)) ==
         GetAlgebraicDofExtent(vector_space),
      "Vector L2 dimensions do not distinguish component logical counts from the final algebraic extent." ) && success;

   success = Check(
      GetGlobalDofIndex(
         vector_space,
         c0,
         element_index,
         c0_i11 ) ==
         c0_global_offset + element_index * c0_ndofs + FlattenMultiIndex< Component0DofShape >( c0_i11 ),
      "Unexpected vector component 0 L2 global DoF index." ) && success;

   success = Check(
      GetGlobalDofIndex(
         vector_space,
         c1,
         element_index,
         c1_i21 ) ==
         c1_global_offset + element_index * c1_ndofs + FlattenMultiIndex< Component1DofShape >( c1_i21 ),
      "Unexpected vector component 1 L2 global DoF index." ) && success;
   success = Check(
      GetGlobalDofIndex(
         vector_space,
         element_index,
         vector_descriptor ) ==
         GetGlobalDofIndex(
            vector_space,
            c1,
            element_index,
            c1_i21 ),
      "Vector descriptor lookup disagrees with component tensor-index lookup." ) &&
      success;

   const LocalComponentDoFIndex< 1, 2 > component_local_dof{ c1_i21 };
   success = Check(
      GetGlobalDofIndex(
         vector_space.restriction,
         element_index,
         component_local_dof ) ==
         GetGlobalDofIndex(
            GetComponentRestriction< 1 >( vector_space.restriction ),
            element_index,
            c1_i21 ),
      "Vector wrapper std::array visitation disagrees with selected-child visitation." ) && success;

   success = Check(
      GetGlobalDofIndex(
         vector_space,
         c0,
         element_index,
         FlattenComponentLocalDof< VectorShape >( c0, c0_i11 ) ) ==
         GetGlobalDofIndex(
            vector_space,
            c0,
            element_index,
            c0_i11 ),
      "Vector L2 component-flat wrapper changed component 0 behavior." ) && success;
   success = Check(
      GetGlobalDofIndex(
         vector_space,
         c1,
         element_index,
         FlattenComponentLocalDof< VectorShape >( c1, c1_i21 ) ) ==
         GetGlobalDofIndex(
            vector_space,
            c1,
            element_index,
            c1_i21 ),
      "Vector L2 component-flat wrapper changed component 1 behavior." ) && success;

   std::vector< Real > data( GetAlgebraicDofExtent( vector_space ) );
   for ( GlobalIndex i = 0; i < data.size(); ++i )
   {
      data[i] = static_cast< Real >( i );
   }

   auto evector = MakeVectorElementTensorView( vector_space, data.data() );
   const GlobalIndex c0_global_index =
      GetGlobalDofIndex( vector_space, c0, element_index, c0_i11 );
   const GlobalIndex c1_global_index =
      GetGlobalDofIndex( vector_space, c1, element_index, c1_i21 );

   success = Check(
      std::get< 0 >( evector )( c0_i11[0], c0_i11[1], element_index ) == data[c0_global_index],
      "Vector GetGlobalDofIndex disagrees with component 0 ElementTensorView layout." ) && success;
   success = Check(
      std::get< 1 >( evector )( c1_i21[0], c1_i21[1], element_index ) == data[c1_global_index],
      "Vector GetGlobalDofIndex disagrees with component 1 ElementTensorView layout." ) && success;

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

template < typename View, Integer Dim, size_t... I >
Real ReadViewAt(
   const View & view,
   const std::array< GlobalIndex, Dim > & indices,
   std::index_sequence< I... > )
{
   return view( indices[I]... );
}

template < typename Layout, Integer Dim, size_t... I >
size_t LayoutOffsetAt(
   const Layout & layout,
   const std::array< GlobalIndex, Dim > & indices,
   std::index_sequence< I... > )
{
   return layout.Offset( indices[I]... );
}

template <
   size_t Component,
   typename ThreadShape,
   typename RegisterShape,
   typename VectorSpaceType,
   typename Orientation >
bool CheckComponentDescriptorOrientation(
   const VectorSpaceType & vector_space,
   const Orientation & orientation )
{
   using ShapeFunctions =
      finite_element_space_shape_functions_t< VectorSpaceType >;
   using ComponentDofShape =
      component_dof_shape_t< ShapeFunctions, Component >;
   constexpr Integer Dim = ComponentDofShape::size();
   static_assert( Dim == 3 );
   static_assert(
      ThreadShape::size() + RegisterShape::size() ==
         ComponentDofShape::size() );

   const auto dof_sizes = to_array( ComponentDofShape{} );
   if ( !OrientedTensorDofShapeIsCompatible( dof_sizes, orientation ) )
   {
      std::cerr << "Supported descriptor test received an incompatible orientation.\n";
      return false;
   }

   constexpr size_t NumDofs = Product( ComponentDofShape{} );
   std::array< Real, NumDofs > native_values{};
   for ( size_t i = 0; i < NumDofs; ++i )
   {
      native_values[i] = static_cast< Real >( 1000 * Component + i + 1 );
   }
   auto global_dofs = MakeFIFOView(
      native_values.data(),
      static_cast< GlobalIndex >( dof_sizes[0] ),
      static_cast< GlobalIndex >( dof_sizes[1] ),
      static_cast< GlobalIndex >( dof_sizes[2] ),
      GlobalIndex{1} );
   const auto oriented_global_dofs = MakeOrientedGlobalDofView(
      global_dofs,
      GlobalIndex{0},
      dof_sizes,
      orientation );
   const auto staged_layout = MakeOrientedLayout(
      dof_sizes,
      orientation );

   bool success = true;
   ForEachLocalDofWithShapes<
      Component,
      true,
      ThreadShape,
      RegisterShape >( [&] ( const auto & reference_dof )
   {
      using ReferenceDescriptor =
         std::remove_cvref_t< decltype(reference_dof) >;
      const auto native_dof = OrientReferenceDofToNative(
         vector_space,
         reference_dof,
         orientation );
      using NativeDescriptor =
         std::remove_cvref_t< decltype(native_dof) >;

      static_assert(
         NativeDescriptor::component_id ==
            ReferenceDescriptor::component_id );
      static_assert(
         NativeDescriptor::is_vector ==
            ReferenceDescriptor::is_vector );
      static_assert(
         NativeDescriptor::thread_dim ==
            ReferenceDescriptor::thread_dim );
      static_assert(
         NativeDescriptor::register_dim ==
            ReferenceDescriptor::register_dim );
      static_assert(std::same_as<
         typename NativeDescriptor::thread_shape,
         typename ReferenceDescriptor::thread_shape >);
      static_assert(std::same_as<
         typename NativeDescriptor::register_shape,
         typename ReferenceDescriptor::register_shape >);

      const auto expected_native_indices = ReferenceToNativeIndex(
         reference_dof.indices,
         dof_sizes,
         orientation );
      success = Check(
         native_dof.indices == expected_native_indices,
         "Sparse descriptor mapping differs from ReferenceToNativeIndex." ) &&
         success;

      for ( Integer i = 0; i < NativeDescriptor::thread_dim; ++i )
      {
         success = Check(
            native_dof.thread_indices[i] == native_dof.indices[i],
            "Oriented descriptor did not preserve its threaded-index split." ) &&
            success;
      }
      for ( Integer i = 0; i < NativeDescriptor::register_dim; ++i )
      {
         success = Check(
            native_dof.register_indices[i] ==
               native_dof.indices[NativeDescriptor::thread_dim + i],
            "Oriented descriptor did not preserve its register-index split." ) &&
            success;
      }

      const Real oriented_value = ReadViewAt(
         oriented_global_dofs,
         reference_dof.indices,
         std::make_index_sequence< Dim >{} );
      const Real native_value = native_values[
         FlattenMultiIndex< ComponentDofShape >( expected_native_indices )];
      success = Check(
         oriented_value == native_value,
         "Sparse descriptor and oriented global DoF view address different values." ) &&
         success;

      const size_t staged_reference_offset = LayoutOffsetAt(
         staged_layout,
         expected_native_indices,
         std::make_index_sequence< Dim >{} );
      success = Check(
         staged_reference_offset ==
            FlattenMultiIndex< ComponentDofShape >( reference_dof.indices ),
         "Direct descriptor mapping differs from the supported staging-layout oracle." ) &&
         success;
   });

   return success;
}

template < typename Orientation >
bool CheckHeterogeneousVectorDescriptorOrientation(
   const Orientation & orientation )
{
   Cartesian3DMesh mesh( 1.0, 1, 1, 1 );
   auto vector_space = MakeFiniteElementSpace(
      mesh,
      AnisotropicVectorFE{} );

   bool success = true;
   success = CheckComponentDescriptorOrientation<
      0,
      std::index_sequence< 2 >,
      std::index_sequence< 2, 2 > >(
         vector_space,
         orientation ) && success;
   success = CheckComponentDescriptorOrientation<
      1,
      std::index_sequence< 4 >,
      std::index_sequence< 3, 3 > >(
         vector_space,
         orientation ) && success;
   return success;
}

bool TestHeterogeneousVectorDescriptorOrientationMapping()
{
   bool success = true;

   // Both components support every reversal and the only nontrivial axis
   // permutation shared by their distinct shapes: swapping axes 1 and 2.
   for ( Integer swap_equal_axes = 0;
         swap_equal_axes < 2;
         ++swap_equal_axes )
   {
      for ( Integer reversal_mask = 0;
            reversal_mask < 8;
            ++reversal_mask )
      {
         Permutation< 3 > orientation = swap_equal_axes == 0
            ? Permutation< 3 >{ { 1, 2, 3 } }
            : Permutation< 3 >{ { 1, 3, 2 } };
         for ( Integer axis = 0; axis < 3; ++axis )
         {
            if ( reversal_mask & ( 1 << axis ) )
            {
               orientation( axis ) = -orientation( axis );
            }
         }
         success = CheckHeterogeneousVectorDescriptorOrientation(
            orientation ) && success;
      }
   }

   success = CheckHeterogeneousVectorDescriptorOrientation(
      IdentityOrientation< 3 >{} ) && success;

   const auto mixed_orientation = MakeTensorProductOrientation(
      IdentityOrientation< 1 >{},
      Permutation< 2 >{ { 2, -1 } } );
   static_assert(is_tensor_product_orientation_v<
      decltype(mixed_orientation) >);
   success = CheckHeterogeneousVectorDescriptorOrientation(
      mixed_orientation ) && success;

   const Permutation< 3 > incompatible_orientation{ { 2, -1, 3 } };
   success = Check(
      !OrientedTensorDofShapeIsCompatible(
         to_array( AnisotropicComponent1DofShape{} ),
         incompatible_orientation ),
      "Unequal-extent axis permutations must remain unsupported." ) && success;

   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestRestrictionTraits() && success;
   success = TestThirdPartyContextualCompletion() && success;
   success = TestScalarFlattenLocalDof() && success;
   success = TestAnisotropicMultiIndexRoundTrip() && success;
   success = TestVectorFlattenLocalDof() && success;
   success = TestScalarGlobalDofIndex() && success;
   success = TestScalarH1GlobalDofIndex() && success;
   success = TestTensorProductH1ElementTensorView() && success;
   success = TestTensorProductThreeFactorElementTensorView() && success;
   success = TestVectorTensorProductFiniteElementSpace() && success;
   success = TestVectorH1GlobalDofIndex() && success;
   success = TestVectorGlobalDofIndex() && success;
   success = TestVectorTrialTraversalAndSetLocalDof() && success;
   success = TestDescriptorThreadRegisterSplit() && success;
   success = TestHeterogeneousVectorDescriptorOrientationMapping() && success;

   return success ? 0 : 1;
}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionweight.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/tensorproductrestriction.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionoperations.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"
#include "gendil/Utilities/View/Layouts/vectorlayout.hpp"

#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

using namespace gendil;

namespace gendil {

struct TestRestrictionEntry
{
   GlobalIndex index{};
   Real weight{};
};

struct SignedRestriction
{
   using dof_shape_type = std::index_sequence< 1 >;
   static constexpr Integer tensor_dim = 1;
};

inline GlobalIndex GetNumberOfLocalDofs( const SignedRestriction & ) { return 1; }
inline GlobalIndex GetNumberOfGlobalDofs( const SignedRestriction & ) { return 1; }
inline GlobalIndex GetAlgebraicDofExtent( const SignedRestriction & ) { return 1; }

template < class Visitor >
GENDIL_HOST_DEVICE
void ForEachRestrictionEntry(
   const SignedRestriction &,
   GlobalIndex,
   const std::array< GlobalIndex, 1 > &,
   Visitor && visitor )
{
   std::forward< Visitor >( visitor )( 0, RestrictionSign{ true } );
}

template <>
inline constexpr size_t static_restriction_entry_count_v< SignedRestriction > = 1;

struct UnitRestriction
{
   using dof_shape_type = std::index_sequence< 1 >;
   static constexpr Integer tensor_dim = 1;
};

inline GlobalIndex GetNumberOfLocalDofs( const UnitRestriction & ) { return 1; }
inline GlobalIndex GetNumberOfGlobalDofs( const UnitRestriction & ) { return 1; }
inline GlobalIndex GetAlgebraicDofExtent( const UnitRestriction & ) { return 1; }

template < class Visitor >
GENDIL_HOST_DEVICE
void ForEachRestrictionEntry(
   const UnitRestriction &,
   GlobalIndex,
   const std::array< GlobalIndex, 1 > &,
   Visitor && visitor )
{
   std::forward< Visitor >( visitor )( 0, RestrictionUnitWeight{} );
}

template <>
inline constexpr size_t static_restriction_entry_count_v< UnitRestriction > = 1;

struct TwoEntryRestriction
{
   using dof_shape_type = std::index_sequence< 1 >;
   static constexpr Integer tensor_dim = 1;
   std::array< TestRestrictionEntry, 2 > entries;
};

inline GlobalIndex GetNumberOfLocalDofs( const TwoEntryRestriction & ) { return 1; }
inline GlobalIndex GetNumberOfGlobalDofs( const TwoEntryRestriction & ) { return 1; }
inline GlobalIndex GetAlgebraicDofExtent( const TwoEntryRestriction & ) { return 2; }

template < class Visitor >
GENDIL_HOST_DEVICE
void ForEachRestrictionEntry(
   const TwoEntryRestriction & restriction,
   GlobalIndex,
   const std::array< GlobalIndex, 1 > &,
   Visitor && visitor )
{
   for ( const auto & entry : restriction.entries )
   {
      std::forward< Visitor >( visitor )( entry.index, entry.weight );
   }
}

template <>
inline constexpr size_t static_restriction_entry_count_v< TwoEntryRestriction > = 2;

struct EmptyRestriction
{
   using dof_shape_type = std::index_sequence< 1 >;
   static constexpr Integer tensor_dim = 1;
};

inline GlobalIndex GetNumberOfLocalDofs( const EmptyRestriction & ) { return 1; }
inline GlobalIndex GetNumberOfGlobalDofs( const EmptyRestriction & ) { return 0; }
inline GlobalIndex GetAlgebraicDofExtent( const EmptyRestriction & ) { return 2; }

template < class Visitor >
GENDIL_HOST_DEVICE
void ForEachRestrictionEntry(
   const EmptyRestriction &,
   GlobalIndex,
   const std::array< GlobalIndex, 1 > &,
   Visitor && )
{ }

template <>
inline constexpr size_t static_restriction_entry_count_v< EmptyRestriction > = 0;

template < class EntryView >
struct RuntimeCardinalityTestRestriction
{
   using dof_shape_type = std::index_sequence< 1 >;
   static constexpr Integer tensor_dim = 1;
   EntryView entries;
};

template < class EntryView >
inline GlobalIndex GetNumberOfLocalDofs(
   const RuntimeCardinalityTestRestriction< EntryView > & ) { return 1; }

template < class EntryView >
inline GlobalIndex GetNumberOfGlobalDofs(
   const RuntimeCardinalityTestRestriction< EntryView > & ) { return 1; }

template < class EntryView >
inline GlobalIndex GetAlgebraicDofExtent(
   const RuntimeCardinalityTestRestriction< EntryView > & ) { return 2; }

template < class EntryView, class Visitor >
GENDIL_HOST_DEVICE
void ForEachRestrictionEntry(
   const RuntimeCardinalityTestRestriction< EntryView > & restriction,
   GlobalIndex,
   const std::array< GlobalIndex, 1 > &,
   Visitor && visitor )
{
   for ( size_t entry_index = 0;
         entry_index < restriction.entries.Size();
         ++entry_index )
   {
      const auto & entry = restriction.entries[entry_index];
      std::forward< Visitor >( visitor )( entry.index, entry.weight );
   }
}

} // namespace gendil

namespace {

struct AssignmentOnlyValue
{
   Real value{};

   AssignmentOnlyValue & operator+=( const AssignmentOnlyValue & ) = delete;
};

bool Near( Real left, Real right )
{
   return std::abs( left - right ) <= 1.0e-14;
}

template < class ViewType >
concept HasAdditiveWrite = requires( ViewType & view )
{
   view( 0, 0 ) += 1.0;
};

template < class ViewType >
concept HasConstAdditiveWrite = requires( const ViewType & view )
{
   view( 0, 0 ) += 1.0;
};

template < class ViewType >
concept HasAssignment = requires( ViewType & view )
{
   view( 0, 0 ) = 1.0;
};

template < class ViewType >
concept HasConstAssignment = requires( const ViewType & view )
{
   view( 0, 0 ) = 1.0;
};

template < class ViewType >
concept HasAtomicWrite = requires( ViewType & view )
{
   AtomicAddInPlace( view( 0, 0 ), 1.0 );
};

template < class ViewType >
concept HasConstAtomicWrite = requires( const ViewType & view )
{
   AtomicAddInPlace( view( 0, 0 ), 1.0 );
};

template < class ViewType >
concept HasBorrowingRvalueAccess = requires( ViewType view )
{
   std::move( view )( 0, 0 );
};

template < class ViewType >
concept HasConstBorrowingRvalueAccess = requires( const ViewType view )
{
   std::move( view )( 0, 0 );
};

bool TestSignedWeight()
{
   const SignedRestriction restriction;
   const std::array< GlobalIndex, 1 > row{ 0 };
   std::array< Real, 1 > global{ 3.0 };

   const Real gathered = GatherRestrictionRow( restriction, 0, row, global );
   const GlobalIndex coordinate = GetGlobalDofIndex(
      restriction, 0, row );
   ScatterAssignRestrictionRowAdjoint( restriction, 0, row, 2.0, global );
   const bool assigned = Near( global[0], -2.0 );
   ScatterAddRestrictionRowAdjoint( restriction, 0, row, 1.5, global );
   const bool added = Near( global[0], -3.5 );
   AtomicScatterAddRestrictionRowAdjoint( restriction, 0, row, -0.5, global );

   return
      coordinate == 0 &&
      Near( gathered, -3.0 ) &&
      assigned &&
      added &&
      Near( global[0], -3.0 );
}

bool TestOneEntryGatherUsesAssignment()
{
   const UnitRestriction restriction;
   const std::array< GlobalIndex, 1 > row{ 0 };
   const std::array< AssignmentOnlyValue, 1 > global{
      AssignmentOnlyValue{ 3.25 } };
   const auto gathered = GatherRestrictionRow(
      restriction,
      0,
      row,
      global );
   return Near( gathered.value, 3.25 );
}

bool TestFixedStencilAndAdjoint()
{
   const TwoEntryRestriction restriction{
      { TestRestrictionEntry{ 0, 2.0 },
        TestRestrictionEntry{ 1, -0.5 } } };
   const std::array< GlobalIndex, 1 > row{ 0 };
   const std::array< Real, 2 > x{ 1.25, -2.0 };
   const Real z = 0.75;

   const Real ex = GatherRestrictionRow( restriction, 0, row, x );
   std::array< Real, 2 > etz{};
   ScatterAddRestrictionRowAdjoint( restriction, 0, row, z, etz );

   const Real left = ex * z;
   const Real right = x[0] * etz[0] + x[1] * etz[1];
   return
      Near( ex, 3.5 ) &&
      Near( etz[0], 1.5 ) &&
      Near( etz[1], -0.375 ) &&
      Near( left, right );
}

bool TestWeightedDofProxy()
{
   const TwoEntryRestriction source_restriction{
      { TestRestrictionEntry{ 0, 2.0 },
        TestRestrictionEntry{ 1, -0.5 } } };
   const TwoEntryRestriction destination_restriction{
      { TestRestrictionEntry{ 1, 0.25 },
        TestRestrictionEntry{ 2, 3.0 } } };
   std::array< Real, 3 > values{ 1.25, -2.0, 9.0 };

   auto source = MakeRestrictionElementView(
      source_restriction, values.data() );
   auto destination = MakeRestrictionElementView(
      destination_restriction, values.data() );

   auto source_proxy = source( 0, 0 );
   using Storage = PointerContainer< Real >;
   using Reference = WeightedDofReference< TwoEntryRestriction, Storage, 1 >;
   using WeightedView = View<
      WeightedRestrictionContainer< TwoEntryRestriction, Storage >,
      WeightedRestrictionLayout< TwoEntryRestriction > >;
   static_assert( std::is_same_v< decltype( source_proxy ), Reference > );
   static_assert( is_weighted_dof_reference< Reference >::value );
   static_assert( is_weighted_dof_reference_v< const Reference & > );
   static_assert( !is_weighted_dof_reference_v< Real > );
   static_assert( std::is_same_v< decltype( source ), WeightedView > );
   static_assert( std::is_same_v<
      decltype( MakeWeightedRestrictionView(
         source_restriction, values.data() ) ),
      WeightedView > );
   static_assert( std::is_same_v<
      decltype( source.layout.Offset( 0, 0 ) ),
      WeightedRestrictionKey< 1 > > );
   const Real gathered = Materialize( source_proxy );
   const Real const_gathered = Materialize(
      std::as_const( source )( 0, 0 ) );
   destination( 0, 0 ) = source_proxy;
   const bool assigned =
      Near( gathered, 3.5 ) &&
      Near( const_gathered, 3.5 ) &&
      Near( values[1], 0.875 ) &&
      Near( values[2], 10.5 );
   destination( 0, 0 ) += 2.0;
   AtomicAddInPlace( destination( 0, 0 ), -1.0 );

   const auto read_only = MakeRestrictionElementView(
      source_restriction,
      static_cast< const Real * >( values.data() ) );
   static_assert( HasAssignment< decltype( destination ) > );
   static_assert( HasAdditiveWrite< decltype( destination ) > );
   static_assert( HasAtomicWrite< decltype( destination ) > );
   static_assert( !HasConstAssignment< decltype( destination ) > );
   static_assert( !HasConstAdditiveWrite< decltype( destination ) > );
   static_assert( !HasConstAtomicWrite< decltype( destination ) > );
   static_assert( !HasAssignment< decltype( read_only ) > );
   static_assert( !HasAdditiveWrite< decltype( read_only ) > );
   static_assert( !HasAtomicWrite< decltype( read_only ) > );
   static_assert( !HasBorrowingRvalueAccess< decltype( source ) > );
   static_assert( !HasConstBorrowingRvalueAccess< decltype( source ) > );
   static_assert( !HasBorrowingRvalueAccess< decltype( read_only ) > );
   static_assert( !HasConstBorrowingRvalueAccess< decltype( read_only ) > );

   return
      assigned &&
      Near( values[1], 1.125 ) &&
      Near( values[2], 13.5 );
}

bool TestWeightedDofProxyBinding()
{
   const TwoEntryRestriction source_restriction{
      { TestRestrictionEntry{ 0, 2.0 },
        TestRestrictionEntry{ 1, -0.5 } } };
   const TwoEntryRestriction destination_restriction{
      { TestRestrictionEntry{ 1, 0.25 },
        TestRestrictionEntry{ 2, 3.0 } } };
   std::array< Real, 3 > source_values{ 1.25, -2.0, 9.0 };
   std::array< Real, 3 > destination_values{ -7.0, -8.0, -9.0 };
   auto source = MakeRestrictionElementView(
      source_restriction, source_values.data() );
   auto destination = MakeRestrictionElementView(
      destination_restriction, destination_values.data() );

   auto source_proxy = source( 0, 0 );
   auto copied_proxy = source_proxy;
   auto moved_proxy = std::move( copied_proxy );
   const bool construction_did_not_scatter =
      Near( source_values[0], 1.25 ) &&
      Near( source_values[1], -2.0 ) &&
      Near( source_values[2], 9.0 );

   source_values[0] = 2.0;
   source_values[1] = -4.0;
   const bool construction_preserved_binding =
      Near( Materialize( source_proxy ), 6.0 ) &&
      Near( Materialize( copied_proxy ), 6.0 ) &&
      Near( Materialize( moved_proxy ), 6.0 );

   auto destination_proxy = destination( 0, 0 );
   destination_proxy = source_proxy;
   const bool copy_assigned =
      Near( destination_values[1], 1.5 ) &&
      Near( destination_values[2], 18.0 );
   destination_proxy = 4.0;
   const bool copy_assignment_did_not_rebind =
      Near( destination_values[1], 1.0 ) &&
      Near( destination_values[2], 12.0 );

   destination_proxy = std::move( moved_proxy );
   const bool move_assigned =
      Near( destination_values[1], 1.5 ) &&
      Near( destination_values[2], 18.0 );
   destination_proxy += 2.0;

   return
      construction_did_not_scatter &&
      construction_preserved_binding &&
      copy_assigned &&
      copy_assignment_did_not_rebind &&
      move_assigned &&
      Near( destination_values[0], -7.0 ) &&
      Near( destination_values[1], 2.0 ) &&
      Near( destination_values[2], 24.0 ) &&
      Near( source_values[0], 2.0 ) &&
      Near( source_values[1], -4.0 ) &&
      Near( source_values[2], 9.0 ) &&
      Near( Materialize( moved_proxy ), 6.0 );
}

bool TestEmptyRowTyping()
{
   const EmptyRestriction restriction;
   const std::array< GlobalIndex, 1 > row{ 0 };
   std::array< Real, 2 > values{ 4.0, -3.0 };
   const Real gathered = GatherRestrictionRow(
      restriction, 0, row, values );
   ScatterAssignRestrictionRowAdjoint(
      restriction, 0, row, 7.0, values );
   ScatterAddRestrictionRowAdjoint(
      restriction, 0, row, 2.0, values );
   return
      Near( gathered, 0.0 ) &&
      Near( values[0], 4.0 ) &&
      Near( values[1], -3.0 );
}

bool TestDynamicDuplicateAndZeroEntries()
{
   std::array< TestRestrictionEntry, 3 > entries{
      TestRestrictionEntry{ 0, 2.0 },
      TestRestrictionEntry{ 0, -0.5 },
      TestRestrictionEntry{ 1, 0.0 } };
   const auto entry_view = MakeVectorView(
      entries.data(),
      static_cast< GlobalIndex >( entries.size() ) );
   const RuntimeCardinalityTestRestriction< decltype( entry_view ) > restriction{
      entry_view };
   static_assert(
      static_restriction_entry_count_v< decltype( restriction ) > ==
         dynamic_restriction_entry_count );

   const std::array< GlobalIndex, 1 > row{ 0 };
   const std::array< Real, 2 > x{ 1.25, -7.0 };
   const Real z = 0.75;
   const Real ex = GatherRestrictionRow( restriction, 0, row, x );
   std::array< Real, 2 > etz{};
   ScatterAddRestrictionRowAdjoint( restriction, 0, row, z, etz );

   return
      Near( ex, 1.875 ) &&
      Near( etz[0], 1.125 ) &&
      Near( etz[1], 0.0 ) &&
      Near( ex * z, x[0] * etz[0] + x[1] * etz[1] );
}

bool TestTensorProductWeightComposition()
{
   const TwoEntryRestriction weighted{
      { TestRestrictionEntry{ 0, 2.0 },
        TestRestrictionEntry{ 1, -0.5 } } };
   const TensorProductRestriction< TwoEntryRestriction, SignedRestriction >
      product{
         std::tuple{ weighted, SignedRestriction{} },
         { 1, 1 },
         { 1, 2 },
         1,
         1,
         2 };
   static_assert(
      static_restriction_entry_count_v<
         std::remove_cvref_t< decltype( product ) > > == 2 );
   static_assert(
      !restriction_supports_element_reference_view_v<
         std::remove_cvref_t< decltype( product ) > > );

   const std::array< Real, 2 > values{ 1.25, -2.0 };
   const std::array< GlobalIndex, 2 > row{ 0, 0 };
   const Real z = 0.75;
   const Real gathered = GatherRestrictionRow( product, 0, row, values );
   std::array< Real, 2 > scattered{};
   ScatterAddRestrictionRowAdjoint(
      product, 0, row, z, scattered );
   return
      Near( gathered, -3.5 ) &&
      Near( scattered[0], -1.5 ) &&
      Near( scattered[1], 0.375 ) &&
      Near(
         gathered * z,
         values[0] * scattered[0] + values[1] * scattered[1] );
}

} // namespace

int main()
{
   static_assert( std::is_trivially_copyable_v< RestrictionSign > );
   static_assert( sizeof( RestrictionSign ) == sizeof( bool ) );
   return
      TestSignedWeight() &&
      TestOneEntryGatherUsesAssignment() &&
      TestFixedStencilAndAdjoint() &&
      TestWeightedDofProxy() &&
      TestWeightedDofProxyBinding() &&
      TestEmptyRowTyping() &&
      TestDynamicDuplicateAndZeroEntries() &&
      TestTensorProductWeightComposition() ? 0 : 1;
}

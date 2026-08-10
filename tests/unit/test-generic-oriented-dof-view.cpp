// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

namespace test_types
{

template < gendil::Integer Rank, typename T = gendil::Real >
struct OpaqueElementView
{
   T * values;
   std::array< gendil::GlobalIndex, Rank + 1 > extents;

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   decltype(auto) operator()( Indices... indices ) const
   {
      static_assert( sizeof...( Indices ) == Rank + 1 );
      const std::array< gendil::GlobalIndex, Rank + 1 > multi_index{
         static_cast< gendil::GlobalIndex >( indices )... };
      gendil::GlobalIndex offset = 0;
      gendil::GlobalIndex stride = 1;
      for ( gendil::Integer axis = 0; axis < Rank + 1; ++axis )
      {
         offset += multi_index[ axis ] * stride;
         stride *= extents[ axis ];
      }
      return values[ offset ];
   }
};

struct RealProxy
{
   gendil::Real * value;

   GENDIL_HOST_DEVICE
   operator gendil::Real() const { return *value; }

   GENDIL_HOST_DEVICE
   RealProxy & operator=( const gendil::Real rhs )
   {
      *value = rhs;
      return *this;
   }
};

template < gendil::Integer Rank >
struct ProxyElementView
{
   gendil::Real * values;
   std::array< gendil::GlobalIndex, Rank + 1 > extents;

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   RealProxy operator()( Indices... indices ) const
   {
      return RealProxy{
         &OpaqueElementView< Rank >{ values, extents }( indices... ) };
   }
};

} // namespace test_types

namespace gendil
{

template < Integer Rank, typename T >
struct get_rank< test_types::OpaqueElementView< Rank, T > >
{
   static constexpr Integer value = Rank + 1;
};

template < Integer Rank >
struct get_rank< test_types::ProxyElementView< Rank > >
{
   static constexpr Integer value = Rank + 1;
};

} // namespace gendil

using namespace gendil;

namespace
{

template < Integer Dim >
struct TestFaceView
{
   using orientation_type = Permutation< Dim >;
   static constexpr Integer dim = Dim;
   static constexpr bool is_conforming = true;

   GlobalIndex cell_index;
   orientation_type orientation;

   GENDIL_HOST_DEVICE
   GlobalIndex GetCellIndex() const { return cell_index; }

   GENDIL_HOST_DEVICE
   const orientation_type & GetOrientation() const { return orientation; }

   GENDIL_HOST_DEVICE
   Point< Dim > GetReferenceNormal() const { return {}; }
};

struct FullSharedSerialKernelConfiguration : HostKernelConfiguration
{
   using face_read_dofs_policy = FullSharedFaceReadDofsPolicy;
   using face_write_dofs_policy = FullSharedFaceWriteDofsPolicy;
};

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cerr << "FAILED: " << message << "\n";
   }
   return condition;
}

template < Integer Rank >
GlobalIndex NumValues(
   const std::array< GlobalIndex, Rank + 1 > & extents )
{
   GlobalIndex count = 1;
   for ( const auto extent : extents )
   {
      count *= extent;
   }
   return count;
}

template < Integer Rank >
bool TestOpaqueOrientedView(
   const std::array< size_t, Rank > & sizes,
   const Permutation< Rank > & orientation )
{
   constexpr GlobalIndex num_elements = 3;
   constexpr GlobalIndex element_index = 2;
   std::array< GlobalIndex, Rank + 1 > extents{};
   for ( Integer axis = 0; axis < Rank; ++axis )
   {
      extents[ axis ] = sizes[ axis ];
   }
   extents[ Rank ] = num_elements;

   std::vector< Real > values( NumValues< Rank >( extents ) );
   for ( GlobalIndex i = 0; i < values.size(); ++i )
   {
      values[ i ] = 10.0 + static_cast< Real >( i );
   }

   test_types::OpaqueElementView< Rank > opaque{ values.data(), extents };
   static_assert(
      !detail::is_explicitly_strided_element_view_v< decltype( opaque ) > );
   auto oriented = MakeOrientedGlobalDofView(
      opaque, element_index, sizes, orientation );
   static_assert(
      std::is_same_v<
         decltype( oriented ),
         detail::GenericOrientedGlobalDofView<
            decltype( opaque ), Rank > > );

   bool success = true;
   std::array< Integer, Rank > reference{};
   while ( true )
   {
      const auto native =
         ReferenceToNativeIndex( reference, sizes, orientation );
      const Real expected = detail::OrientedGlobalDofValueAt(
         opaque, native, element_index );
      success = Check(
         std::abs(
            detail::OrientedGlobalDofValueAt(
               opaque, native, element_index ) - expected ) < 1e-12,
         "opaque native access mismatch" ) && success;

      const Real replacement = -1000.0 - static_cast< Real >(
         FaceReadDofsFIFOOffset( reference, sizes ) );
      [&]< size_t... Is >( std::index_sequence< Is... > )
      {
         success = Check(
            std::abs( oriented( reference[ Is ]... ) - expected ) < 1e-12,
            "generic oriented read mismatch" ) && success;
         oriented( reference[ Is ]... ) = replacement;
      }( std::make_index_sequence< Rank >{} );
      success = Check(
         std::abs(
            detail::OrientedGlobalDofValueAt(
               opaque, native, element_index ) - replacement ) < 1e-12,
         "generic oriented write mismatch" ) && success;

      Integer axis = 0;
      for ( ; axis < Rank; ++axis )
      {
         ++reference[ axis ];
         if ( reference[ axis ] < sizes[ axis ] )
         {
            break;
         }
         reference[ axis ] = 0;
      }
      if ( axis == Rank )
      {
         break;
      }
   }

   return success;
}

bool TestProxyPreservation()
{
   std::array< Real, 8 > values{};
   test_types::ProxyElementView< 2 > proxy_view{
      values.data(), { 2, 2, 2 } };
   const std::array< size_t, 2 > sizes{ 2, 2 };
   const Permutation< 2 > orientation{ { 2, -1 } };
   auto oriented = detail::MakeGenericOrientedGlobalDofView(
      proxy_view, GlobalIndex{ 1 }, sizes, orientation );
   static_assert(
      std::is_same_v< decltype( oriented( 0, 1 ) ), test_types::RealProxy > );

   oriented( 0, 1 ) = 17.0;
   const std::array< Integer, 2 > reference{ 0, 1 };
   const auto native = ReferenceToNativeIndex(
      reference, sizes, orientation );
   return Check(
      std::abs( static_cast< Real >(
         proxy_view( native[ 0 ], native[ 1 ], 1 ) ) - 17.0 ) < 1e-12,
      "generic adapter did not preserve proxy access" );
}

bool TestIndirectedStridedFastPathEquivalence()
{
   std::array< Real, 8 > values{};
   std::array< int, 8 > indirections{ 7, 5, 3, 1, 6, 4, 2, 0 };
   for ( GlobalIndex i = 0; i < values.size(); ++i )
   {
      values[ i ] = 50.0 + static_cast< Real >( i );
   }
   HostDevicePointer< const int > pointer{};
   pointer.host_pointer = indirections.data();
   auto view = MakeIndirectedFIFOView(
      values.data(), pointer, GlobalIndex{ 2 }, GlobalIndex{ 2 },
      GlobalIndex{ 2 } );
   static_assert(
      detail::is_explicitly_strided_element_view_v< decltype( view ) > );

   const std::array< size_t, 2 > sizes{ 2, 2 };
   const Permutation< 2 > orientation{ { 2, -1 } };
   const auto fast = MakeOrientedGlobalDofView(
      view, GlobalIndex{ 1 }, sizes, orientation );
   const auto generic = detail::MakeGenericOrientedGlobalDofView(
      view, GlobalIndex{ 1 }, sizes, orientation );

   bool success = true;
   for ( Integer j = 0; j < 2; ++j )
   {
      for ( Integer i = 0; i < 2; ++i )
      {
         success = Check(
            std::abs( fast( i, j ) - generic( i, j ) ) < 1e-12,
            "indirected strided fast and generic paths disagree" ) &&
            success;
      }
   }
   return success;
}

template < Integer Rank >
bool TestStridedFastPathEquivalence(
   const std::array< size_t, Rank > & sizes,
   const std::vector< Permutation< Rank > > & orientations )
{
   constexpr GlobalIndex num_elements = 2;
   std::array< GlobalIndex, Rank + 1 > extents{};
   GlobalIndex count = num_elements;
   for ( Integer axis = 0; axis < Rank; ++axis )
   {
      extents[ axis ] = sizes[ axis ];
      count *= sizes[ axis ];
   }
   extents[ Rank ] = num_elements;
   std::vector< Real > values( count );
   for ( GlobalIndex i = 0; i < count; ++i )
   {
      values[ i ] = static_cast< Real >( i + 1 );
   }
   const auto layout = MakeFIFOStridedLayout( extents );
   auto view = MakeView( values.data(), layout );
   static_assert(
      detail::is_explicitly_strided_element_view_v< const decltype( view ) & > );

   bool success = true;
   for ( const auto & orientation : orientations )
   {
      const auto fast = MakeOrientedGlobalDofView(
         view, GlobalIndex{ 1 }, sizes, orientation );
      const auto generic = detail::MakeGenericOrientedGlobalDofView(
         view, GlobalIndex{ 1 }, sizes, orientation );
      static_assert(
         std::is_same_v<
            std::remove_cvref_t< decltype( fast ) >,
            OrientedGlobalDofView< decltype( view ), Rank > > );

      std::array< Integer, Rank > reference{};
      while ( true )
      {
         [&]< size_t... Is >( std::index_sequence< Is... > )
         {
            success = Check(
               std::abs(
                  fast( reference[ Is ]... ) -
                  generic( reference[ Is ]... ) ) < 1e-12,
               "signed-stride and generic paths disagree" ) && success;
         }( std::make_index_sequence< Rank >{} );

         Integer axis = 0;
         for ( ; axis < Rank; ++axis )
         {
            ++reference[ axis ];
            if ( reference[ axis ] < sizes[ axis ] )
            {
               break;
            }
            reference[ axis ] = 0;
         }
         if ( axis == Rank )
         {
            break;
         }
      }
   }
   return success;
}

template < bool UseH1 >
auto MakeFactorSpace(
   const Cartesian1DMesh & mesh,
   const std::array< int, 4 > & indices )
{
   using FactorFE = GLFiniteElement< 1 >;
   if constexpr ( UseH1 )
   {
      HostDevicePointer< const int > pointer{};
      pointer.host_pointer = indices.data();
      return MakeFiniteElementSpace(
         mesh, FactorFE{}, H1Restriction{ pointer, 3 } );
   }
   else
   {
      return MakeFiniteElementSpace(
         mesh, FactorFE{}, L2Restriction{} );
   }
}

template < WriteOp Op,
           typename Context,
           typename Space,
           typename Face,
           typename LocalDofs,
           typename GlobalDofs >
void ApplyWrite(
   Context & context,
   const Space & space,
   const Face & face,
   const LocalDofs & local_dofs,
   GlobalDofs & global_dofs )
{
   if constexpr ( Op == WriteAdd )
   {
      WriteAddDofs( context, space, face, local_dofs, global_dofs );
   }
   else if constexpr ( Op == WriteSub )
   {
      WriteSubDofs( context, space, face, local_dofs, global_dofs );
   }
   else
   {
      WriteDofs( context, space, face, local_dofs, global_dofs );
   }
}

template < WriteOp Op, typename Space >
bool TestTensorRestrictionWritePolicy(
   const Space & space,
   const TestFaceView< 2 > & face,
   const std::vector< Real > & baseline )
{
   using DofShape = std::index_sequence< 2, 2 >;
   std::vector< Real > direct_values = baseline;
   std::vector< Real > shared_values = baseline;
   std::vector< Real > expected_values = baseline;
   auto direct_view = MakeScalarElementTensorView(
      space, direct_values.data() );
   auto shared_view = MakeScalarElementTensorView(
      space, shared_values.data() );
   auto expected_view = MakeScalarElementTensorView(
      space, expected_values.data() );
   auto local_dofs = MakeSerialRecursiveArray< Real >( DofShape{} );
   constexpr std::array< size_t, 2 > sizes{ 2, 2 };
   UnitLoop< DofShape >( [&]( auto... indices )
   {
      const std::array< Integer, 2 > index{
         static_cast< Integer >( indices )... };
      local_dofs( indices... ) =
         30.0 + index[ 0 ] + 7.0 * index[ 1 ];
      const auto native = ReferenceToNativeIndex(
         index, sizes, face.orientation );
      auto & expected = expected_view(
         native[ 0 ], native[ 1 ], face.cell_index );
      if constexpr ( Op == WriteAdd )
      {
         expected += local_dofs( indices... );
      }
      else if constexpr ( Op == WriteSub )
      {
         expected -= local_dofs( indices... );
      }
      else
      {
         expected = local_dofs( indices... );
      }
   });

   Real * no_shared_memory = nullptr;
   KernelContext< HostKernelConfiguration, 0 > direct_context(
      no_shared_memory );
   KernelContext< FullSharedSerialKernelConfiguration, 0 > shared_context(
      no_shared_memory );
   ApplyWrite< Op >(
      direct_context, space, face, local_dofs, direct_view );
   ApplyWrite< Op >(
      shared_context, space, face, local_dofs, shared_view );

   return Check(
      direct_values == expected_values,
      "tensor-restriction direct write used the wrong destination indices" ) &&
      Check(
         shared_values == expected_values,
         "tensor-restriction full-shared write used the wrong destination indices" );
}

template < bool FirstH1, bool SecondH1 >
bool TestTwoFactorRestriction()
{
   const Cartesian1DMesh first_mesh( 0.5, 2 );
   const Cartesian1DMesh second_mesh( 0.5, 2 );
   const std::array< int, 4 > first_indices{ 0, 1, 1, 2 };
   const std::array< int, 4 > second_indices{ 0, 1, 1, 2 };
   const auto first_space =
      MakeFactorSpace< FirstH1 >( first_mesh, first_indices );
   const auto second_space =
      MakeFactorSpace< SecondH1 >( second_mesh, second_indices );
   const auto product_mesh =
      MakeCartesianProductMesh( first_mesh, second_mesh );
   const auto restriction =
      MakeTensorProductRestriction( first_space, second_space );
   const auto space = MakeFiniteElementSpace(
      product_mesh, GLFiniteElement< 1, 1 >{}, restriction );

   std::vector< Real > baseline( space.GetNumberOfFiniteElementDofs() );
   for ( GlobalIndex i = 0; i < baseline.size(); ++i )
   {
      baseline[ i ] = 1.0 + static_cast< Real >( i );
   }
   auto view = MakeScalarElementTensorView( space, baseline.data() );
   static_assert(
      !detail::is_explicitly_strided_element_view_v< decltype( view ) > );

   const TestFaceView< 2 > face{
      GlobalIndex{ 3 }, Permutation< 2 >{ { 2, -1 } } };
   Real * no_shared_memory = nullptr;
   KernelContext< HostKernelConfiguration, 0 > direct_context(
      no_shared_memory );
   KernelContext< FullSharedSerialKernelConfiguration, 0 > shared_context(
      no_shared_memory );
   const auto direct = ReadDofs(
      direct_context, space, face, view );
   const auto shared = ReadDofs(
      shared_context, space, face, view );

   bool success = true;
   UnitLoop< std::index_sequence< 2, 2 > >( [&]( auto... indices )
   {
      const std::array< Integer, 2 > reference{
         static_cast< Integer >( indices )... };
      const auto native = ReferenceToNativeIndex(
         reference,
         std::array< size_t, 2 >{ 2, 2 },
         face.orientation );
      const Real expected = view(
         native[ 0 ], native[ 1 ], face.cell_index );
      success = Check(
         std::abs( direct( indices... ) - expected ) < 1e-12,
         "tensor-restriction direct read used the wrong source index" ) &&
         success;
      success = Check(
         std::abs( shared( indices... ) - expected ) < 1e-12,
         "tensor-restriction full-shared read used the wrong source index" ) &&
         success;
   });
   success = TestTensorRestrictionWritePolicy< Write >(
      space, face, baseline ) && success;
   success = TestTensorRestrictionWritePolicy< WriteAdd >(
      space, face, baseline ) && success;
   success = TestTensorRestrictionWritePolicy< WriteSub >(
      space, face, baseline ) && success;
   return success;
}

bool TestThreeFactorRestriction()
{
   const Cartesian1DMesh mesh0( 0.5, 2 );
   const Cartesian1DMesh mesh1( 0.5, 2 );
   const Cartesian1DMesh mesh2( 0.5, 2 );
   const std::array< int, 4 > indices{ 0, 1, 1, 2 };
   const auto space0 = MakeFactorSpace< false >( mesh0, indices );
   const auto space1 = MakeFactorSpace< true >( mesh1, indices );
   const auto space2 = MakeFactorSpace< false >( mesh2, indices );
   const auto product_mesh = MakeCartesianProductMesh(
      mesh0, mesh1, mesh2 );
   const auto restriction = MakeTensorProductRestriction(
      space0, space1, space2 );
   const auto space = MakeFiniteElementSpace(
      product_mesh, GLFiniteElement< 1, 1, 1 >{}, restriction );

   std::vector< Real > values( space.GetNumberOfFiniteElementDofs() );
   for ( GlobalIndex i = 0; i < values.size(); ++i )
   {
      values[ i ] = static_cast< Real >( 100 + i );
   }
   auto view = MakeScalarElementTensorView( space, values.data() );
   const TestFaceView< 3 > face{
      GlobalIndex{ 7 }, Permutation< 3 >{ { -2, 3, -1 } } };
   Real * no_shared_memory = nullptr;
   KernelContext< HostKernelConfiguration, 0 > direct_context(
      no_shared_memory );
   KernelContext< FullSharedSerialKernelConfiguration, 0 > shared_context(
      no_shared_memory );
   const auto direct = ReadDofs( direct_context, space, face, view );
   const auto shared = ReadDofs( shared_context, space, face, view );

   bool success = true;
   UnitLoop< std::index_sequence< 2, 2, 2 > >( [&]( auto... indices )
   {
      const std::array< Integer, 3 > reference{
         static_cast< Integer >( indices )... };
      const auto native = ReferenceToNativeIndex(
         reference,
         std::array< size_t, 3 >{ 2, 2, 2 },
         face.orientation );
      const Real expected = view(
         native[ 0 ], native[ 1 ], native[ 2 ], face.cell_index );
      success = Check(
         std::abs( direct( indices... ) - expected ) < 1e-12,
         "three-factor direct read used the wrong source index" ) && success;
      success = Check(
         std::abs( shared( indices... ) - expected ) < 1e-12,
         "three-factor full-shared read used the wrong source index" ) &&
         success;
   });
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestOpaqueOrientedView< 1 >(
      { 4 }, Permutation< 1 >{ { -1 } } ) && success;
   success = TestOpaqueOrientedView< 2 >(
      { 2, 3 }, Permutation< 2 >{ { -1, -2 } } ) && success;
   success = TestOpaqueOrientedView< 3 >(
      { 2, 3, 2 }, Permutation< 3 >{ { 3, -2, -1 } } ) && success;
   success = TestOpaqueOrientedView< 4 >(
      { 2, 3, 2, 3 },
      Permutation< 4 >{ { 3, -4, -1, 2 } } ) && success;
   success = TestProxyPreservation() && success;
   success = TestIndirectedStridedFastPathEquivalence() && success;

   const auto orientations2 = std::vector< Permutation< 2 > >{
      { { 1, 2 } }, { { -1, 2 } }, { { 2, -1 } }, { { -2, -1 } } };
   const auto orientations3 = std::vector< Permutation< 3 > >{
      { { 1, 2, 3 } }, { { -1, 3, 2 } }, { { 2, -3, 1 } } };
   success = TestStridedFastPathEquivalence< 2 >(
      { 3, 3 }, orientations2 ) && success;
   success = TestStridedFastPathEquivalence< 3 >(
      { 2, 2, 2 }, orientations3 ) && success;

   success = TestTwoFactorRestriction< false, false >() && success;
   success = TestTwoFactorRestriction< true, false >() && success;
   success = TestTwoFactorRestriction< false, true >() && success;
   success = TestTwoFactorRestriction< true, true >() && success;
   success = TestThreeFactorRestriction() && success;
   return success ? 0 : 1;
}

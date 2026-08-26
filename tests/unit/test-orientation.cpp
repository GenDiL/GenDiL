// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace gendil;

namespace orientation_test
{

template < Integer CellDim >
struct RecordingCell
{
   static constexpr Integer Dim = CellDim;
   using geometry = HyperCube< Dim >;
   using physical_coordinates = std::array< Real, Dim >;
   using jacobian = std::array< Real, Dim >;
   template < typename >
   using QuadData = Empty;

   Permutation< Dim > applied_orientation =
      MakeReferencePermutation< Dim >();

   GENDIL_HOST_DEVICE
   jacobian ComputeJacobian( const Point< Dim > & ) const { return {}; }
};

template < Integer Dim >
GENDIL_HOST_DEVICE
void ApplyOrientationToCell(
   const Permutation< Dim > & orientation,
   RecordingCell< Dim > & cell )
{
   cell.applied_orientation = orientation;
}

template < Integer Dim >
GENDIL_HOST_DEVICE
void ApplyOrientationToCell(
   const IdentityOrientation< Dim > &,
   RecordingCell< Dim > & cell )
{
   cell.applied_orientation = MakeReferencePermutation< Dim >();
}

template < Integer MeshDim >
struct OrientedMesh
{
   static constexpr Integer Dim = MeshDim;
   using geometry = HyperCube< Dim >;
   using cell_type = RecordingCell< Dim >;

   GlobalIndex num_cells;
   Permutation< Dim > neighbor_orientation;

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfCells() const { return num_cells; }

   GENDIL_HOST_DEVICE
   cell_type GetCell( GlobalIndex ) const { return {}; }

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetLocalFaceInfo(
      const GlobalIndex cell_index,
      std::integral_constant< Integer, FaceIndex > ) const
   {
      constexpr Integer normal_axis =
         HyperCube< Dim >::GetNormalDimensionIndex( FaceIndex );
      constexpr int normal_sign =
         HyperCube< Dim >::GetNormalSign( FaceIndex );
      using face_info_type = ConformingCellFaceView<
         HyperCube< Dim >,
         std::integral_constant< Integer, FaceIndex >,
         std::integral_constant<
            Integer,
            HyperCube< Dim >::GetOppositeFaceIndex( FaceIndex ) >,
         Permutation< Dim >,
         CanonicalVector< Dim, normal_axis, normal_sign >,
         CanonicalVector< Dim, normal_axis, -normal_sign >,
         bool >;

      const GlobalIndex neighbor = ( cell_index + 1 ) % num_cells;
      return face_info_type{
         { cell_index, {}, {}, {}, {}, false },
         { neighbor, {}, neighbor_orientation, {}, {}, false } };
   }
};

} // namespace orientation_test

namespace
{

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cerr << "FAILED: " << message << "\n";
   }
   return condition;
}

constexpr Real identity_tolerance = 1.0e-13;

bool NearlyEqual( const Real left, const Real right )
{
   const Real scale = std::max(
      Real{ 1.0 }, std::max( std::abs( left ), std::abs( right ) ) );
   return std::abs( left - right ) <= identity_tolerance * scale;
}

template < typename T, size_t N >
bool NearlyEqual(
   const std::array< T, N > & left,
   const std::array< T, N > & right )
{
   for ( size_t i = 0; i < N; ++i )
   {
      if ( !NearlyEqual( left[ i ], right[ i ] ) )
      {
         return false;
      }
   }
   return true;
}

template < typename... T >
bool NearlyEqual(
   const std::tuple< T... > & left,
   const std::tuple< T... > & right );

template < typename... T, size_t... Is >
bool NearlyEqualTuple(
   const std::tuple< T... > & left,
   const std::tuple< T... > & right,
   std::index_sequence< Is... > )
{
   return ( NearlyEqual( std::get< Is >( left ), std::get< Is >( right ) ) &&
            ... );
}

template < typename... T >
bool NearlyEqual(
   const std::tuple< T... > & left,
   const std::tuple< T... > & right )
{
   return NearlyEqualTuple(
      left, right, std::index_sequence_for< T... >{} );
}

struct ProductCellProbeDofToQuad
{
   std::array< std::array< Real, 3 >, 2 > value{
      std::array< Real, 3 >{ 0.13, 0.61, 0.26 },
      std::array< Real, 3 >{ 0.42, 0.17, 0.41 } };
   std::array< std::array< Real, 3 >, 2 > gradient{
      std::array< Real, 3 >{ -0.9, 0.2, 0.7 },
      std::array< Real, 3 >{ -0.2, -0.4, 0.6 } };

   Real values( const LocalIndex q, const LocalIndex dof ) const
   {
      return value[ q ][ dof ];
   }

   Real gradients( const LocalIndex q, const LocalIndex dof ) const
   {
      return gradient[ q ][ dof ];
   }
};

LineCell< 3 > MakeNonAffineLineCell()
{
   static const Real nodes[]{ 0.2, 0.85, 2.1 };
   static const int restriction[]{ 0, 1, 2 };
   const StridedView< 1, const Real > node_view{
      PointerContainer< const Real >{ nodes },
      StridedLayout< 1 >{ GlobalIndex{ 1 } } };
   HostDevicePointer< const int > restriction_pointer;
   restriction_pointer.host_pointer = restriction;
   const HostDeviceStridedView< 2, const int > restriction_view{
      restriction_pointer,
      StridedLayout< 2 >{ GlobalIndex{ 1 }, GlobalIndex{ 3 } } };
   return { node_view, restriction_view, 0 };
}

QuadCell< 3 > MakeNonAffineQuadCell()
{
   static Real nodes[ 18 ];
   static int restriction[ 9 ];
   for ( int j = 0; j < 3; ++j )
   {
      for ( int i = 0; i < 3; ++i )
      {
         const int dof = i + 3 * j;
         restriction[ dof ] = dof;
         nodes[ 2 * dof ] =
            1.0 + 0.4 * i + 0.07 * j + 0.02 * i * j;
         nodes[ 2 * dof + 1 ] =
            -0.3 + 0.1 * i + 0.8 * j + 0.03 * i * j;
      }
   }
   const StridedView< 2, const Real > node_view{
      PointerContainer< const Real >{ nodes },
      StridedLayout< 2 >{ GlobalIndex{ 1 }, GlobalIndex{ 2 } } };
   HostDevicePointer< const int > restriction_pointer;
   restriction_pointer.host_pointer = restriction;
   const HostDeviceStridedView< 3, const int > restriction_view{
      restriction_pointer,
      StridedLayout< 3 >{
         GlobalIndex{ 1 }, GlobalIndex{ 3 }, GlobalIndex{ 9 } } };
   return { node_view, restriction_view, 0 };
}

template < typename Orientation, typename Cell >
concept CanApplyOrientationToProductCell = requires(
   const Orientation & orientation,
   Cell & cell )
{
   ApplyOrientationToCell( orientation, cell );
};

using HighOrderProductCell = ProductCell< LineCell< 3 >, QuadCell< 3 > >;

static_assert(
   CanApplyOrientationToProductCell<
      IdentityOrientation< HighOrderProductCell::Dim >,
      HighOrderProductCell > );
static_assert(
   !CanApplyOrientationToProductCell<
      IdentityOrientation< HighOrderProductCell::Dim - 1 >,
      HighOrderProductCell > );
static_assert(
   !CanApplyOrientationToProductCell<
      Permutation< HighOrderProductCell::Dim >,
      HighOrderProductCell > );
using HighOrderProductOrientation = TensorProductOrientation<
   Permutation< LineCell< 3 >::Dim >,
   Permutation< QuadCell< 3 >::Dim > >;
static_assert(
   CanApplyOrientationToProductCell<
      HighOrderProductOrientation,
      HighOrderProductCell > );

template < typename Product >
bool CheckProductCellNodes(
   const Product & left,
   const Product & right,
   const char * message )
{
   const auto & left_line = std::get< 0 >( left.Cells );
   const auto & right_line = std::get< 0 >( right.Cells );
   for ( LocalIndex i = 0; i < 3; ++i )
   {
      if ( left_line.nodes[ i ] != right_line.nodes[ i ] )
      {
         return Check( false, message );
      }
   }

   const auto & left_quad = std::get< 1 >( left.Cells );
   const auto & right_quad = std::get< 1 >( right.Cells );
   for ( LocalIndex j = 0; j < 3; ++j )
   {
      for ( LocalIndex i = 0; i < 3; ++i )
      {
         for ( LocalIndex component = 0; component < 2; ++component )
         {
            if ( left_quad.nodes[ i ][ j ][ component ] !=
                 right_quad.nodes[ i ][ j ][ component ] )
            {
               return Check( false, message );
            }
         }
      }
   }
   return true;
}

template < typename Cell, typename QData >
bool CheckCellEvaluation(
   const Cell & left,
   const Cell & right,
   const QData & qdata,
   const std::array< TensorIndex< Cell::Dim >, 3 > & quad_points,
   const char * message )
{
   for ( const auto & quad_point : quad_points )
   {
      typename Cell::physical_coordinates left_x{};
      typename Cell::physical_coordinates right_x{};
      typename Cell::jacobian left_j{};
      typename Cell::jacobian right_j{};
      left.GetValuesAndJacobian( quad_point, qdata, left_x, left_j );
      right.GetValuesAndJacobian( quad_point, qdata, right_x, right_j );
      if ( !NearlyEqual( left_x, right_x ) ||
           !NearlyEqual( left_j, right_j ) )
      {
         return Check( false, message );
      }
   }
   return true;
}

template < Integer Dim >
std::vector< Permutation< Dim > > MakeAllSignedPermutations()
{
   std::array< LocalIndex, Dim > axes{};
   for ( Integer axis = 0; axis < Dim; ++axis )
   {
      axes[ axis ] = static_cast< LocalIndex >( axis + 1 );
   }

   std::vector< Permutation< Dim > > permutations;
   do
   {
      for ( Integer signs = 0; signs < ( Integer{ 1 } << Dim ); ++signs )
      {
         Permutation< Dim > permutation{};
         for ( Integer axis = 0; axis < Dim; ++axis )
         {
            permutation( axis ) =
               signs & ( Integer{ 1 } << axis )
                  ? -axes[ axis ]
                  : axes[ axis ];
         }
         permutations.push_back( permutation );
      }
   } while ( std::next_permutation( axes.begin(), axes.end() ) );
   return permutations;
}

template < Integer Dim >
bool TestRoundTrips(
   const std::array< size_t, Dim > & sizes )
{
   bool success = true;
   for ( const auto & orientation : MakeAllSignedPermutations< Dim >() )
   {
      if ( !OrientedTensorDofShapeIsCompatible( sizes, orientation ) )
      {
         continue;
      }

      std::array< Integer, Dim > reference{};
      while ( true )
      {
         const auto native = ReferenceToNativeIndex(
            reference, sizes, orientation );
         const auto round_trip = NativeToReferenceIndex(
            native, sizes, orientation );
         success = Check(
            round_trip == reference,
            "reference/native orientation round trip failed" ) && success;
         for ( Integer axis = 0; axis < Dim; ++axis )
         {
            success = Check(
               native[ axis ] < sizes[ axis ],
               "oriented native index is out of bounds" ) && success;
         }

         Integer axis = 0;
         for ( ; axis < Dim; ++axis )
         {
            ++reference[ axis ];
            if ( reference[ axis ] < sizes[ axis ] )
            {
               break;
            }
            reference[ axis ] = 0;
         }
         if ( axis == Dim )
         {
            break;
         }
      }
   }
   return success;
}

bool TestValidityAndExtents()
{
   const std::array< size_t, 3 > anisotropic{ 2, 3, 4 };
   const std::array< size_t, 3 > repeated{ 2, 3, 2 };
   const auto structured_valid = MakeTensorProductOrientation(
      IdentityOrientation< 1 >{},
      Permutation< 2 >{ { -1, 2 } } );
   const auto structured_invalid = MakeTensorProductOrientation(
      IdentityOrientation< 1 >{},
      Permutation< 2 >{ { 0, 2 } } );
   const TensorProductOrientation nested_valid{
      IdentityOrientation< 1 >{},
      TensorProductOrientation{
         Permutation< 1 >{ { -1 } },
         IdentityOrientation< 1 >{} } };

   bool success = true;
   static_assert( IsValidOrientation( IdentityOrientation< 3 >{} ) );
   success = Check(
      IsValidOrientation(
         Permutation< 3 >{ { -1, 2, -3 } } ),
      "valid signed permutation rejected" ) && success;
   success = Check(
      !IsValidOrientation(
         Permutation< 3 >{ { 0, 2, 3 } } ),
      "zero orientation entry accepted" ) && success;
   success = Check(
      !IsValidOrientation(
         Permutation< 3 >{ { 1, 1, 3 } } ),
      "duplicate orientation axis accepted" ) && success;
   success = Check(
      !IsValidOrientation(
         Permutation< 3 >{ { 1, 2, 4 } } ),
      "out-of-range orientation axis accepted" ) && success;
   success = Check(
      IsValidOrientation( structured_valid ),
      "valid structured orientation rejected" ) && success;
   success = Check(
      !IsValidOrientation( structured_invalid ),
      "invalid structured orientation accepted" ) && success;
   success = Check(
      IsValidOrientation( nested_valid ),
      "valid nested structured orientation rejected" ) && success;
   success = Check(
      OrientedTensorDofShapeIsCompatible(
         anisotropic, Permutation< 3 >{ { -1, 2, -3 } } ),
      "anisotropic reversal rejected" ) && success;
   success = Check(
      !OrientedTensorDofShapeIsCompatible(
         anisotropic, Permutation< 3 >{ { 2, 1, 3 } } ),
      "unequal-extent permutation accepted" ) && success;
   success = Check(
      OrientedTensorDofShapeIsCompatible(
         repeated, Permutation< 3 >{ { 3, 2, -1 } } ),
      "equal-extent permutation rejected" ) && success;
   success = Check(
      OrientedTensorDofShapeIsCompatible(
         anisotropic, structured_valid ),
      "compatible structured orientation rejected" ) && success;
   success = Check(
      !OrientedTensorDofShapeIsCompatible(
         anisotropic,
         MakeTensorProductOrientation(
            IdentityOrientation< 1 >{},
            Permutation< 2 >{ { 2, 1 } } ) ),
      "structured unequal-extent permutation accepted" ) && success;
   return success;
}

bool TestSetAndGetSubPermutation()
{
   bool success = true;
   const Permutation< 2 > first{ { -2, 1 } };
   const Permutation< 2 > second{ { 2, -1 } };
   Permutation< 5 > product = MakeReferencePermutation< 5 >();
   Set< 0 >( product, first );
   Set< 3 >( product, second );

   success = Check(
      product == Permutation< 5 >{ { -2, 1, 3, 5, -4 } },
      "signed sub-permutation lifting is incorrect" ) && success;
   success = Check(
      GetSubPermutation< 2 >( product, 0 ) == first,
      "Set/GetSubPermutation round trip failed at offset zero" ) && success;
   success = Check(
      GetSubPermutation< 2 >( product, 3 ) == second,
      "Set/GetSubPermutation round trip failed at nonzero offset" ) && success;
   return success;
}

template < Integer HeadDim, Integer TailDim >
bool TestDirectSumTransform(
   const std::array< size_t, HeadDim > & head_sizes,
   const Permutation< HeadDim > & head_orientation,
   const std::array< size_t, TailDim > & tail_sizes,
   const Permutation< TailDim > & tail_orientation )
{
   constexpr Integer product_dim = HeadDim + TailDim;
   std::array< size_t, product_dim > product_sizes{};
   for ( Integer axis = 0; axis < HeadDim; ++axis )
   {
      product_sizes[ axis ] = head_sizes[ axis ];
   }
   for ( Integer axis = 0; axis < TailDim; ++axis )
   {
      product_sizes[ HeadDim + axis ] = tail_sizes[ axis ];
   }

   const auto product_orientation = MakeTensorProductOrientation(
      head_orientation, tail_orientation );
   const auto flattened_orientation = FlattenOrientation(
      product_orientation );
   auto flat_oracle = MakeReferencePermutation< product_dim >();
   Set< 0 >( flat_oracle, head_orientation );
   Set< HeadDim >( flat_oracle, tail_orientation );

   bool success = true;
   success = Check(
      flattened_orientation == flat_oracle,
      "structured product orientation does not match its flat oracle" ) &&
      success;
   const auto structured_layout = MakeOrientedLayout(
      product_sizes,
      product_orientation );
   const auto flat_layout = MakeOrientedLayout(
      product_sizes,
      flat_oracle );
   success = Check(
      structured_layout.offset == flat_layout.offset,
      "structured product layout offset does not match its flat oracle" ) &&
      success;
   for ( Integer axis = 0; axis < product_dim; ++axis )
   {
      success = Check(
         structured_layout.strides[ axis ] == flat_layout.strides[ axis ],
         "structured product layout stride does not match its flat oracle" ) &&
         success;
   }

   std::array< Integer, product_dim > product_reference{};
   while ( true )
   {
      std::array< Integer, HeadDim > head_reference{};
      std::array< Integer, TailDim > tail_reference{};
      for ( Integer axis = 0; axis < HeadDim; ++axis )
      {
         head_reference[ axis ] = product_reference[ axis ];
      }
      for ( Integer axis = 0; axis < TailDim; ++axis )
      {
         tail_reference[ axis ] = product_reference[ HeadDim + axis ];
      }

      const auto product_native = ReferenceToNativeIndex(
         product_reference, product_sizes, product_orientation );
      const auto flat_native = ReferenceToNativeIndex(
         product_reference, product_sizes, flat_oracle );
      const auto head_native = ReferenceToNativeIndex(
         head_reference, head_sizes, head_orientation );
      const auto tail_native = ReferenceToNativeIndex(
         tail_reference, tail_sizes, tail_orientation );
      for ( Integer axis = 0; axis < HeadDim; ++axis )
      {
         success = Check(
            product_native[ axis ] == head_native[ axis ] &&
               product_native[ axis ] == flat_native[ axis ],
            "product index transform crossed the head-factor boundary" ) &&
            success;
      }
      for ( Integer axis = 0; axis < TailDim; ++axis )
      {
         success = Check(
            product_native[ HeadDim + axis ] == tail_native[ axis ] &&
               product_native[ HeadDim + axis ] == flat_native[ HeadDim + axis ],
            "product index transform crossed the tail-factor boundary" ) &&
            success;
      }

      Integer axis = 0;
      for ( ; axis < product_dim; ++axis )
      {
         ++product_reference[ axis ];
         if ( product_reference[ axis ] < product_sizes[ axis ] )
         {
            break;
         }
         product_reference[ axis ] = 0;
      }
      if ( axis == product_dim )
      {
         break;
      }
   }
   return success;
}

bool TestTensorProductTransformComposition()
{
   using MixedOrientation = decltype( MakeTensorProductOrientation(
      Permutation< 2 >{}, IdentityOrientation< 4 >{} ) );
   static_assert( sizeof( MixedOrientation ) == sizeof( Permutation< 2 > ) );
   using AllRuntimeOrientation = decltype( MakeTensorProductOrientation(
      Permutation< 2 >{}, Permutation< 1 >{} ) );
   static_assert(
      is_runtime_permutation_orientation_v< AllRuntimeOrientation > );
   static_assert(
      !is_runtime_permutation_orientation_v< MixedOrientation > );
   static_assert(
      sizeof( AllRuntimeOrientation ) ==
         sizeof( Permutation< 2 > ) + sizeof( Permutation< 1 > ) );
   using NestedRuntimeOrientation = TensorProductOrientation<
      Permutation< 2 >,
      TensorProductOrientation< Permutation< 2 >, Permutation< 2 > > >;
   static_assert(
      sizeof( NestedRuntimeOrientation ) ==
         3 * sizeof( Permutation< 2 > ) );
   static_assert( std::same_as<
      decltype( MakeTensorProductOrientation(
         IdentityOrientation< 2 >{}, IdentityOrientation< 4 >{} ) ),
      IdentityOrientation< 6 > > );

   bool success = true;
   success = TestDirectSumTransform< 1, 1 >(
      { 3 }, Permutation< 1 >{ { -1 } },
      { 5 }, Permutation< 1 >{ { -1 } } ) && success;
   success = TestDirectSumTransform< 2, 1 >(
      { 2, 2 }, Permutation< 2 >{ { 2, -1 } },
      { 4 }, Permutation< 1 >{ { -1 } } ) && success;
   success = TestDirectSumTransform< 1, 1 >(
      { 2 }, Permutation< 1 >{ { -1 } },
      { 2 }, Permutation< 1 >{ { 1 } } ) && success;

   Permutation< 2 > first_two = MakeReferencePermutation< 2 >();
   Set< 0 >( first_two, Permutation< 1 >{ { -1 } } );
   Set< 1 >( first_two, Permutation< 1 >{ { -1 } } );
   success = TestDirectSumTransform< 2, 1 >(
      { 2, 3 }, first_two,
      { 4 }, Permutation< 1 >{ { -1 } } ) && success;

   Permutation< 3 > left_associated = MakeReferencePermutation< 3 >();
   Set< 0 >( left_associated, first_two );
   Set< 2 >( left_associated, Permutation< 1 >{ { -1 } } );
   Permutation< 3 > factorwise = MakeReferencePermutation< 3 >();
   Set< 0 >( factorwise, Permutation< 1 >{ { -1 } } );
   Set< 1 >( factorwise, Permutation< 1 >{ { -1 } } );
   Set< 2 >( factorwise, Permutation< 1 >{ { -1 } } );
   success = Check(
      left_associated == factorwise,
      "recursive product orientation composition is not associative" ) &&
      success;
   return success;
}

bool TestExhaustiveTensorProductTransformComposition()
{
   bool success = true;
   const auto head_orientations = MakeAllSignedPermutations< 2 >();
   const auto tail_orientations = MakeAllSignedPermutations< 2 >();
   for ( const auto & head : head_orientations )
   {
      for ( const auto & tail : tail_orientations )
      {
         success = TestDirectSumTransform< 2, 2 >(
            { 2, 2 }, head,
            { 3, 3 }, tail ) && success;
      }
   }
   const auto three_dimensional_orientations =
      MakeAllSignedPermutations< 3 >();
   for ( const auto & head : head_orientations )
   {
      for ( const auto & tail : three_dimensional_orientations )
      {
         success = TestDirectSumTransform< 2, 3 >(
            { 2, 2 }, head,
            { 2, 2, 2 }, tail ) && success;
      }
   }
   return success;
}

bool TestProductMeshConnectivity()
{
   using orientation_test::OrientedMesh;
   const OrientedMesh< 2 > head{
      2, Permutation< 2 >{ { -2, 1 } } };
   const OrientedMesh< 1 > tail{
      3, Permutation< 1 >{ { -1 } } };
   const auto mesh = MakeCartesianProductMesh( head, tail );

   const auto head_face = mesh.GetLocalFaceInfo(
      GlobalIndex{ 0 }, std::integral_constant< Integer, 0 >{} );
   const auto tail_face = mesh.GetLocalFaceInfo(
      GlobalIndex{ 0 }, std::integral_constant< Integer, 2 >{} );

   using HeadInfo = std::remove_cvref_t< decltype( head_face ) >;
   using TailInfo = std::remove_cvref_t< decltype( tail_face ) >;
   bool success = true;
   success = Check(
      FlattenOrientation( head_face.PlusSide().GetOrientation() ) ==
         Permutation< 3 >{ { -2, 1, 3 } },
      "head-factor orientation was not embedded as a leading block" ) &&
      success;
   success = Check(
      FlattenOrientation( tail_face.PlusSide().GetOrientation() ) ==
         Permutation< 3 >{ { 1, 2, -3 } },
      "tail-factor orientation was not lifted by the full head rank" ) &&
      success;
   success = Check(
      FlattenOrientation( head_face.MinusSide().GetOrientation() ) ==
         MakeReferencePermutation< 3 >(),
      "product global-face minus orientation is not canonical" ) && success;
   success = Check(
      !head_face.IsBoundary() && !tail_face.IsBoundary(),
      "product connectivity changed the boundary flag" ) && success;
   success = Check(
      HeadInfo::minus_side_type::local_face_index_type::value == 0 &&
      HeadInfo::plus_side_type::local_face_index_type::value == 3,
      "product head face IDs are incorrect" ) && success;
   success = Check(
      TailInfo::minus_side_type::local_face_index_type::value == 2 &&
      TailInfo::plus_side_type::local_face_index_type::value == 5,
      "product tail face IDs are incorrect" ) && success;
   success = Check(
      HeadInfo::minus_side_type::normal_type::index == 0 &&
      HeadInfo::minus_side_type::normal_type::sign == -1 &&
      HeadInfo::plus_side_type::normal_type::sign == 1,
      "product head normals are incorrect" ) && success;
   success = Check(
      TailInfo::minus_side_type::normal_type::index == 2 &&
      TailInfo::minus_side_type::normal_type::sign == -1 &&
      TailInfo::plus_side_type::normal_type::sign == 1,
      "product tail normals are incorrect" ) && success;

   auto cell = mesh.GetCell( 0 );
   const auto cell_orientation = MakeTensorProductOrientation(
      Permutation< 2 >{ { -2, 1 } },
      Permutation< 1 >{ { -1 } } );
   ApplyOrientationToCell( cell_orientation, cell );
   success = Check(
      std::get< 0 >( cell.Cells ).applied_orientation ==
         Permutation< 2 >{ { -2, 1 } } &&
      std::get< 1 >( cell.Cells ).applied_orientation ==
         Permutation< 1 >{ { -1 } },
      "product-cell geometry orientation was not split by factor" ) &&
      success;
   return success;
}

bool TestRecursiveProductComposition()
{
   using orientation_test::OrientedMesh;
   const OrientedMesh< 2 > first{
      2, Permutation< 2 >{ { 2, -1 } } };
   const OrientedMesh< 1 > second{
      2, Permutation< 1 >{ { -1 } } };
   const OrientedMesh< 1 > third{
      2, Permutation< 1 >{ { -1 } } };
   const auto nested = MakeCartesianProductMesh( first, second );
   const auto recursive = MakeCartesianProductMesh( nested, third );
   const auto right_nested = MakeCartesianProductMesh( second, third );
   const auto right_recursive = MakeCartesianProductMesh( first, right_nested );

   const auto second_factor_face = recursive.GetLocalFaceInfo(
      GlobalIndex{ 0 }, std::integral_constant< Integer, 2 >{} );
   const auto third_factor_face = recursive.GetLocalFaceInfo(
      GlobalIndex{ 0 }, std::integral_constant< Integer, 3 >{} );
   const auto right_second_factor_face = right_recursive.GetLocalFaceInfo(
      GlobalIndex{ 0 }, std::integral_constant< Integer, 2 >{} );
   using RightOrientation = std::remove_cvref_t< decltype(
      right_second_factor_face.PlusSide().GetOrientation() ) >;
   static_assert( RightOrientation::num_components == 2 );
   static_assert( is_tensor_product_orientation_v<
      std::tuple_element_t< 1, typename RightOrientation::component_types > > );

   auto right_cell = right_recursive.GetCell( 0 );
   ApplyOrientationToCell(
      right_second_factor_face.PlusSide().GetOrientation(),
      right_cell );

   return Check(
      FlattenOrientation( second_factor_face.PlusSide().GetOrientation() ) ==
         Permutation< 4 >{ { 1, 2, -3, 4 } },
      "recursive product did not preserve the nested factor block" ) &&
      Check(
         FlattenOrientation( third_factor_face.PlusSide().GetOrientation() ) ==
            Permutation< 4 >{ { 1, 2, 3, -4 } },
         "recursive product did not lift the final factor orientation" ) &&
      Check(
         FlattenOrientation(
            right_second_factor_face.PlusSide().GetOrientation() ) ==
            Permutation< 4 >{ { 1, 2, -3, 4 } },
         "right-nested product did not preserve the nested factor block" ) &&
      Check(
         std::get< 0 >(
            std::get< 1 >( right_cell.Cells ).Cells ).applied_orientation ==
            Permutation< 1 >{ { -1 } },
         "right-nested product-cell orientation did not reach its leaf" );
}

bool TestRuntimeIdentityIsProductCellNoOp()
{
   const ProductCell original{
      MakeNonAffineLineCell(), MakeNonAffineQuadCell() };
   auto runtime_identity = original;
   auto static_identity = original;
   const auto runtime_identity_orientation = MakeTensorProductOrientation(
      Permutation< 1 >{ { 1 } },
      Permutation< 2 >{ { 1, 2 } } );
   ApplyOrientationToCell(
      runtime_identity_orientation,
      runtime_identity );
   ApplyOrientationToCell(
      IdentityOrientation< decltype( original )::Dim >{},
      static_identity );

   const ProductCellProbeDofToQuad basis{};
   const auto line_qdata = MakeTensorProductData( basis );
   const auto quad_qdata = MakeTensorProductData( basis, basis );
   const auto product_qdata =
      MakeTensorProductData( line_qdata, quad_qdata );
   const std::array line_quad_points{
      TensorIndex< 1 >{ GlobalIndex{ 0 } },
      TensorIndex< 1 >{ GlobalIndex{ 1 } },
      TensorIndex< 1 >{ GlobalIndex{ 0 } } };
   const std::array quad_quad_points{
      TensorIndex< 2 >{ GlobalIndex{ 0 }, GlobalIndex{ 0 } },
      TensorIndex< 2 >{ GlobalIndex{ 1 }, GlobalIndex{ 0 } },
      TensorIndex< 2 >{ GlobalIndex{ 0 }, GlobalIndex{ 1 } } };
   const std::array product_quad_points{
      TensorIndex< 3 >{
         GlobalIndex{ 0 }, GlobalIndex{ 0 }, GlobalIndex{ 0 } },
      TensorIndex< 3 >{
         GlobalIndex{ 1 }, GlobalIndex{ 0 }, GlobalIndex{ 1 } },
      TensorIndex< 3 >{
         GlobalIndex{ 1 }, GlobalIndex{ 1 }, GlobalIndex{ 0 } } };

   bool success = CheckProductCellNodes(
      original,
      runtime_identity,
      "runtime identity changed ProductCell component nodes" );
   success = CheckProductCellNodes(
      original,
      static_identity,
      "static identity changed ProductCell component nodes" ) &&
      success;
   success = CheckCellEvaluation(
      std::get< 0 >( original.Cells ),
      std::get< 0 >( runtime_identity.Cells ),
      line_qdata,
      line_quad_points,
      "runtime identity changed the line-cell component result" ) &&
      success;
   success = CheckCellEvaluation(
      std::get< 1 >( original.Cells ),
      std::get< 1 >( runtime_identity.Cells ),
      quad_qdata,
      quad_quad_points,
      "runtime identity changed the quad-cell component result" ) &&
      success;
   success = CheckCellEvaluation(
      std::get< 0 >( original.Cells ),
      std::get< 0 >( static_identity.Cells ),
      line_qdata,
      line_quad_points,
      "static identity changed the line-cell component result" ) &&
      success;
   success = CheckCellEvaluation(
      std::get< 1 >( original.Cells ),
      std::get< 1 >( static_identity.Cells ),
      quad_qdata,
      quad_quad_points,
      "static identity changed the quad-cell component result" ) &&
      success;
   success = CheckCellEvaluation(
      original,
      runtime_identity,
      product_qdata,
      product_quad_points,
      "runtime identity changed ProductCell coordinates or Jacobians" ) &&
      success;
   success = CheckCellEvaluation(
      original,
      static_identity,
      product_qdata,
      product_quad_points,
      "static identity changed ProductCell coordinates or Jacobians" ) &&
      success;

   const ProductCell nested_original{ original, MakeNonAffineLineCell() };
   auto nested_runtime_identity = nested_original;
   auto nested_static_identity = nested_original;
   const auto nested_runtime_identity_orientation =
      MakeTensorProductOrientation(
         runtime_identity_orientation,
         Permutation< 1 >{ { 1 } } );
   ApplyOrientationToCell(
      nested_runtime_identity_orientation,
      nested_runtime_identity );
   ApplyOrientationToCell(
      IdentityOrientation< decltype( nested_original )::Dim >{},
      nested_static_identity );
   const auto nested_qdata =
      MakeTensorProductData( product_qdata, line_qdata );
   const std::array nested_quad_points{
      TensorIndex< 4 >{
         GlobalIndex{ 0 }, GlobalIndex{ 0 }, GlobalIndex{ 0 },
         GlobalIndex{ 1 } },
      TensorIndex< 4 >{
         GlobalIndex{ 1 }, GlobalIndex{ 0 }, GlobalIndex{ 1 },
         GlobalIndex{ 0 } },
      TensorIndex< 4 >{
         GlobalIndex{ 1 }, GlobalIndex{ 1 }, GlobalIndex{ 0 },
         GlobalIndex{ 1 } } };

   success = CheckProductCellNodes(
      std::get< 0 >( nested_original.Cells ),
      std::get< 0 >( nested_runtime_identity.Cells ),
      "runtime identity changed nested ProductCell component nodes" ) &&
      success;
   success = CheckProductCellNodes(
      std::get< 0 >( nested_original.Cells ),
      std::get< 0 >( nested_static_identity.Cells ),
      "static identity changed nested ProductCell component nodes" ) &&
      success;
   for ( LocalIndex i = 0; i < 3; ++i )
   {
      success = Check(
         std::get< 1 >( nested_original.Cells ).nodes[ i ] ==
            std::get< 1 >( nested_runtime_identity.Cells ).nodes[ i ],
         "runtime identity changed nested ProductCell tail nodes" ) &&
         success;
      success = Check(
         std::get< 1 >( nested_original.Cells ).nodes[ i ] ==
            std::get< 1 >( nested_static_identity.Cells ).nodes[ i ],
         "static identity changed nested ProductCell tail nodes" ) &&
         success;
   }
   success = CheckCellEvaluation(
      nested_original,
      nested_runtime_identity,
      nested_qdata,
      nested_quad_points,
      "runtime identity changed nested ProductCell coordinates or Jacobians" ) &&
      success;
   success = CheckCellEvaluation(
      nested_original,
      nested_static_identity,
      nested_qdata,
      nested_quad_points,
      "static identity changed nested ProductCell coordinates or Jacobians" ) &&
      success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestRoundTrips< 1 >( { 4 } ) && success;
   success = TestRoundTrips< 2 >( { 3, 3 } ) && success;
   success = TestRoundTrips< 2 >( { 2, 3 } ) && success;
   success = TestRoundTrips< 3 >( { 2, 2, 2 } ) && success;
   success = TestRoundTrips< 3 >( { 2, 3, 4 } ) && success;
   success = TestValidityAndExtents() && success;
   success = TestSetAndGetSubPermutation() && success;
   success = TestTensorProductTransformComposition() && success;
   success = TestExhaustiveTensorProductTransformComposition() && success;
   success = TestProductMeshConnectivity() && success;
   success = TestRecursiveProductComposition() && success;
   success = TestRuntimeIdentityIsProductCellNoOp() && success;
   return success ? 0 : 1;
}

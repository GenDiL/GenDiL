// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <iostream>
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

   bool success = true;
   success = Check(
      FaceReadDofsOrientationIsValid(
         Permutation< 3 >{ { -1, 2, -3 } } ),
      "valid signed permutation rejected" ) && success;
   success = Check(
      !FaceReadDofsOrientationIsValid(
         Permutation< 3 >{ { 0, 2, 3 } } ),
      "zero orientation entry accepted" ) && success;
   success = Check(
      !FaceReadDofsOrientationIsValid(
         Permutation< 3 >{ { 1, 1, 3 } } ),
      "duplicate orientation axis accepted" ) && success;
   success = Check(
      !FaceReadDofsOrientationIsValid(
         Permutation< 3 >{ { 1, 2, 4 } } ),
      "out-of-range orientation axis accepted" ) && success;
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

   auto product_orientation = MakeReferencePermutation< product_dim >();
   Set< 0 >( product_orientation, head_orientation );
   Set< HeadDim >( product_orientation, tail_orientation );

   bool success = true;
   success = Check(
      GetSubPermutation< HeadDim >( product_orientation, 0 ) ==
         head_orientation,
      "product orientation changed the head-factor block" ) && success;
   success = Check(
      GetSubPermutation< TailDim >( product_orientation, HeadDim ) ==
         tail_orientation,
      "product orientation changed the tail-factor block" ) && success;

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
      const auto head_native = ReferenceToNativeIndex(
         head_reference, head_sizes, head_orientation );
      const auto tail_native = ReferenceToNativeIndex(
         tail_reference, tail_sizes, tail_orientation );
      for ( Integer axis = 0; axis < HeadDim; ++axis )
      {
         success = Check(
            product_native[ axis ] == head_native[ axis ],
            "product index transform crossed the head-factor boundary" ) &&
            success;
      }
      for ( Integer axis = 0; axis < TailDim; ++axis )
      {
         success = Check(
            product_native[ HeadDim + axis ] == tail_native[ axis ],
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
      head_face.PlusSide().GetOrientation() ==
         Permutation< 3 >{ { -2, 1, 3 } },
      "head-factor orientation was not embedded as a leading block" ) &&
      success;
   success = Check(
      tail_face.PlusSide().GetOrientation() ==
         Permutation< 3 >{ { 1, 2, -3 } },
      "tail-factor orientation was not lifted by the full head rank" ) &&
      success;
   success = Check(
      static_cast< Permutation< 3 > >(
         head_face.MinusSide().GetOrientation() ) ==
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
   const Permutation< 3 > cell_orientation{ { -2, 1, -3 } };
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

   const auto second_factor_face = recursive.GetLocalFaceInfo(
      GlobalIndex{ 0 }, std::integral_constant< Integer, 2 >{} );
   const auto third_factor_face = recursive.GetLocalFaceInfo(
      GlobalIndex{ 0 }, std::integral_constant< Integer, 3 >{} );
   return Check(
      second_factor_face.PlusSide().GetOrientation() ==
         Permutation< 4 >{ { 1, 2, -3, 4 } },
      "recursive product did not preserve the nested factor block" ) &&
      Check(
         third_factor_face.PlusSide().GetOrientation() ==
            Permutation< 4 >{ { 1, 2, 3, -4 } },
         "recursive product did not lift the final factor orientation" );
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
   success = TestProductMeshConnectivity() && success;
   success = TestRecursiveProductComposition() && success;
   return success ? 0 : 1;
}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mfem_point_matrix_decoder_test_helpers.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace gendil;
using namespace gendil::tests::mfem_nc_preflight;

namespace
{

bool Check( bool condition, const std::string & message )
{
   if ( !condition )
   {
      std::cerr << "FAILED: " << message << "\n";
   }
   return condition;
}

template < Integer Dim >
bool CheckPermutation(
   const Permutation< Dim > & got,
   const Permutation< Dim > & expected,
   const std::string & message )
{
   if ( got != expected )
   {
      std::cerr << "FAILED: " << message
                << " got " << PermutationString( got )
                << " expected " << PermutationString( expected ) << "\n";
      return false;
   }
   return true;
}

template < Integer Dim >
bool CheckPoint(
   const Point< Dim > & got,
   const Point< Dim > & expected,
   const std::string & message )
{
   if ( MaxError( got, expected ) > DecoderTolerance() )
   {
      std::cerr << "FAILED: " << message
                << " got " << PointString( got )
                << " expected " << PointString( expected ) << "\n";
      return false;
   }
   return true;
}

template < Integer Dim >
bool CheckSize(
   const NonconformingHyperCubeFaceMap< Dim > & map,
   const std::array< Real, Dim > & expected,
   const std::string & message )
{
   bool ok = true;
   for ( Integer d = 0; d < Dim; ++d )
   {
      if ( std::abs( map.size[d] - expected[d] ) > DecoderTolerance() )
      {
         std::cerr << "FAILED: " << message << " axis " << d
                   << " got " << map.size[d]
                   << " expected " << expected[d] << "\n";
         ok = false;
      }
   }
   return ok;
}

PointMatrixDecodeResult< 2 > Decode2(
   int coarse_lf,
   int fine_lf,
   const std::array< Real, 2 > & pm )
{
   return DecodeMFEMPointMatrix(
      coarse_lf,
      fine_lf,
      TranslateMFEMFace2D( coarse_lf ),
      TranslateMFEMFace2D( fine_lf ),
      SegmentPointMatrix{ pm } );
}

PointMatrixDecodeResult< 3 > Decode3(
   int coarse_lf,
   int fine_lf,
   const std::array< Real, 8 > & pm )
{
   return DecodeMFEMPointMatrix(
      coarse_lf,
      fine_lf,
      TranslateMFEMFace3D( coarse_lf ),
      TranslateMFEMFace3D( fine_lf ),
      SquarePointMatrix{ {
         Point< 2 >{ pm[0], pm[4] },
         Point< 2 >{ pm[1], pm[5] },
         Point< 2 >{ pm[2], pm[6] },
         Point< 2 >{ pm[3], pm[7] }
      } } );
}

bool CheckValid2D(
   const std::string & name,
   int coarse_lf,
   int fine_lf,
   const std::array< Real, 2 > & pm,
   const Permutation< 2 > & expected_orientation,
   const Point< 2 > & expected_origin,
   const std::array< Real, 2 > & expected_size )
{
   const auto decoded = Decode2( coarse_lf, fine_lf, pm );
   bool ok = true;
   ok = Check(
      IsSuccessfulDecode( decoded ),
      name + ": decode succeeded: " + DecodeErrorString( decoded.error ) ) && ok;
   ok = Check( decoded.face_candidate_count == 2, name + ": face candidate count" ) && ok;
   ok = Check( decoded.corner_candidate_count == 2, name + ": corner candidate count" ) && ok;
   ok = Check( decoded.map_candidate_count == 1, name + ": map candidate count" ) && ok;
   ok = CheckPermutation( decoded.value.plus_orientation, expected_orientation, name + ": orientation" ) && ok;
   ok = CheckPoint( decoded.value.minus_map.origin, expected_origin, name + ": origin" ) && ok;
   ok = CheckSize< 2 >( decoded.value.minus_map, expected_size, name + ": size" ) && ok;
   return ok;
}

bool CheckValid3D(
   const std::string & name,
   int coarse_lf,
   int fine_lf,
   const std::array< Real, 8 > & pm,
   const Permutation< 3 > & expected_orientation,
   const Point< 3 > & expected_origin,
   const std::array< Real, 3 > & expected_size )
{
   const auto decoded = Decode3( coarse_lf, fine_lf, pm );
   bool ok = true;
   ok = Check(
      IsSuccessfulDecode( decoded ),
      name + ": decode succeeded: " + DecodeErrorString( decoded.error ) ) && ok;
   ok = Check( decoded.face_candidate_count == 8, name + ": face candidate count" ) && ok;
   ok = Check( decoded.corner_candidate_count == 8, name + ": corner candidate count" ) && ok;
   ok = Check( decoded.map_candidate_count == 1, name + ": map candidate count" ) && ok;
   ok = CheckPermutation( decoded.value.plus_orientation, expected_orientation, name + ": orientation" ) && ok;
   ok = CheckPoint( decoded.value.minus_map.origin, expected_origin, name + ": origin" ) && ok;
   ok = CheckSize< 3 >( decoded.value.minus_map, expected_size, name + ": size" ) && ok;
   return ok;
}

bool CheckInvalid2D(
   const std::string & name,
   int coarse_lf,
   int fine_lf,
   const std::array< Real, 2 > & pm )
{
   const auto decoded = Decode2( coarse_lf, fine_lf, pm );
   return Check( !IsSuccessfulDecode( decoded ), name + ": invalid matrix rejected" );
}

bool CheckInvalid3D(
   const std::string & name,
   int coarse_lf,
   int fine_lf,
   const std::array< Real, 8 > & pm )
{
   const auto decoded = Decode3( coarse_lf, fine_lf, pm );
   return Check( !IsSuccessfulDecode( decoded ), name + ": invalid matrix rejected" );
}

std::array< Real, 8 > Matrix3D( Real x0, Real z0, Real sx, Real sz )
{
   return {
      x0, x0 + sx, x0 + sx, x0,
      z0, z0,      z0 + sz, z0 + sz
   };
}

template < Integer Dim >
std::array< Real, Dim > SizeArray(
   const NonconformingHyperCubeFaceMap< Dim > & map )
{
   std::array< Real, Dim > out{};
   for ( Integer d = 0; d < Dim; ++d ) { out[d] = map.size[d]; }
   return out;
}

template < Integer Dim >
NonconformingHyperCubeFaceMap< Dim > MakeExpectedMap(
   Integer coarse_g0,
   const Point< Dim > & origin,
   const std::array< Real, Dim > & size )
{
   NonconformingHyperCubeFaceMap< Dim > map;
   map.origin = origin;
   for ( Integer d = 0; d < Dim; ++d ) { map.size[d] = size[d]; }
   const Integer normal_axis =
      HyperCube< Dim >::GetNormalDimensionIndex( coarse_g0 );
   map.origin[normal_axis] = Real( 0 );
   map.size[normal_axis] = Real( 1 );
   return map;
}

std::array< Real, 2 > EncodeSegmentPointMatrix(
   int coarse_lf,
   int fine_lf,
   const Permutation< 2 > & orientation,
   const NonconformingHyperCubeFaceMap< 2 > & map )
{
   const Integer coarse_g0 = TranslateMFEMFace2D( coarse_lf );
   const Integer static_plus = HyperCube< 2 >::GetOppositeFaceIndex( coarse_g0 );
   std::array< Real, 2 > pm{};
   for ( Real t : { Real( 0 ), Real( 1 ) } )
   {
      const std::array< Real, 1 > tangent{ t };
      const auto q_plus = CanonicalFacePoint< 2 >( static_plus, tangent );
      const auto fine_native = ApplyOrientation( q_plus, orientation );
      const Real fine_t = NativeToMFEMFacePoint2D( fine_lf, fine_native );
      const int col = SegmentPointMatrixColumnForSlaveCoordinate( fine_t );
      const auto q_leaf = CanonicalFacePoint< 2 >( coarse_g0, tangent );
      const auto parent = ApplyDiagonalMap< 2 >( map, q_leaf );
      pm[col] = NativeToMFEMFacePoint2D( coarse_lf, parent );
   }
   return pm;
}

std::array< Real, 8 > EncodeSquarePointMatrix(
   int coarse_lf,
   int fine_lf,
   const Permutation< 3 > & orientation,
   const NonconformingHyperCubeFaceMap< 3 > & map )
{
   const Integer coarse_g0 = TranslateMFEMFace3D( coarse_lf );
   const Integer static_plus = HyperCube< 3 >::GetOppositeFaceIndex( coarse_g0 );
   std::array< Real, 8 > pm{};
   const std::array< std::array< Real, 2 >, 4 > corners{
      std::array< Real, 2 >{ 0, 0 },
      std::array< Real, 2 >{ 1, 0 },
      std::array< Real, 2 >{ 0, 1 },
      std::array< Real, 2 >{ 1, 1 }
   };
   for ( const auto & tangent : corners )
   {
      const auto q_plus = CanonicalFacePoint< 3 >( static_plus, tangent );
      const auto fine_native = ApplyOrientation( q_plus, orientation );
      const auto fine_uv = NativeToMFEMFacePoint3D( fine_lf, fine_native );
      const int col = SquarePointMatrixColumnForSlaveCoordinate( fine_uv );
      const auto q_leaf = CanonicalFacePoint< 3 >( coarse_g0, tangent );
      const auto parent = ApplyDiagonalMap< 3 >( map, q_leaf );
      const auto coarse_uv = NativeToMFEMFacePoint3D( coarse_lf, parent );
      pm[col] = coarse_uv[0];
      pm[4 + col] = coarse_uv[1];
   }
   return pm;
}

bool TestLiteralValidCases()
{
   bool ok = true;
   ok = CheckValid2D(
      "2D lower interval",
      1,
      1,
      { 0.5, 0.0 },
      MakePermutation< 2 >( { -1, 2 } ),
      Point< 2 >{ 0.0, 0.0 },
      { 1.0, 0.5 } ) && ok;
   ok = CheckValid2D(
      "2D upper interval",
      1,
      1,
      { 1.0, 0.5 },
      MakePermutation< 2 >( { -1, 2 } ),
      Point< 2 >{ 0.0, 0.5 },
      { 1.0, 0.5 } ) && ok;
   ok = CheckValid2D(
      "2D reversed fine parameterization",
      1,
      2,
      { 0.5, 0.0 },
      MakePermutation< 2 >( { -2, -1 } ),
      Point< 2 >{ 0.0, 0.0 },
      { 1.0, 0.5 } ) && ok;
   ok = CheckValid2D(
      "2D nonidentity normal sign",
      1,
      0,
      { 0.5, 0.0 },
      MakePermutation< 2 >( { 2, 1 } ),
      Point< 2 >{ 0.0, 0.0 },
      { 1.0, 0.5 } ) && ok;

   ok = CheckValid3D(
      "3D quadrant 00",
      1,
      3,
      Matrix3D( 0.0, 0.0, 0.5, 0.5 ),
      MakePermutation< 3 >( { -1, 2, 3 } ),
      Point< 3 >{ 0.0, 0.0, 0.0 },
      { 0.5, 1.0, 0.5 } ) && ok;
   ok = CheckValid3D(
      "3D quadrant 10",
      1,
      3,
      Matrix3D( 0.5, 0.0, 0.5, 0.5 ),
      MakePermutation< 3 >( { -1, 2, 3 } ),
      Point< 3 >{ 0.5, 0.0, 0.0 },
      { 0.5, 1.0, 0.5 } ) && ok;
   ok = CheckValid3D(
      "3D quadrant 01",
      1,
      3,
      Matrix3D( 0.0, 0.5, 0.5, 0.5 ),
      MakePermutation< 3 >( { -1, 2, 3 } ),
      Point< 3 >{ 0.0, 0.0, 0.5 },
      { 0.5, 1.0, 0.5 } ) && ok;
   ok = CheckValid3D(
      "3D quadrant 11 symmetric",
      1,
      3,
      Matrix3D( 0.5, 0.5, 0.5, 0.5 ),
      MakePermutation< 3 >( { -1, 2, 3 } ),
      Point< 3 >{ 0.5, 0.0, 0.5 },
      { 0.5, 1.0, 0.5 } ) && ok;
   ok = CheckValid3D(
      "3D tangential swap",
      1,
      1,
      { 0.5, 0.5, 0.0, 0.0,
        0.25, 0.0, 0.0, 0.25 },
      MakePermutation< 3 >( { -3, -2, -1 } ),
      Point< 3 >{ 0.0, 0.0, 0.0 },
      { 0.5, 1.0, 0.25 } ) && ok;
   ok = CheckValid3D(
      "3D swap plus reflection",
      1,
      1,
      { 0.0, 0.0, 0.25, 0.25,
        0.0, 0.5, 0.5, 0.0 },
      MakePermutation< 3 >( { 3, -2, 1 } ),
      Point< 3 >{ 0.0, 0.0, 0.0 },
      { 0.25, 1.0, 0.5 } ) && ok;
   ok = CheckValid3D(
      "3D nonidentity normal sign",
      1,
      1,
      Matrix3D( 0.25, 0.25, 0.5, 0.5 ),
      MakePermutation< 3 >( { 1, -2, 3 } ),
      Point< 3 >{ 0.25, 0.0, 0.25 },
      { 0.5, 1.0, 0.5 } ) && ok;
   return ok;
}

bool TestInvalidCases()
{
   bool ok = true;
   ok = Check(
      !PointMatrixDecodeResult< 2 >{}.HasValue(),
      "default 2D decode result is not successful" ) && ok;
   ok = Check(
      !PointMatrixDecodeResult< 3 >{}.HasValue(),
      "default 3D decode result is not successful" ) && ok;
   ok = Check(
      DecodeMFEMPointMatrix(
         4,
         1,
         TranslateMFEMFace2D( 1 ),
         TranslateMFEMFace2D( 1 ),
         SegmentPointMatrix{ { 0.5, 0.0 } } ).error ==
            PointMatrixDecodeError::InvalidCoarseFace,
      "2D invalid coarse face rejected" ) && ok;
   ok = Check(
      DecodeMFEMPointMatrix(
         1,
         4,
         TranslateMFEMFace2D( 1 ),
         TranslateMFEMFace2D( 1 ),
         SegmentPointMatrix{ { 0.5, 0.0 } } ).error ==
            PointMatrixDecodeError::InvalidFineFace,
      "2D invalid fine face rejected" ) && ok;
   ok = Check(
      DecodeMFEMPointMatrix(
         6,
         3,
         TranslateMFEMFace3D( 1 ),
         TranslateMFEMFace3D( 3 ),
         SquarePointMatrix{ {
            Point< 2 >{ 0.0, 0.0 },
            Point< 2 >{ 0.5, 0.0 },
            Point< 2 >{ 0.5, 0.5 },
            Point< 2 >{ 0.0, 0.5 }
         } } ).error == PointMatrixDecodeError::InvalidCoarseFace,
      "3D invalid coarse face rejected" ) && ok;
   ok = Check(
      DecodeMFEMPointMatrix(
         1,
         6,
         TranslateMFEMFace3D( 1 ),
         TranslateMFEMFace3D( 3 ),
         SquarePointMatrix{ {
            Point< 2 >{ 0.0, 0.0 },
            Point< 2 >{ 0.5, 0.0 },
            Point< 2 >{ 0.5, 0.5 },
            Point< 2 >{ 0.0, 0.5 }
         } } ).error == PointMatrixDecodeError::InvalidFineFace,
      "3D invalid fine face rejected" ) && ok;

   ok = CheckInvalid2D(
      "2D duplicate endpoints",
      1,
      1,
      { 0.0, 0.0 } ) && ok;
   ok = CheckValid2D(
      "2D reversed raw endpoint order is represented by orientation",
      1,
      1,
      { 0.0, 0.5 },
      MakePermutation< 2 >( { -1, -2 } ),
      Point< 2 >{ 0.0, 0.0 },
      { 1.0, 0.5 } ) && ok;
   ok = CheckInvalid3D(
      "3D min-max would accept unordered columns",
      1,
      3,
      { 0.0, 0.5, 0.0, 0.5,
        0.0, 0.0, 0.5, 0.5 } ) && ok;
   ok = CheckInvalid3D(
      "3D cross-axis coupling",
      1,
      3,
      { 0.0, 0.5, 0.5, 0.0,
        0.0, 0.25, 0.5, 0.5 } ) && ok;
   ok = CheckInvalid3D(
      "3D negative-scale nonpermutation",
      1,
      3,
      { 0.5, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.5, 0.5 } ) && ok;
   ok = CheckInvalid3D(
      "3D inconsistent fourth corner",
      1,
      3,
      { 0.0, 0.5, 0.75, 0.0,
        0.0, 0.0, 0.5, 0.5 } ) && ok;
   ok = CheckInvalid3D(
      "3D out-of-range child",
      1,
      3,
      { 0.75, 1.25, 1.25, 0.75,
        0.0, 0.0, 0.5, 0.5 } ) && ok;
   ok = CheckInvalid3D(
      "3D duplicate missing corner",
      1,
      3,
      { 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0 } ) && ok;
   return ok;
}

bool TestExhaustiveUniqueness()
{
   bool ok = true;
   const std::vector< std::pair< Real, Real > > intervals{
      { 0.0, 0.25 },
      { 0.0, 0.5 },
      { 0.25, 0.25 },
      { 0.25, 0.5 },
      { 0.5, 0.25 },
      { 0.5, 0.5 }
   };

   Integer cases2 = 0;
   for ( int coarse_lf = 0; coarse_lf < 4; ++coarse_lf )
   {
      for ( int fine_lf = 0; fine_lf < 4; ++fine_lf )
      {
         const Integer coarse_g = TranslateMFEMFace2D( coarse_lf );
         const Integer fine_g = TranslateMFEMFace2D( fine_lf );
         const Integer normal_axis = coarse_g % 2;
         for ( const auto & candidate : SignedPermutations< 2 >() )
         {
            if ( TestNativeFaceFromReferenceFace< 2 >(
                    HyperCube< 2 >::GetOppositeFaceIndex( coarse_g ), candidate ) != fine_g )
            {
               continue;
            }
            for ( const auto & interval : intervals )
            {
               Point< 2 > origin{ 0.0, 0.0 };
               std::array< Real, 2 > size{ 1.0, 1.0 };
               const Integer tangent_axis = 1 - normal_axis;
               origin[tangent_axis] = interval.first;
               size[tangent_axis] = interval.second;
               const auto expected_map =
                  MakeExpectedMap< 2 >( coarse_g, origin, size );
               const auto pm = EncodeSegmentPointMatrix(
                  coarse_lf,
                  fine_lf,
                  candidate,
                  expected_map );
               const auto decoded = Decode2( coarse_lf, fine_lf, pm );
               ++cases2;
               ok = Check( decoded.map_candidate_count == 1,
                           "2D exhaustive uniqueness candidate count" ) && ok;
               ok = CheckPermutation(
                  decoded.value.plus_orientation,
                  candidate,
                  "2D exhaustive uniqueness orientation" ) && ok;
            }
         }
      }
   }

   Integer cases3 = 0;
   for ( int coarse_lf = 0; coarse_lf < 6; ++coarse_lf )
   {
      for ( int fine_lf = 0; fine_lf < 6; ++fine_lf )
      {
         const Integer coarse_g = TranslateMFEMFace3D( coarse_lf );
         const Integer fine_g = TranslateMFEMFace3D( fine_lf );
         const Integer normal_axis = coarse_g % 3;
         for ( const auto & candidate : SignedPermutations< 3 >() )
         {
            if ( TestNativeFaceFromReferenceFace< 3 >(
                    HyperCube< 3 >::GetOppositeFaceIndex( coarse_g ), candidate ) != fine_g )
            {
               continue;
            }
            for ( const auto & ix : intervals )
            {
               for ( const auto & iz : intervals )
               {
                  Point< 3 > origin{ 0.0, 0.0, 0.0 };
                  std::array< Real, 3 > size{ 1.0, 1.0, 1.0 };
                  Integer tangent_index = 0;
                  for ( Integer d = 0; d < 3; ++d )
                  {
                     if ( d == normal_axis ) { continue; }
                     const auto & interval = tangent_index == 0 ? ix : iz;
                     origin[d] = interval.first;
                     size[d] = interval.second;
                     ++tangent_index;
                  }
                  const auto expected_map =
                     MakeExpectedMap< 3 >( coarse_g, origin, size );
                  const auto pm = EncodeSquarePointMatrix(
                     coarse_lf,
                     fine_lf,
                     candidate,
                     expected_map );
                  const auto decoded = Decode3( coarse_lf, fine_lf, pm );
                  ++cases3;
                  ok = Check( decoded.map_candidate_count == 1,
                              "3D exhaustive uniqueness candidate count" ) && ok;
                  ok = CheckPermutation(
                     decoded.value.plus_orientation,
                     candidate,
                     "3D exhaustive uniqueness orientation" ) && ok;
               }
            }
         }
      }
   }

   std::cout << "Exhaustive decoder uniqueness cases: 2D=" << cases2
             << " 3D=" << cases3 << "\n";
   return ok;
}

} // namespace

int main()
{
   bool ok = true;
   ok = TestLiteralValidCases() && ok;
   ok = TestInvalidCases() && ok;
   ok = TestExhaustiveUniqueness() && ok;
   std::cout << ( ok ? "PASS" : "FAIL" )
             << ": MFEM point-matrix pure decoder preflight\n";
   return ok ? 0 : 1;
}

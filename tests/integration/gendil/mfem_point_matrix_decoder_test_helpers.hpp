// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <gendil/gendil.hpp>
#include <gendil/Interfaces/MFEM/GlobalFaceConnectivity/NonconformingPointMatrix/nonconformingpointmatrix.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace gendil::tests::mfem_nc_preflight
{

using mfem_interface::detail::DecodedNonconformingFace;
using mfem_interface::detail::DecodeMFEMPointMatrix;
using mfem_interface::detail::PointMatrixDecodeError;
using mfem_interface::detail::PointMatrixDecodeResult;
using mfem_interface::detail::SegmentPointMatrix;
using mfem_interface::detail::SquarePointMatrix;

constexpr Real DecoderTolerance()
{
   return mfem_interface::detail::PointMatrixDecodeTolerance( 3 );
}

inline Integer TranslateMFEMFace2D( int mfem_face )
{
   constexpr Integer map[4] = { 1, 2, 3, 0 };
   return map[mfem_face];
}

inline Integer TranslateMFEMFace3D( int mfem_face )
{
   constexpr Integer map[6] = { 2, 1, 3, 4, 0, 5 };
   return map[mfem_face];
}

template < Integer Dim >
Permutation< Dim > MakePermutation(
   const std::array< LocalIndex, Dim > & entries )
{
   return mfem_interface::detail::MakePointMatrixPermutation( entries );
}

template < Integer Dim >
std::string PermutationString( const Permutation< Dim > & p )
{
   std::ostringstream os;
   os << "{";
   for ( Integer d = 0; d < Dim; ++d )
   {
      if ( d != 0 ) { os << ","; }
      os << p( d );
   }
   os << "}";
   return os.str();
}

template < Integer Dim >
std::string PointString( const Point< Dim > & p )
{
   std::ostringstream os;
   os << "(";
   for ( Integer d = 0; d < Dim; ++d )
   {
      if ( d != 0 ) { os << ","; }
      os << p[d];
   }
   os << ")";
   return os.str();
}

inline std::string DecodeErrorString( PointMatrixDecodeError error )
{
   switch ( error )
   {
      case PointMatrixDecodeError::None: return "None";
      case PointMatrixDecodeError::InvalidCoarseFace:
         return "InvalidCoarseFace";
      case PointMatrixDecodeError::InvalidFineFace:
         return "InvalidFineFace";
      case PointMatrixDecodeError::NoOrientationCandidate:
         return "NoOrientationCandidate";
      case PointMatrixDecodeError::MultipleOrientationCandidates:
         return "MultipleOrientationCandidates";
   }
   return "UnknownPointMatrixDecodeError";
}

template < Integer Dim >
bool Near( Real a, Real b, Real tol = DecoderTolerance() )
{
   return std::abs( a - b ) <= tol;
}

template < Integer Dim >
Real MaxError( const Point< Dim > & a, const Point< Dim > & b )
{
   Real err = 0.0;
   for ( Integer d = 0; d < Dim; ++d )
   {
      err = std::max( err, std::abs( a[d] - b[d] ) );
   }
   return err;
}

template < Integer Dim >
std::vector< Permutation< Dim > > SignedPermutations();

template <>
inline std::vector< Permutation< 2 > > SignedPermutations< 2 >()
{
   return {
      MakePermutation< 2 >( {  1,  2 } ),
      MakePermutation< 2 >( {  1, -2 } ),
      MakePermutation< 2 >( { -1,  2 } ),
      MakePermutation< 2 >( { -1, -2 } ),
      MakePermutation< 2 >( {  2,  1 } ),
      MakePermutation< 2 >( {  2, -1 } ),
      MakePermutation< 2 >( { -2,  1 } ),
      MakePermutation< 2 >( { -2, -1 } )
   };
}

template <>
inline std::vector< Permutation< 3 > > SignedPermutations< 3 >()
{
   std::vector< Permutation< 3 > > out;
   std::array< LocalIndex, 3 > axes{ 1, 2, 3 };
   do
   {
      for ( LocalIndex sx : { LocalIndex( -1 ), LocalIndex( 1 ) } )
      {
         for ( LocalIndex sy : { LocalIndex( -1 ), LocalIndex( 1 ) } )
         {
            for ( LocalIndex sz : { LocalIndex( -1 ), LocalIndex( 1 ) } )
            {
               out.push_back(
                  MakePermutation< 3 >(
                     { LocalIndex( sx * axes[0] ),
                       LocalIndex( sy * axes[1] ),
                       LocalIndex( sz * axes[2] ) } ) );
            }
         }
      }
   }
   while ( std::next_permutation( axes.begin(), axes.end() ) );
   return out;
}

template < Integer Dim >
Point< Dim > CanonicalFacePoint(
   Integer face,
   const std::array< Real, Dim - 1 > & tangential )
{
   return mfem_interface::detail::CanonicalFacePoint< Dim >( face, tangential );
}

template < Integer Dim >
Point< Dim > ApplyOrientation(
   const Point< Dim > & reference,
   const Permutation< Dim > & orientation )
{
   return mfem_interface::detail::ReferenceToNativePoint( reference, orientation );
}

template < Integer Dim >
Integer TestNativeFaceFromReferenceFace(
   Integer reference_face,
   const Permutation< Dim > & orientation )
{
   return static_cast< Integer >(
      mfem_interface::detail::PointMatrixNativeFaceFromReferenceFace(
         reference_face,
         orientation ) );
}

inline Point< 2 > MFEMFacePointToNative2D( int mfem_face, Real t )
{
   return mfem_interface::detail::MFEMQuadFacePointToNative(
      static_cast< Integer >( mfem_face ),
      t );
}

inline Real NativeToMFEMFacePoint2D(
   int mfem_face,
   const Point< 2 > & native )
{
   return mfem_interface::detail::NativeToMFEMQuadFacePoint(
      static_cast< Integer >( mfem_face ),
      native );
}

inline Point< 3 > MFEMFacePointToNative3D(
   int mfem_face,
   const std::array< Real, 2 > & uv )
{
   return mfem_interface::detail::MFEMHexFacePointToNative(
      static_cast< Integer >( mfem_face ),
      Point< 2 >{ uv[0], uv[1] } );
}

inline Point< 3 > MFEMFacePointToNative3D(
   int mfem_face,
   const Point< 2 > & uv )
{
   return mfem_interface::detail::MFEMHexFacePointToNative(
      static_cast< Integer >( mfem_face ),
      uv );
}

inline std::array< Real, 2 > NativeToMFEMFacePoint3D(
   int mfem_face,
   const Point< 3 > & native )
{
   const auto uv = mfem_interface::detail::NativeToMFEMHexFacePoint(
      static_cast< Integer >( mfem_face ),
      native );
   return { uv[0], uv[1] };
}

inline int SegmentPointMatrixColumnForSlaveCoordinate( Real t )
{
   if ( Near< 2 >( t, Real( 0 ) ) ) { return 1; }
   if ( Near< 2 >( t, Real( 1 ) ) ) { return 0; }
   return -1;
}

inline int SquarePointMatrixColumnForSlaveCoordinate(
   const std::array< Real, 2 > & uv )
{
   const bool u0 = Near< 3 >( uv[0], Real( 0 ) );
   const bool u1 = Near< 3 >( uv[0], Real( 1 ) );
   const bool v0 = Near< 3 >( uv[1], Real( 0 ) );
   const bool v1 = Near< 3 >( uv[1], Real( 1 ) );
   if ( u0 && v0 ) { return 0; }
   if ( u1 && v0 ) { return 1; }
   if ( u1 && v1 ) { return 2; }
   if ( u0 && v1 ) { return 3; }
   return -1;
}

template < Integer Dim >
Point< Dim > ApplyDiagonalMap(
   const NonconformingHyperCubeFaceMap< Dim > & map,
   const Point< Dim > & q_leaf )
{
   return mfem_interface::detail::ApplyDiagonalMap< Dim >( map, q_leaf );
}

inline bool IsSuccessfulDecode( const PointMatrixDecodeResult< 2 > & d )
{
   return d.HasValue() &&
          d.face_candidate_count > 0 &&
          d.corner_candidate_count > 0 &&
          d.map_candidate_count == 1;
}

inline bool IsSuccessfulDecode( const PointMatrixDecodeResult< 3 > & d )
{
   return d.HasValue() &&
          d.face_candidate_count > 0 &&
          d.corner_candidate_count > 0 &&
          d.map_candidate_count == 1;
}

} // namespace gendil::tests::mfem_nc_preflight

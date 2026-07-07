// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Candidate enumeration and positive diagonal map validation.
 *
 * Production decoding uses only copied MFEM point-matrix data. The decode path
 * is:
 *  - enumerate signed plus-orientation candidates;
 *  - keep candidates that recover the fine native face;
 *  - order raw point-matrix columns as canonical GenDiL leaf corners;
 *  - fit a positive diagonal NonconformingHyperCubeFaceMap on the coarse side;
 *  - validate corners and one interpolated interior point;
 *  - accept only a unique surviving candidate.
 *
 * The search is bounded by dimension (8 signed permutations in 2D and 48 in
 * 3D) and performs fixed corner/interior checks. It is intended for
 * construction-time decoding; repeated MFEM refinement requires rebuilding the
 * materialized connectivity and paying this decode cost again.
 */

#include <limits>

#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/NonconformingPointMatrix/nonconformingpointmatrixdecoder.hpp"

namespace gendil {
namespace mfem_interface {
namespace detail {

/**
 * @brief Reference-space tolerance used after discrete corner ordering.
 *
 * Column selection remains exact; this tolerance is only for floating-point
 * validation of fitted maps.
 */
constexpr Real PointMatrixDecodeTolerance( Integer dim )
{
   return Real( 256 ) * std::numeric_limits< Real >::epsilon() * Real( dim + 1 );
}

/**
 * @brief Build a signed permutation from candidate orientation entries.
 *
 * @details The candidate enumeration stores orientations as signed 1-based
 * axis entries, matching the convention used by GenDiL `Permutation`. Keeping
 * this helper local to the decoder makes the construction explicit at the
 * point where the finite signed-permutation search is performed.
 */
template < Integer Dim >
constexpr Permutation< Dim > MakePointMatrixPermutation(
   const std::array< LocalIndex, Dim > & entries )
{
   return Permutation< Dim >{ entries };
}

/**
 * @brief Apply the fitted coarse-side map to a full-D leaf point.
 *
 * @details `NonconformingHyperCubeFaceMap` is the runtime representation we
 * are trying to materialize from MFEM's point matrix. The leaf point already
 * lies on the canonical coarse/minus face, including the fixed normal
 * coordinate. The map therefore acts on full-D coordinates component-wise.
 */
template < Integer Dim >
Point< Dim > ApplyDiagonalMap(
   const NonconformingHyperCubeFaceMap< Dim > & map,
   const Point< Dim > & leaf )
{
   Point< Dim > parent;
   for ( Integer d = 0; d < Dim; ++d )
   {
      parent[d] = map.origin[d] + map.size[d] * leaf[d];
   }
   return parent;
}

/**
 * @brief Determine the native face produced by applying an orientation.
 *
 * @details A nonsymmetric interior point on `reference_face` is enough to
 * identify the output face after the signed permutation is applied: exactly
 * one coordinate remains fixed to 0 or 1. This is used as the first cheap
 * candidate filter before inspecting point-matrix columns.
 */
template < Integer Dim >
int PointMatrixNativeFaceFromReferenceFace(
   Integer reference_face,
   const Permutation< Dim > & orientation )
{
   std::array< Real, Dim - 1 > interior{};
   for ( Integer i = 0; i < Dim - 1; ++i )
   {
      interior[i] = Real( 0.37 );
   }
   const auto reference = CanonicalFacePoint< Dim >( reference_face, interior );
   const auto native = ReferenceToNativePoint( reference, orientation );
   for ( Integer d = 0; d < Dim; ++d )
   {
      if ( native[d] == Real( 0 ) ) { return static_cast< int >( d ); }
      if ( native[d] == Real( 1 ) ) { return static_cast< int >( d + Dim ); }
   }
   return -1;
}

/**
 * @brief Test whether a candidate maps the static plus face to the fine face.
 *
 * @details GenDiL families use a static plus face equal to the opposite of the
 * coarse canonical minus face. The candidate orientation is responsible for
 * recovering the actual MFEM fine native face. Candidates that cannot do that
 * are rejected before any corner or map fitting work.
 */
template < Integer Dim >
bool CandidateMapsToFineFace(
   Integer static_plus,
   const Permutation< Dim > & candidate,
   Integer fine_g1 )
{
   return PointMatrixNativeFaceFromReferenceFace( static_plus, candidate ) ==
          static_cast< int >( fine_g1 );
}

/**
 * @brief Ordered endpoint correspondence for a 2D leaf-edge candidate.
 *
 * @details `leaf_points` are canonical GenDiL coordinates on the coarse face
 * at leaf endpoints t=0 and t=1. `coarse_targets` are the corresponding
 * full-D coarse native coordinates obtained by reading the MFEM point matrix
 * columns selected by the candidate orientation.
 */
struct OrderedSegmentCorners
{
   std::array< Point< 2 >, 2 > coarse_targets{};
   std::array< Point< 2 >, 2 > leaf_points{};
};

/**
 * @brief Ordered corner correspondence for a 3D leaf-face candidate.
 *
 * Corners are stored in GenDiL leaf order 00,10,01,11; raw MFEM square columns
 * are read in their documented 00,10,11,01 order.
 *
 * @details This structure is the bridge between two orderings. It lets map
 * fitting operate in GenDiL canonical leaf order while preserving the raw MFEM
 * column convention used to obtain each coarse target point.
 */
struct OrderedSquareCorners
{
   std::array< Point< 3 >, 4 > coarse_targets{};
   std::array< Point< 3 >, 4 > leaf_points{};
};

/**
 * @brief Build ordered 2D endpoint data for one orientation candidate.
 *
 * @details Each canonical plus endpoint is pushed through the candidate
 * orientation into the fine native frame. The fine MFEM local-face convention
 * identifies which slave endpoint was reached, and the segment column helper
 * selects the raw point-matrix column for that endpoint. Failure means the
 * candidate does not induce a valid endpoint correspondence on this fine face.
 */
inline bool OrderSegmentCandidateCorners(
   Integer coarse_lf,
   Integer fine_lf,
   Integer coarse_g0,
   Integer static_plus,
   const Permutation< 2 > & candidate,
   const SegmentPointMatrix & pm,
   OrderedSegmentCorners & ordered )
{
   constexpr std::array< std::array< Real, 1 >, 2 > corners{ {
      { Real( 0 ) }, { Real( 1 ) }
   } };

   for ( Integer c = 0; c < 2; ++c )
   {
      const auto plus_canonical = CanonicalFacePoint< 2 >( static_plus, corners[c] );
      const auto fine_native = ReferenceToNativePoint( plus_canonical, candidate );
      Integer endpoint = 0;
      if ( !MFEMQuadNativeCornerEndpoint( fine_lf, fine_native, endpoint ) )
      {
         return false;
      }
      const Integer column = SegmentPointMatrixColumn( endpoint );
      ordered.coarse_targets[c] =
         MFEMQuadFacePointToNative( coarse_lf, pm.columns[column] );
      ordered.leaf_points[c] = CanonicalFacePoint< 2 >( coarse_g0, corners[c] );
   }
   return true;
}

/**
 * @brief Build ordered 3D square-corner data for one orientation candidate.
 *
 * @details The same idea as the segment case, but for square face corners.
 * Candidate orientation chooses a fine native corner, MFEM face-coordinate
 * tables recover the discrete `(u,v)` corner, and the raw square column helper
 * selects the corresponding point-matrix column. All stored targets are
 * converted to full-D coarse native coordinates before fitting.
 */
inline bool OrderSquareCandidateCorners(
   Integer coarse_lf,
   Integer fine_lf,
   Integer coarse_g0,
   Integer static_plus,
   const Permutation< 3 > & candidate,
   const SquarePointMatrix & pm,
   OrderedSquareCorners & ordered )
{
   constexpr std::array< std::array< Real, 2 >, 4 > corners{ {
      { Real( 0 ), Real( 0 ) },
      { Real( 1 ), Real( 0 ) },
      { Real( 0 ), Real( 1 ) },
      { Real( 1 ), Real( 1 ) }
   } };

   for ( Integer c = 0; c < 4; ++c )
   {
      const auto plus_canonical = CanonicalFacePoint< 3 >( static_plus, corners[c] );
      const auto fine_native = ReferenceToNativePoint( plus_canonical, candidate );
      Integer u = 0;
      Integer v = 0;
      if ( !MFEMHexNativeCornerPoint( fine_lf, fine_native, u, v ) )
      {
         return false;
      }
      const int column = SquarePointMatrixColumn( u, v );
      if ( column < 0 )
      {
         return false;
      }
      ordered.coarse_targets[c] = MFEMHexFacePointToNative(
         coarse_lf,
         pm.columns[static_cast< Integer >( column )] );
      ordered.leaf_points[c] = CanonicalFacePoint< 3 >( coarse_g0, corners[c] );
   }
   return true;
}

/**
 * @brief Evaluate the raw MFEM segment point matrix.
 *
 * @details The 2D local-slave point-matrix convention is reversed with
 * respect to the slave endpoint parameter: t=0 reads raw column 1 and t=1
 * reads raw column 0. The interior check uses this helper so it validates
 * against the original MFEM data rather than against the fitted map.
 */
inline Real InterpolateSegmentPointMatrix(
   const SegmentPointMatrix & pm,
   Real t )
{
   return ( Real( 1 ) - t ) * pm.columns[1] + t * pm.columns[0];
}

/**
 * @brief Evaluate the raw MFEM square point matrix in 00,10,11,01 order.
 *
 * @details MFEM stores square point-matrix columns in raw order 00,10,11,01.
 * The decoder uses this bilinear interpolation at a nonsymmetric point to
 * catch ordering mistakes and maps that would fit the corners only by accident.
 */
inline Point< 2 > InterpolateSquarePointMatrix(
   const SquarePointMatrix & pm,
   const Point< 2 > & uv )
{
   const Real u = uv[0];
   const Real v = uv[1];
   return Point< 2 >{
      ( Real( 1 ) - u ) * ( Real( 1 ) - v ) * pm.columns[0][0] +
      u * ( Real( 1 ) - v ) * pm.columns[1][0] +
      u * v * pm.columns[2][0] +
      ( Real( 1 ) - u ) * v * pm.columns[3][0],
      ( Real( 1 ) - u ) * ( Real( 1 ) - v ) * pm.columns[0][1] +
      u * ( Real( 1 ) - v ) * pm.columns[1][1] +
      u * v * pm.columns[2][1] +
      ( Real( 1 ) - u ) * v * pm.columns[3][1] };
}

/**
 * @brief Fit and validate the positive diagonal map for a segment leaf.
 *
 * @details The two ordered endpoints define the tangential origin and size on
 * the coarse face; the normal axis is identity. The candidate survives only if
 * the tangential size is positive, the image stays inside the parent face, both
 * endpoints are reproduced, and one interior point agrees with direct
 * interpolation of the raw segment point matrix.
 */
inline bool FitAndValidateSegmentMap(
   Integer coarse_lf,
   Integer fine_lf,
   Integer coarse_g0,
   Integer static_plus,
   const Permutation< 2 > & candidate,
   const SegmentPointMatrix & pm,
   const OrderedSegmentCorners & ordered,
   Real tol,
   NonconformingHyperCubeFaceMap< 2 > & map )
{
   const Integer normal_axis = coarse_g0 % 2;
   const Integer tangent_axis = 1 - normal_axis;
   map.origin[normal_axis] = Real( 0 );
   map.size[normal_axis] = Real( 1 );
   map.origin[tangent_axis] = ordered.coarse_targets[0][tangent_axis];
   map.size[tangent_axis] =
      ordered.coarse_targets[1][tangent_axis] -
      ordered.coarse_targets[0][tangent_axis];

   bool map_ok =
      map.size[tangent_axis] > tol &&
      map.origin[tangent_axis] >= -tol &&
      map.origin[tangent_axis] + map.size[tangent_axis] <= Real( 1 ) + tol;

   for ( Integer c = 0; c < 2; ++c )
   {
      map_ok = map_ok &&
         Distance(
            ApplyDiagonalMap< 2 >( map, ordered.leaf_points[c] ),
            ordered.coarse_targets[c] ) <= tol;
   }

   const std::array< Real, 1 > interior{ Real( 0.37 ) };
   const auto plus_canonical = CanonicalFacePoint< 2 >( static_plus, interior );
   const auto fine_native = ReferenceToNativePoint( plus_canonical, candidate );
   const Real fine_t = NativeToMFEMQuadFacePoint( fine_lf, fine_native );
   const Real coarse_t = InterpolateSegmentPointMatrix( pm, fine_t );
   const auto target_interior = MFEMQuadFacePointToNative( coarse_lf, coarse_t );
   const auto leaf_interior = CanonicalFacePoint< 2 >( coarse_g0, interior );
   return map_ok &&
      Distance(
         ApplyDiagonalMap< 2 >( map, leaf_interior ),
         target_interior ) <= tol;
}

/**
 * @brief Return the two full-D coordinate axes tangent to a 3D face.
 *
 * @details Fitting is done in full-D coarse native coordinates. The tangent
 * axes are the two coordinates not fixed by the canonical coarse face normal.
 */
inline std::array< Integer, 2 > GetSquareMapTangentAxes( Integer normal_axis )
{
   std::array< Integer, 2 > tangent_axes{};
   Integer tangent_count = 0;
   for ( Integer d = 0; d < 3; ++d )
   {
      if ( d != normal_axis )
      {
         tangent_axes[tangent_count++] = d;
      }
   }
   return tangent_axes;
}

/**
 * @brief Symmetric scalar reference-coordinate comparison.
 *
 * @details Used for axis-coupling checks where the code wants an explicit
 * scalar predicate and avoids pulling in additional math utilities.
 */
inline bool CoordinatesMatch( Real a, Real b, Real tol )
{
   return ( a - b <= tol ) && ( b - a <= tol );
}

/**
 * @brief Fit the 3D diagonal map from ordered 00,10,01 corners.
 *
 * @details Corner 00 determines the tangential origin. Corners 10 and 01
 * determine the two positive diagonal sizes. The normal component remains the
 * identity map because the leaf point already lies on the static coarse face.
 * This function only computes a candidate map; later checks validate it.
 */
inline std::array< Integer, 2 > FitSquareDiagonalMapFromCorners(
   Integer coarse_g0,
   const OrderedSquareCorners & ordered,
   NonconformingHyperCubeFaceMap< 3 > & map )
{
   const Integer normal_axis = coarse_g0 % 3;
   map.origin[normal_axis] = Real( 0 );
   map.size[normal_axis] = Real( 1 );

   const auto tangent_axes = GetSquareMapTangentAxes( normal_axis );
   const Integer a = tangent_axes[0];
   const Integer b = tangent_axes[1];
   map.origin[a] = ordered.coarse_targets[0][a];
   map.origin[b] = ordered.coarse_targets[0][b];
   map.size[a] = ordered.coarse_targets[1][a] - ordered.coarse_targets[0][a];
   map.size[b] = ordered.coarse_targets[2][b] - ordered.coarse_targets[0][b];
   return tangent_axes;
}

/**
 * @brief Check positive tangential sizes and parent-side range.
 *
 * @details Reflections and tangent-axis permutations must be represented by
 * `plus_orientation`, not by negative map sizes. A valid GenDiL map therefore
 * has strictly positive tangential sizes and remains in `[0,1]` on the coarse
 * reference face.
 */
inline bool SquareMapHasPositiveTangentialRange(
   const NonconformingHyperCubeFaceMap< 3 > & map,
   const std::array< Integer, 2 > & tangent_axes,
   Real tol )
{
   const Integer a = tangent_axes[0];
   const Integer b = tangent_axes[1];
   return
      map.size[a] > tol &&
      map.size[b] > tol &&
      map.origin[a] >= -tol &&
      map.origin[b] >= -tol &&
      map.origin[a] + map.size[a] <= Real( 1 ) + tol &&
      map.origin[b] + map.size[b] <= Real( 1 ) + tol;
}

/**
 * @brief Reject cross-axis coupling and changed normal coordinates.
 *
 * @details `NonconformingHyperCubeFaceMap` stores a diagonal map only. Moving
 * canonical leaf tangent axis 0 may change only the first coarse tangent axis,
 * and moving tangent axis 1 may change only the second. The normal coordinate
 * must stay on the same coarse face.
 */
inline bool SquareMapHasNoTangentialCoupling(
   const OrderedSquareCorners & ordered,
   Integer normal_axis,
   const std::array< Integer, 2 > & tangent_axes,
   Real tol )
{
   const Integer a = tangent_axes[0];
   const Integer b = tangent_axes[1];
   return
      CoordinatesMatch( ordered.coarse_targets[1][b],
                        ordered.coarse_targets[0][b],
                        tol ) &&
      CoordinatesMatch( ordered.coarse_targets[2][a],
                        ordered.coarse_targets[0][a],
                        tol ) &&
      CoordinatesMatch( ordered.coarse_targets[1][normal_axis],
                        ordered.leaf_points[1][normal_axis],
                        tol ) &&
      CoordinatesMatch( ordered.coarse_targets[2][normal_axis],
                        ordered.leaf_points[2][normal_axis],
                        tol );
}

/**
 * @brief Check that the fitted map reproduces every ordered square corner.
 *
 * @details This validates the fourth corner as well as the three corners used
 * for fitting. It is the corner-level proof that the selected orientation,
 * raw column ordering, and diagonal map all describe the same parameterization.
 */
inline bool SquareMapReproducesOrderedCorners(
   const NonconformingHyperCubeFaceMap< 3 > & map,
   const OrderedSquareCorners & ordered,
   Real tol )
{
   for ( Integer c = 0; c < 4; ++c )
   {
      if ( Distance(
              ApplyDiagonalMap< 3 >( map, ordered.leaf_points[c] ),
              ordered.coarse_targets[c] ) > tol )
      {
         return false;
      }
   }
   return true;
}

/**
 * @brief Check the fitted map against one raw-matrix interior interpolation.
 *
 * @details The interior target is computed directly from the MFEM point
 * matrix, using the fine-face coordinates induced by the candidate orientation.
 * This avoids validating the fitted map against values generated from itself.
 */
inline bool SquareMapReproducesInterpolatedInterior(
   Integer coarse_lf,
   Integer fine_lf,
   Integer coarse_g0,
   Integer static_plus,
   const Permutation< 3 > & candidate,
   const SquarePointMatrix & pm,
   const NonconformingHyperCubeFaceMap< 3 > & map,
   Real tol )
{
   const std::array< Real, 2 > interior{ Real( 0.23 ), Real( 0.67 ) };
   const auto plus_canonical = CanonicalFacePoint< 3 >( static_plus, interior );
   const auto fine_native = ReferenceToNativePoint( plus_canonical, candidate );
   const auto fine_uv = NativeToMFEMHexFacePoint( fine_lf, fine_native );
   const auto coarse_uv = InterpolateSquarePointMatrix( pm, fine_uv );
   const auto target_interior = MFEMHexFacePointToNative( coarse_lf, coarse_uv );
   const auto leaf_interior = CanonicalFacePoint< 3 >( coarse_g0, interior );
   return Distance(
      ApplyDiagonalMap< 3 >( map, leaf_interior ),
      target_interior ) <= tol;
}

/**
 * @brief Fit and validate the positive diagonal map for a square leaf.
 *
 * @details This is the final 3D map stage for one orientation candidate. The
 * checks are deliberately ordered from cheap structural checks to the direct
 * point-matrix interpolation check: positive range, no tangential coupling,
 * corner reproduction, then one nonsymmetric interior sample.
 */
inline bool FitAndValidateSquareMap(
   Integer coarse_lf,
   Integer fine_lf,
   Integer coarse_g0,
   Integer static_plus,
   const Permutation< 3 > & candidate,
   const SquarePointMatrix & pm,
   const OrderedSquareCorners & ordered,
   Real tol,
   NonconformingHyperCubeFaceMap< 3 > & map )
{
   const Integer normal_axis = coarse_g0 % 3;
   const auto tangent_axes =
      FitSquareDiagonalMapFromCorners( coarse_g0, ordered, map );

   if ( !SquareMapHasPositiveTangentialRange( map, tangent_axes, tol ) )
   {
      return false;
   }
   if ( !SquareMapHasNoTangentialCoupling(
           ordered,
           normal_axis,
           tangent_axes,
           tol ) )
   {
      return false;
   }
   if ( !SquareMapReproducesOrderedCorners( map, ordered, tol ) )
   {
      return false;
   }
   return SquareMapReproducesInterpolatedInterior(
      coarse_lf,
      fine_lf,
      coarse_g0,
      static_plus,
      candidate,
      pm,
      map,
      tol );
}

/**
 * @brief Store one successfully decoded candidate in the result object.
 *
 * @details The decoder does not resolve competing successful candidates here.
 * It records the candidate count and keeps the last value; `FinalDecodeError`
 * reports multiple accepted candidates as an error.
 */
template < Integer Dim >
void AcceptDecodedCandidate(
   PointMatrixDecodeResult< Dim > & result,
   const auto & plus_orientation,
   const auto & minus_map )
{
   ++result.map_candidate_count;
   result.value.plus_orientation = plus_orientation;
   result.value.minus_map = minus_map;
}

/**
 * @brief Convert candidate counts into the public decode status.
 *
 * @details A successful decode requires exactly one map candidate. A zero
 * count is reported as `NoOrientationCandidate`; callers can inspect the face
 * and corner counters to see which earlier stage eliminated the candidates.
 */
inline PointMatrixDecodeError FinalDecodeError(
   Integer face_candidate_count,
   Integer map_candidate_count )
{
   if ( map_candidate_count == 1 )
   {
      return PointMatrixDecodeError::None;
   }
   if ( map_candidate_count > 1 )
   {
      return PointMatrixDecodeError::MultipleOrientationCandidates;
   }
   (void) face_candidate_count;
   return PointMatrixDecodeError::NoOrientationCandidate;
}

/**
 * @brief Decode a 2D leaf-edge point matrix by endpoint correspondence.
 *
 * @details This overload enumerates the eight signed 2D permutations. Each
 * candidate must recover the fine face, produce a complete endpoint ordering,
 * and fit a positive in-range segment map. The counters in the result record
 * how many candidates reached each stage.
 */
inline PointMatrixDecodeResult< 2 > DecodeMFEMPointMatrix(
   Integer coarse_lf,
   Integer fine_lf,
   Integer coarse_g0,
   Integer fine_g1,
   SegmentPointMatrix pm )
{
   PointMatrixDecodeResult< 2 > result;
   if ( !IsValidQuadFace( coarse_lf ) || !IsValidGenDiLFace( 2, coarse_g0 ) )
   {
      result.error = PointMatrixDecodeError::InvalidCoarseFace;
      return result;
   }
   if ( !IsValidQuadFace( fine_lf ) || !IsValidGenDiLFace( 2, fine_g1 ) )
   {
      result.error = PointMatrixDecodeError::InvalidFineFace;
      return result;
   }

   constexpr std::array< std::array< LocalIndex, 2 >, 8 > signed_permutations{ {
      {  1,  2 }, {  1, -2 }, { -1,  2 }, { -1, -2 },
      {  2,  1 }, {  2, -1 }, { -2,  1 }, { -2, -1 }
   } };

   const Integer static_plus = HyperCube< 2 >::GetOppositeFaceIndex( coarse_g0 );
   const Real tol = PointMatrixDecodeTolerance( 2 );

   for ( const auto & permutation_entries : signed_permutations )
   {
      const auto candidate = MakePointMatrixPermutation( permutation_entries );
      if ( !CandidateMapsToFineFace( static_plus, candidate, fine_g1 ) )
      {
         continue;
      }
      ++result.face_candidate_count;

      OrderedSegmentCorners ordered;
      if ( !OrderSegmentCandidateCorners(
              coarse_lf,
              fine_lf,
              coarse_g0,
              static_plus,
              candidate,
              pm,
              ordered ) )
      {
         continue;
      }
      ++result.corner_candidate_count;

      NonconformingHyperCubeFaceMap< 2 > map;
      if ( FitAndValidateSegmentMap(
              coarse_lf,
              fine_lf,
              coarse_g0,
              static_plus,
              candidate,
              pm,
              ordered,
              tol,
              map ) )
      {
         AcceptDecodedCandidate( result, candidate, map );
      }
   }

   result.error = FinalDecodeError(
      result.face_candidate_count,
      result.map_candidate_count );
   return result;
}

/**
 * @brief Decode a 3D leaf-face point matrix by square-corner correspondence.
 *
 * @details This overload enumerates the 48 signed 3D permutations. Each
 * candidate must recover the fine face, produce a complete square-corner
 * ordering, and fit a positive diagonal square map with no coupling. The
 * counters in the result record how many candidates reached each stage.
 */
inline PointMatrixDecodeResult< 3 > DecodeMFEMPointMatrix(
   Integer coarse_lf,
   Integer fine_lf,
   Integer coarse_g0,
   Integer fine_g1,
   SquarePointMatrix pm )
{
   PointMatrixDecodeResult< 3 > result;
   if ( !IsValidHexFace( coarse_lf ) || !IsValidGenDiLFace( 3, coarse_g0 ) )
   {
      result.error = PointMatrixDecodeError::InvalidCoarseFace;
      return result;
   }
   if ( !IsValidHexFace( fine_lf ) || !IsValidGenDiLFace( 3, fine_g1 ) )
   {
      result.error = PointMatrixDecodeError::InvalidFineFace;
      return result;
   }

   constexpr std::array< std::array< LocalIndex, 3 >, 6 > axis_orders{ {
      { 1, 2, 3 }, { 1, 3, 2 }, { 2, 1, 3 },
      { 2, 3, 1 }, { 3, 1, 2 }, { 3, 2, 1 }
   } };

   const Integer static_plus = HyperCube< 3 >::GetOppositeFaceIndex( coarse_g0 );
   const Real tol = PointMatrixDecodeTolerance( 3 );

   for ( const auto & axes : axis_orders )
   {
      for ( LocalIndex sx : { LocalIndex( -1 ), LocalIndex( 1 ) } )
      {
         for ( LocalIndex sy : { LocalIndex( -1 ), LocalIndex( 1 ) } )
         {
            for ( LocalIndex sz : { LocalIndex( -1 ), LocalIndex( 1 ) } )
            {
               const std::array permutation_entries{
                  LocalIndex( sx * axes[0] ),
                  LocalIndex( sy * axes[1] ),
                  LocalIndex( sz * axes[2] ) };
               const auto candidate = MakePointMatrixPermutation( permutation_entries );
               if ( !CandidateMapsToFineFace( static_plus, candidate, fine_g1 ) )
               {
                  continue;
               }
               ++result.face_candidate_count;

               OrderedSquareCorners ordered;
               if ( !OrderSquareCandidateCorners(
                       coarse_lf,
                       fine_lf,
                       coarse_g0,
                       static_plus,
                       candidate,
                       pm,
                       ordered ) )
               {
                  continue;
               }
               ++result.corner_candidate_count;

               NonconformingHyperCubeFaceMap< 3 > map;
               if ( FitAndValidateSquareMap(
                       coarse_lf,
                       fine_lf,
                       coarse_g0,
                       static_plus,
                       candidate,
                       pm,
                       ordered,
                       tol,
                       map ) )
               {
                  AcceptDecodedCandidate( result, candidate, map );
               }
            }
         }
      }
   }

   result.error = FinalDecodeError(
      result.face_candidate_count,
      result.map_candidate_count );
   return result;
}

} // namespace detail
} // namespace mfem_interface
} // namespace gendil

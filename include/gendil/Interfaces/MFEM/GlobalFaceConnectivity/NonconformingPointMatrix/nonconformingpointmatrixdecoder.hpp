// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Copied point-matrix input types and decoder result declarations.
 *
 * This facade is internal to the GenDiL MFEM interface. It includes no MFEM
 * library headers, retains no MFEM pointers, and only declares the
 * allocation-free hypercube decode entrypoints implemented by the algorithm
 * subheader.
 */

#include <array>
#include <type_traits>

#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/NonconformingPointMatrix/nonconformingpointmatrixcoordinates.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"

namespace gendil {
namespace mfem_interface {
namespace detail {

/// Copied raw MFEM segment point matrix for a 2D nonconforming leaf edge.
struct SegmentPointMatrix
{
   // Raw MFEM columns 0 and 1. Each column stores one master-edge coordinate.
   std::array< Real, 2 > columns{};
};

/// Copied raw MFEM quadrilateral point matrix for a 3D nonconforming leaf face.
struct SquarePointMatrix
{
   // Raw MFEM column order: (0,0), (1,0), (1,1), (0,1).
   std::array< Point< 2 >, 4 > columns{};
};

static_assert( std::is_trivially_copyable_v< SegmentPointMatrix > );
static_assert( std::is_trivially_copyable_v< SquarePointMatrix > );

/// Error category reported after bounded orientation/map candidate search.
enum class PointMatrixDecodeError
{
   None,
   InvalidCoarseFace,
   InvalidFineFace,
   NoOrientationCandidate,
   MultipleOrientationCandidates
};

/// Successful decoder payload used to materialize a GenDiL NC face record.
template < Integer Dim >
struct DecodedNonconformingFace
{
   Permutation< Dim > plus_orientation = MakeReferencePermutation< Dim >();
   NonconformingHyperCubeFaceMap< Dim > minus_map{};
};

/// Decoder result plus candidate counters used by diagnostics and tests.
template < Integer Dim >
struct PointMatrixDecodeResult
{
   PointMatrixDecodeError error = PointMatrixDecodeError::NoOrientationCandidate;
   /// Number of signed permutations whose static plus face recovers the fine face.
   Integer face_candidate_count = 0;
   /// Number of face candidates whose ordered corners match point-matrix columns.
   Integer corner_candidate_count = 0;
   /// Number of corner candidates whose fitted positive diagonal map validates.
   Integer map_candidate_count = 0;
   DecodedNonconformingFace< Dim > value{};

   /// True only when exactly one positive axis-aligned map candidate survived.
   constexpr bool HasValue() const
   {
      return error == PointMatrixDecodeError::None && map_candidate_count == 1;
   }
};

/// Decode a copied MFEM 2D leaf-edge point matrix into orientation and minus map.
inline PointMatrixDecodeResult< 2 > DecodeMFEMPointMatrix(
   Integer coarse_lf, Integer fine_lf,
   Integer coarse_g0, Integer fine_g1,
   SegmentPointMatrix pm);

/// Decode a copied MFEM 3D leaf-face point matrix into orientation and minus map.
inline PointMatrixDecodeResult< 3 > DecodeMFEMPointMatrix(
   Integer coarse_lf, Integer fine_lf,
   Integer coarse_g0, Integer fine_g1,
   SquarePointMatrix pm);

} // namespace detail
} // namespace mfem_interface
} // namespace gendil

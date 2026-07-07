// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief MFEM face-reference coordinate and point-matrix column conventions.
 *
 * These helpers convert between MFEM face coordinates, GenDiL full-D
 * reference-cell coordinates, and oriented native coordinates used by the
 * internal point-matrix decoder. Candidate enumeration and map fitting live in
 * the decode algorithm header. These helpers do not access MFEM runtime
 * objects.
 */

#include <array>

#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Meshes/Geometries/point.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"

namespace gendil {
namespace mfem_interface {
namespace detail {

/**
 * @brief Embed canonical face coordinates in a full-D hypercube point.
 *
 * The normal coordinate is fixed to 0 for faces `[0, Dim)` and to 1 for faces
 * `[Dim, 2*Dim)`. Tangential coordinates are assigned to the remaining axes in
 * increasing axis order.
 *
 * @param[in] face GenDiL hypercube face id.
 * @param[in] tangential Coordinates on the canonical face.
 * @return Full-D reference-cell point on @p face.
 */
template < Integer Dim >
Point< Dim > CanonicalFacePoint(
   Integer face,
   const std::array< Real, Dim - 1 > & tangential )
{
   Point< Dim > point;
   const Integer normal_axis = HyperCube< Dim >::GetNormalDimensionIndex( face );
   Integer tangential_axis = 0;
   for ( Integer d = 0; d < Dim; ++d )
   {
      if ( d == normal_axis )
      {
         point[d] = face < Dim ? Real( 0 ) : Real( 1 );
      }
      else
      {
         point[d] = tangential[tangential_axis++];
      }
   }
   return point;
}

/**
 * @brief Apply a signed permutation to a continuous reference point.
 *
 * Each orientation entry is a signed 1-based reference axis. A positive entry
 * copies that reference coordinate; a negative entry reflects it as `1 - x`.
 *
 * @param[in] reference Full-D reference point in the canonical frame.
 * @param[in] orientation Signed permutation mapping reference to native axes.
 * @return Full-D native point after applying @p orientation.
 */
template < Integer Dim >
Point< Dim > ReferenceToNativePoint(
   const Point< Dim > & reference,
   const Permutation< Dim > & orientation )
{
   Point< Dim > native;
   for ( Integer native_axis = 0; native_axis < Dim; ++native_axis )
   {
      const LocalIndex entry = orientation( native_axis );
      const Integer reference_axis =
         static_cast< Integer >( entry > 0 ? entry - 1 : -entry - 1 );
      native[native_axis] =
         entry > 0 ? reference[reference_axis] : Real( 1 ) - reference[reference_axis];
   }
   return native;
}

/**
 * @brief Check the MFEM quadrilateral local-face id range.
 *
 * @param[in] face MFEM quad local face id.
 * @return True for local faces 0, 1, 2, and 3.
 */
constexpr bool IsValidQuadFace( Integer face )
{
   return face < 4;
}

/**
 * @brief Check the MFEM hexahedron local-face id range.
 *
 * @param[in] face MFEM hex local face id.
 * @return True for local faces 0 through 5.
 */
constexpr bool IsValidHexFace( Integer face )
{
   return face < 6;
}

/**
 * @brief Check the translated GenDiL hypercube face-id range.
 *
 * @param[in] dim Hypercube dimension.
 * @param[in] face Translated GenDiL local face id.
 * @return True when @p face is in `[0, 2*dim)`.
 */
constexpr bool IsValidGenDiLFace( Integer dim, Integer face )
{
   return face < 2 * dim;
}

/**
 * @brief Embed an MFEM quad edge coordinate in native element coordinates.
 *
 * MFEM quad face convention:
 *  - face 0: `(t, 0)`
 *  - face 1: `(1, t)`
 *  - face 2: `(1 - t, 1)`
 *  - face 3: `(0, 1 - t)`
 *
 * @param[in] mfem_local_face MFEM quad local face id.
 * @param[in] t MFEM edge coordinate.
 * @return Full-D native quad reference coordinate.
 */
inline Point< 2 > MFEMQuadFacePointToNative( Integer mfem_local_face, Real t )
{
   switch ( mfem_local_face )
   {
      case 0: return Point< 2 >{ t, Real( 0 ) };
      case 1: return Point< 2 >{ Real( 1 ), t };
      case 2: return Point< 2 >{ Real( 1 ) - t, Real( 1 ) };
      case 3: return Point< 2 >{ Real( 0 ), Real( 1 ) - t };
      default: return Point< 2 >{};
   }
}

/**
 * @brief Recover the MFEM quad edge coordinate from a native face point.
 *
 * This is the inverse of `MFEMQuadFacePointToNative` on a valid native face
 * point.
 *
 * @param[in] mfem_local_face MFEM quad local face id.
 * @param[in] native Full-D native quad reference coordinate.
 * @return MFEM edge coordinate.
 */
inline Real NativeToMFEMQuadFacePoint(
   Integer mfem_local_face,
   const Point< 2 > & native )
{
   switch ( mfem_local_face )
   {
      case 0: return native[0];
      case 1: return native[1];
      case 2: return Real( 1 ) - native[0];
      case 3: return Real( 1 ) - native[1];
      default: return Real( 0 );
   }
}

/**
 * @brief Embed MFEM hex face coordinates in native element coordinates.
 *
 * MFEM hex face convention:
 *  - face 0: `(u, 1 - v, 0)`
 *  - face 1: `(u, 0, v)`
 *  - face 2: `(1, u, v)`
 *  - face 3: `(1 - u, 1, v)`
 *  - face 4: `(0, 1 - u, v)`
 *  - face 5: `(u, v, 1)`
 *
 * @param[in] mfem_local_face MFEM hex local face id.
 * @param[in] uv MFEM square face coordinates.
 * @return Full-D native hex reference coordinate.
 */
inline Point< 3 > MFEMHexFacePointToNative(
   Integer mfem_local_face,
   const Point< 2 > & uv )
{
   const Real u = uv[0];
   const Real v = uv[1];
   switch ( mfem_local_face )
   {
      case 0: return Point< 3 >{ u, Real( 1 ) - v, Real( 0 ) };
      case 1: return Point< 3 >{ u, Real( 0 ), v };
      case 2: return Point< 3 >{ Real( 1 ), u, v };
      case 3: return Point< 3 >{ Real( 1 ) - u, Real( 1 ), v };
      case 4: return Point< 3 >{ Real( 0 ), Real( 1 ) - u, v };
      case 5: return Point< 3 >{ u, v, Real( 1 ) };
      default: return Point< 3 >{};
   }
}

/**
 * @brief Recover MFEM hex face coordinates from a native face point.
 *
 * This is the inverse of `MFEMHexFacePointToNative` on a valid native face
 * point.
 *
 * @param[in] mfem_local_face MFEM hex local face id.
 * @param[in] native Full-D native hex reference coordinate.
 * @return MFEM square face coordinates.
 */
inline Point< 2 > NativeToMFEMHexFacePoint(
   Integer mfem_local_face,
   const Point< 3 > & native )
{
   switch ( mfem_local_face )
   {
      case 0: return Point< 2 >{ native[0], Real( 1 ) - native[1] };
      case 1: return Point< 2 >{ native[0], native[2] };
      case 2: return Point< 2 >{ native[1], native[2] };
      case 3: return Point< 2 >{ Real( 1 ) - native[0], native[2] };
      case 4: return Point< 2 >{ Real( 1 ) - native[1], native[2] };
      case 5: return Point< 2 >{ native[0], native[1] };
      default: return Point< 2 >{};
   }
}

/**
 * @brief Convert an exact 0/1 coordinate value into a corner endpoint.
 *
 * The decoder reaches this helper only with coordinates generated by signed
 * permutations of canonical corners, so exact comparison is intentional.
 *
 * @param[in] value Coordinate value expected to be exactly 0 or 1.
 * @param[out] endpoint Set to 0 or 1 on success.
 * @return True when @p value is exactly an endpoint.
 */
inline bool ExactEndpoint( Real value, Integer & endpoint )
{
   if ( value == Real( 0 ) )
   {
      endpoint = 0;
      return true;
   }
   if ( value == Real( 1 ) )
   {
      endpoint = 1;
      return true;
   }
   return false;
}

/**
 * @brief Identify the MFEM quad endpoint represented by a native corner.
 *
 * Faces 2 and 3 use reversed MFEM edge coordinates, so the recovered endpoint
 * is flipped on those faces.
 *
 * @param[in] mfem_local_face MFEM quad local face id.
 * @param[in] native Full-D native quad corner.
 * @param[out] endpoint MFEM endpoint id, 0 or 1.
 * @return True when @p native is an exact corner on the requested face.
 */
inline bool MFEMQuadNativeCornerEndpoint(
   Integer mfem_local_face,
   const Point< 2 > & native,
   Integer & endpoint )
{
   switch ( mfem_local_face )
   {
      case 0: return ExactEndpoint( native[0], endpoint );
      case 1: return ExactEndpoint( native[1], endpoint );
      case 2:
      {
         Integer bit = 0;
         if ( !ExactEndpoint( native[0], bit ) ) { return false; }
         endpoint = 1 - bit;
         return true;
      }
      case 3:
      {
         Integer bit = 0;
         if ( !ExactEndpoint( native[1], bit ) ) { return false; }
         endpoint = 1 - bit;
         return true;
      }
      default: return false;
   }
}

/**
 * @brief Identify the MFEM hex face corner represented by a native corner.
 *
 * The returned `(u,v)` coordinates are discrete MFEM square-corner coordinates,
 * not continuous parameters.
 *
 * @param[in] mfem_local_face MFEM hex local face id.
 * @param[in] native Full-D native hex corner.
 * @param[out] u MFEM corner u-coordinate, 0 or 1.
 * @param[out] v MFEM corner v-coordinate, 0 or 1.
 * @return True when @p native is an exact corner on the requested face.
 */
inline bool MFEMHexNativeCornerPoint(
   Integer mfem_local_face,
   const Point< 3 > & native,
   Integer & u,
   Integer & v )
{
   Integer bit = 0;
   switch ( mfem_local_face )
   {
      case 0:
         return ExactEndpoint( native[0], u ) &&
                ExactEndpoint( native[1], bit ) && ( v = 1 - bit, true );
      case 1:
         return ExactEndpoint( native[0], u ) &&
                ExactEndpoint( native[2], v );
      case 2:
         return ExactEndpoint( native[1], u ) &&
                ExactEndpoint( native[2], v );
      case 3:
         return ExactEndpoint( native[0], bit ) &&
                ExactEndpoint( native[2], v ) && ( u = 1 - bit, true );
      case 4:
         return ExactEndpoint( native[1], bit ) &&
                ExactEndpoint( native[2], v ) && ( u = 1 - bit, true );
      case 5:
         return ExactEndpoint( native[0], u ) &&
                ExactEndpoint( native[1], v );
      default: return false;
   }
}

/**
 * @brief Map an MFEM 2D slave endpoint to a raw point-matrix column.
 *
 * Local serial MFEM 2D slave point matrices store endpoint 0 in raw column 1
 * and endpoint 1 in raw column 0.
 *
 * @param[in] endpoint MFEM slave endpoint id.
 * @return Raw segment point-matrix column.
 */
constexpr Integer SegmentPointMatrixColumn( Integer endpoint )
{
   // MFEM local serial 2D slave point matrices are parent-aligned:
   // slave endpoint 0 -> raw column 1, slave endpoint 1 -> raw column 0.
   return endpoint == 0 ? 1 : 0;
}

/**
 * @brief Map an MFEM square face corner to a raw point-matrix column.
 *
 * Raw MFEM square point-matrix order is `(0,0), (1,0), (1,1), (0,1)`.
 *
 * @param[in] u MFEM square corner u-coordinate.
 * @param[in] v MFEM square corner v-coordinate.
 * @return Raw square point-matrix column, or -1 for a non-corner input.
 */
constexpr int SquarePointMatrixColumn( Integer u, Integer v )
{
   // Raw MFEM order is (0,0), (1,0), (1,1), (0,1).
   return ( u == 0 && v == 0 ) ? 0 :
          ( u == 1 && v == 0 ) ? 1 :
          ( u == 1 && v == 1 ) ? 2 :
          ( u == 0 && v == 1 ) ? 3 : -1;
}

} // namespace detail
} // namespace mfem_interface
} // namespace gendil

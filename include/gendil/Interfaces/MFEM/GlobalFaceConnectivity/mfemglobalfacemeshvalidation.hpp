// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include <mfem.hpp>

#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Utilities/debug.hpp"

namespace gendil {
namespace mfem_interface {
namespace detail {

template < typename Geometry >
constexpr mfem::Geometry::Type MFEMElementGeometry()
{
   static_assert(
      is_hypercube_geometry< Geometry >::value,
      "MFEM global face builders support HyperCube geometries only." );
   static_assert(
      Geometry::geometry_dim == Geometry::space_dim,
      "MFEM global face builders do not support embedded HyperCube geometries." );

   if constexpr ( Geometry::geometry_dim == 1 )
   {
      return mfem::Geometry::SEGMENT;
   }
   else if constexpr ( Geometry::geometry_dim == 2 )
   {
      return mfem::Geometry::SQUARE;
   }
   else if constexpr ( Geometry::geometry_dim == 3 )
   {
      return mfem::Geometry::CUBE;
   }
   else
   {
      static_assert(
         Geometry::geometry_dim >= 1 && Geometry::geometry_dim <= 3,
         "MFEM global face builders support serial Line/Quad/Hex meshes only." );
      return mfem::Geometry::INVALID;
   }
}

template < typename Geometry >
constexpr mfem::Geometry::Type MFEMFaceGeometry()
{
   if constexpr ( Geometry::geometry_dim == 1 )
   {
      return mfem::Geometry::POINT;
   }
   else if constexpr ( Geometry::geometry_dim == 2 )
   {
      return mfem::Geometry::SEGMENT;
   }
   else if constexpr ( Geometry::geometry_dim == 3 )
   {
      return mfem::Geometry::SQUARE;
   }
   else
   {
      return mfem::Geometry::INVALID;
   }
}

template < typename Geometry >
void VerifyMFEMMeshGeometry( const mfem::Mesh & mesh )
{
   constexpr Integer dim = Geometry::geometry_dim;
   constexpr auto element_geometry = MFEMElementGeometry< Geometry >();

   GENDIL_VERIFY(
      mesh.Dimension() == dim,
      "MFEM global face builder mesh dimension does not match the requested Geometry." );
   GENDIL_VERIFY(
      mesh.SpaceDimension() == dim,
      "MFEM global face builder does not support embedded meshes." );

   for ( int element = 0; element < mesh.GetNE(); ++element )
   {
      GENDIL_VERIFY(
         mesh.GetElementGeometry( element ) == element_geometry,
         "MFEM global face builder found an unsupported element geometry." );
   }
}

template < typename Geometry >
void VerifyMFEMFaceGeometry( const mfem::Mesh & mesh, int source_face_id )
{
   GENDIL_VERIFY(
      mesh.GetFaceGeometry( source_face_id ) == MFEMFaceGeometry< Geometry >(),
      "MFEM global face builder found an unsupported face geometry." );
}

} // namespace detail
} // namespace mfem_interface
} // namespace gendil

#endif // GENDIL_USE_MFEM

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include <mfem.hpp>

#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/globalfacemetadata.hpp"
#include "gendil/Interfaces/MFEM/meshlocalconnectivity.hpp"
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/mfemglobalfacemeshvalidation.hpp"
#include "gendil/Interfaces/MFEM/orientation.hpp"
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/temporaryglobalfacematerialization.hpp"
#include "gendil/Meshes/Geometries/hypercubefaceorientation.hpp"

namespace gendil {
namespace mfem_interface {
namespace detail {

template < typename Geometry >
void VerifyConformingInteriorFaceInformation(
   const mfem::Mesh & mesh,
   int source_face_id,
   const mfem::Mesh::FaceInformation & face_info )
{
   GENDIL_VERIFY(
      face_info.IsConforming(),
      "MFEM conforming global interior face builder does not support nonconforming faces." );
   GENDIL_VERIFY(
      face_info.element[0].location == mfem::Mesh::ElementLocation::Local &&
      face_info.element[1].location == mfem::Mesh::ElementLocation::Local,
      "MFEM conforming global interior face builder supports serial local faces only." );
   GENDIL_VERIFY(
      face_info.element[0].index >= 0 &&
      face_info.element[0].index < mesh.GetNE() &&
      face_info.element[1].index >= 0 &&
      face_info.element[1].index < mesh.GetNE(),
      "MFEM conforming global interior face builder found an invalid element index." );
   GENDIL_VERIFY(
      face_info.element[0].index != face_info.element[1].index,
      "MFEM conforming global interior face builder rejects self-neighbor faces." );

   VerifyMFEMFaceGeometry< Geometry >( mesh, source_face_id );
}

template < typename Geometry >
void AppendConformingInteriorFaceRecord(
   const mfem::Mesh & mesh,
   int source_face_id,
   const mfem::Mesh::FaceInformation & face_info,
   PendingBuckets<
      UnstructuredInteriorFaceRecord< Geometry::geometry_dim >,
      MFEMInteriorFaceMetadata,
      Geometry::num_faces > & pending )
{
   VerifyConformingInteriorFaceInformation< Geometry >(
      mesh,
      source_face_id,
      face_info );

   const Integer g0 =
      TranslateMFEMFaceIndex< Geometry >(
         face_info.element[0].local_face_id );
   const Integer g1 =
      TranslateMFEMFaceIndex< Geometry >(
         face_info.element[1].local_face_id );
   const auto plus_orientation = TranslateMFEMOrientation(
      mfem_orientation< Geometry::geometry_dim >{
         face_info.element[0].local_face_id,
         face_info.element[1].local_face_id,
         face_info.element[1].orientation
      } );

   GENDIL_VERIFY(
      NativeFaceFromReferenceFace(
         Geometry::GetOppositeFaceIndex( g0 ),
         plus_orientation ) == g1,
      "MFEM conforming global interior face builder produced an invalid plus orientation." );

   using Record = UnstructuredInteriorFaceRecord< Geometry::geometry_dim >;
   using Metadata = MFEMInteriorFaceMetadata;

   pending[ g0 ].push_back(
      PendingFamilyRecord< Record, Metadata >{
         source_face_id,
         g0,
         Record{
            static_cast< GlobalIndex >( face_info.element[0].index ),
            static_cast< GlobalIndex >( face_info.element[1].index ),
            plus_orientation
         },
         Metadata{ source_face_id }
      } );
}

} // namespace detail
} // namespace mfem_interface

/**
 * @brief Build conforming global interior face families from a serial MFEM
 * Line/Quad/Hex mesh.
 *
 * MFEM side 0 is the minus side. Nonconforming, shared, ghost, simplex,
 * embedded, and self-neighbor faces are rejected by this v1 conforming builder.
 */
template < typename Geometry >
MFEMConformingInteriorConnectivityBundle< Geometry >
MakeMFEMConformingGlobalInteriorFaceConnectivity( const mfem::Mesh & mesh )
{
   mfem_interface::detail::VerifyMFEMMeshGeometry< Geometry >( mesh );

   using Record = UnstructuredInteriorFaceRecord< Geometry::geometry_dim >;
   using Metadata = MFEMInteriorFaceMetadata;

   mfem_interface::detail::PendingBuckets<
      Record,
      Metadata,
      Geometry::num_faces > pending;

   const auto & interior_face_ids = mesh.GetFaceIndices( mfem::FaceType::Interior );
   for ( int i = 0; i < interior_face_ids.Size(); ++i )
   {
      const int source_face_id = interior_face_ids[ i ];
      const auto face_info = mesh.GetFaceInformation( source_face_id );
      mfem_interface::detail::AppendConformingInteriorFaceRecord< Geometry >(
         mesh,
         source_face_id,
         face_info,
         pending );
   }

   return mfem_interface::detail::MaterializeConformingInteriorBundle<
      Geometry,
      Metadata >( pending );
}

} // namespace gendil

#endif // GENDIL_USE_MFEM

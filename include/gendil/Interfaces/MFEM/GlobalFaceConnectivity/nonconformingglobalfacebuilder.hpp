// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include <mfem.hpp>

#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/NonconformingPointMatrix/nonconformingpointmatrix.hpp"
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/conformingglobalfacebuilder.hpp"
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/globalfacemetadata.hpp"
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/mfemglobalfacemeshvalidation.hpp"
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/temporaryglobalfacematerialization.hpp"
#include "gendil/Interfaces/MFEM/meshlocalconnectivity.hpp"

namespace gendil {
namespace mfem_interface {
namespace detail {

inline bool IsMFEMLocalNonconformingLeaf(
   const mfem::Mesh::FaceInformation & face_info )
{
   return face_info.topology == mfem::Mesh::FaceTopology::Nonconforming &&
          face_info.tag == mfem::Mesh::FaceInfoTag::LocalSlaveNonconforming;
}

template < typename Geometry >
void VerifyNonconformingInteriorFaceInformation(
   const mfem::Mesh & mesh,
   int source_face_id,
   const mfem::Mesh::FaceInformation & face_info )
{
   GENDIL_VERIFY(
      Geometry::geometry_dim != 1,
      "MFEM nonconforming global interior face builder does not support 1D nonconforming faces." );
   GENDIL_VERIFY(
      IsMFEMLocalNonconformingLeaf( face_info ),
      "MFEM nonconforming global interior face builder supports local slave leaf faces only." );
   GENDIL_VERIFY(
      face_info.element[0].location == mfem::Mesh::ElementLocation::Local &&
      face_info.element[1].location == mfem::Mesh::ElementLocation::Local,
      "MFEM nonconforming global interior face builder supports serial local faces only." );
   GENDIL_VERIFY(
      face_info.element[0].index >= 0 &&
      face_info.element[0].index < mesh.GetNE() &&
      face_info.element[1].index >= 0 &&
      face_info.element[1].index < mesh.GetNE(),
      "MFEM nonconforming global interior face builder found an invalid element index." );
   GENDIL_VERIFY(
      face_info.element[0].index != face_info.element[1].index,
      "MFEM nonconforming global interior face builder rejects self-neighbor faces." );
   GENDIL_VERIFY(
      face_info.ncface >= 0,
      "MFEM nonconforming global interior face builder found an invalid ncface index." );
   GENDIL_VERIFY(
      face_info.point_matrix != nullptr,
      "MFEM nonconforming global interior face builder requires a point matrix." );

   VerifyMFEMFaceGeometry< Geometry >( mesh, source_face_id );
}

template < typename Geometry >
PointMatrixDecodeResult< Geometry::geometry_dim >
DecodeMFEMNonconformingPointMatrix(
   const mfem::Mesh::FaceInformation & face_info )
{
   constexpr Integer dim = Geometry::geometry_dim;

   const int fine_lf = face_info.element[0].local_face_id;
   const int coarse_lf = face_info.element[1].local_face_id;
   const Integer coarse_g0 = TranslateMFEMFaceIndex< Geometry >( coarse_lf );
   const Integer fine_g1 = TranslateMFEMFaceIndex< Geometry >( fine_lf );
   const auto & point_matrix = *face_info.point_matrix;

   if constexpr ( dim == 2 )
   {
      GENDIL_VERIFY(
         point_matrix.Height() == 1 &&
         point_matrix.Width() == 2,
         "MFEM nonconforming global interior face builder expected a 1x2 segment point matrix." );

      return DecodeMFEMPointMatrix(
         coarse_lf,
         fine_lf,
         coarse_g0,
         fine_g1,
         SegmentPointMatrix{ { point_matrix( 0, 0 ), point_matrix( 0, 1 ) } } );
   }
   else if constexpr ( dim == 3 )
   {
      GENDIL_VERIFY(
         point_matrix.Height() == 2 &&
         point_matrix.Width() == 4,
         "MFEM nonconforming global interior face builder expected a 2x4 square point matrix." );

      return DecodeMFEMPointMatrix(
         coarse_lf,
         fine_lf,
         coarse_g0,
         fine_g1,
         SquarePointMatrix{ {
            Point< 2 >{ point_matrix( 0, 0 ), point_matrix( 1, 0 ) },
            Point< 2 >{ point_matrix( 0, 1 ), point_matrix( 1, 1 ) },
            Point< 2 >{ point_matrix( 0, 2 ), point_matrix( 1, 2 ) },
            Point< 2 >{ point_matrix( 0, 3 ), point_matrix( 1, 3 ) }
         } } );
   }
   else
   {
      GENDIL_VERIFY(
         false,
         "MFEM nonconforming global interior face builder supports 2D and 3D leaves only." );
      return {};
   }
}

template < typename Geometry >
void AppendNonconformingInteriorFaceRecord(
   const mfem::Mesh & mesh,
   int source_face_id,
   const mfem::Mesh::FaceInformation & face_info,
   PendingBuckets<
      UnstructuredNonconformingInteriorFaceRecord< Geometry::geometry_dim >,
      MFEMNonconformingInteriorFaceMetadata,
      Geometry::num_faces > & pending )
{
   VerifyNonconformingInteriorFaceInformation< Geometry >(
      mesh,
      source_face_id,
      face_info );

   const int fine_lf = face_info.element[0].local_face_id;
   const int coarse_lf = face_info.element[1].local_face_id;
   const Integer coarse_g0 = TranslateMFEMFaceIndex< Geometry >( coarse_lf );
   const auto decoded =
      DecodeMFEMNonconformingPointMatrix< Geometry >( face_info );

   GENDIL_VERIFY(
      decoded.HasValue(),
      "MFEM nonconforming global interior face builder could not decode point matrix." );

   using Record =
      UnstructuredNonconformingInteriorFaceRecord< Geometry::geometry_dim >;
   using Metadata = MFEMNonconformingInteriorFaceMetadata;

   pending[ coarse_g0 ].push_back(
      PendingFamilyRecord< Record, Metadata >{
         source_face_id,
         coarse_g0,
         Record{
            static_cast< GlobalIndex >( face_info.element[1].index ),
            static_cast< GlobalIndex >( face_info.element[0].index ),
            decoded.value.plus_orientation,
            decoded.value.minus_map
         },
         Metadata{
            source_face_id,
            face_info.element[0].index,
            face_info.element[1].index,
            fine_lf,
            coarse_lf,
            face_info.ncface
         }
      } );
}

} // namespace detail
} // namespace mfem_interface

/**
 * @brief Build nonconforming global interior leaf-face families from a serial
 * MFEM Quad/Hex mesh.
 *
 * The minus side is the coarse/master cell. The plus side is the fine/slave
 * leaf cell. Conforming faces are ignored by this category builder; any
 * unsupported nonconforming, shared, ghost, or self-neighbor face is rejected.
 *
 * Point matrices are decoded during construction. The returned connectivity is
 * a materialized snapshot; after MFEM refinement/finalization changes, rebuild
 * the global-face connectivity.
 */
template < typename Geometry >
MFEMNonconformingInteriorConnectivityBundle< Geometry >
MakeMFEMNonconformingGlobalInteriorFaceConnectivity( const mfem::Mesh & mesh )
{
   mfem_interface::detail::VerifyMFEMMeshGeometry< Geometry >( mesh );

   using Record =
      UnstructuredNonconformingInteriorFaceRecord< Geometry::geometry_dim >;
   using Metadata = MFEMNonconformingInteriorFaceMetadata;

   mfem_interface::detail::PendingBuckets<
      Record,
      Metadata,
      Geometry::num_faces > pending;

   const auto & interior_face_ids = mesh.GetFaceIndices( mfem::FaceType::Interior );
   for ( int i = 0; i < interior_face_ids.Size(); ++i )
   {
      const int source_face_id = interior_face_ids[ i ];
      const auto face_info = mesh.GetFaceInformation( source_face_id );
      if ( face_info.IsConforming() )
      {
         continue;
      }

      mfem_interface::detail::AppendNonconformingInteriorFaceRecord< Geometry >(
         mesh,
         source_face_id,
         face_info,
         pending );
   }

   return mfem_interface::detail::MaterializeNonconformingInteriorBundle<
      Geometry,
      Metadata >( pending );
}

/**
 * @brief Build supported serial MFEM interior global-face categories.
 *
 * Conforming source faces are materialized in the conforming bundle. Local
 * slave nonconforming leaf faces are materialized in the nonconforming bundle.
 * Unsupported categories fail explicitly; no source face is skipped.
 *
 * This builder constructs a snapshot of the current MFEM topology. Rebuild it
 * after MFEM refinement/finalization changes.
 */
template < typename Geometry >
MFEMInteriorConnectivityBundle< Geometry >
MakeMFEMGlobalInteriorFaceConnectivity( const mfem::Mesh & mesh )
{
   mfem_interface::detail::VerifyMFEMMeshGeometry< Geometry >( mesh );

   using ConformingRecord =
      UnstructuredInteriorFaceRecord< Geometry::geometry_dim >;
   using NonconformingRecord =
      UnstructuredNonconformingInteriorFaceRecord< Geometry::geometry_dim >;

   mfem_interface::detail::PendingBuckets<
      ConformingRecord,
      MFEMInteriorFaceMetadata,
      Geometry::num_faces > conforming_pending;
   mfem_interface::detail::PendingBuckets<
      NonconformingRecord,
      MFEMNonconformingInteriorFaceMetadata,
      Geometry::num_faces > nonconforming_pending;

   const auto & interior_face_ids = mesh.GetFaceIndices( mfem::FaceType::Interior );
   for ( int i = 0; i < interior_face_ids.Size(); ++i )
   {
      const int source_face_id = interior_face_ids[ i ];
      const auto face_info = mesh.GetFaceInformation( source_face_id );

      if ( face_info.IsConforming() )
      {
         mfem_interface::detail::AppendConformingInteriorFaceRecord< Geometry >(
            mesh,
            source_face_id,
            face_info,
            conforming_pending );
      }
      else if ( mfem_interface::detail::IsMFEMLocalNonconformingLeaf( face_info ) )
      {
         mfem_interface::detail::AppendNonconformingInteriorFaceRecord< Geometry >(
            mesh,
            source_face_id,
            face_info,
            nonconforming_pending );
      }
      else
      {
         GENDIL_VERIFY(
            false,
            "MFEM global interior face builder found an unsupported interior face category." );
      }
   }

   return MFEMInteriorConnectivityBundle< Geometry >{
      mfem_interface::detail::MaterializeConformingInteriorBundle<
         Geometry,
         MFEMInteriorFaceMetadata >( conforming_pending ),
      mfem_interface::detail::MaterializeNonconformingInteriorBundle<
         Geometry,
         MFEMNonconformingInteriorFaceMetadata >( nonconforming_pending )
   };
}

} // namespace gendil

#endif // GENDIL_USE_MFEM

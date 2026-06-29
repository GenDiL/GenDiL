// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include <mfem.hpp>

#include <vector>

#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/globalfacemetadata.hpp"
#include "gendil/Interfaces/MFEM/meshlocalconnectivity.hpp"
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/mfemglobalfacemeshvalidation.hpp"
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/temporaryglobalfacematerialization.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"

namespace gendil {
namespace mfem_interface {
namespace detail {

inline bool ContainsBoundaryAttribute(
   const std::vector< int > & selected_attributes,
   int boundary_attribute )
{
   for ( int selected_attribute : selected_attributes )
   {
      if ( selected_attribute == boundary_attribute )
      {
         return true;
      }
   }
   return false;
}

template < typename Geometry >
void VerifyBoundaryFaceInformation(
   const mfem::Mesh & mesh,
   int source_face_id,
   const mfem::Mesh::FaceInformation & face_info )
{
   GENDIL_VERIFY(
      face_info.IsBoundary(),
      "MFEM global boundary face builder expected a boundary face." );
   GENDIL_VERIFY(
      face_info.element[0].location == mfem::Mesh::ElementLocation::Local,
      "MFEM global boundary face builder supports serial local boundary faces only." );
   GENDIL_VERIFY(
      face_info.element[1].location == mfem::Mesh::ElementLocation::NA,
      "MFEM global boundary face builder expected no second side." );
   GENDIL_VERIFY(
      face_info.element[0].index >= 0 &&
      face_info.element[0].index < mesh.GetNE(),
      "MFEM global boundary face builder found an invalid element index." );

   VerifyMFEMFaceGeometry< Geometry >( mesh, source_face_id );
}

template < typename Geometry, Integer FaceIndex >
void AppendFilteredBoundaryFamily(
   const MFEMBoundaryConnectivityBundle< Geometry > & bundle,
   const std::vector< int > & selected_attributes,
   PendingBuckets<
      UnstructuredBoundaryFaceRecord,
      MFEMBoundaryFaceMetadata,
      Geometry::num_faces > & pending,
   std::integral_constant< Integer, FaceIndex > )
{
   const auto & connectivity = std::get< FaceIndex >( bundle.connectivity );
   const auto & metadata = std::get< FaceIndex >( bundle.metadata );

   GENDIL_VERIFY(
      static_cast< size_t >( connectivity.GetNumberOfFaces() ) == metadata.size(),
      "MFEM boundary connectivity metadata is not aligned with connectivity records." );

   for ( GlobalIndex i = 0; i < connectivity.GetNumberOfFaces(); ++i )
   {
      const auto & meta = metadata[ static_cast< size_t >( i ) ];
      if ( !ContainsBoundaryAttribute(
              selected_attributes,
              meta.boundary_attribute ) )
      {
         continue;
      }

      pending[ FaceIndex ].push_back(
         mfem_interface::detail::PendingFamilyRecord<
            UnstructuredBoundaryFaceRecord,
            MFEMBoundaryFaceMetadata >{
            meta.source_face_id,
            FaceIndex,
            connectivity.records.host_pointer[ i ],
            meta
         } );
   }
}

template < typename Geometry >
auto FilterMFEMBoundaryFaceConnectivityByAttributes(
   const MFEMBoundaryConnectivityBundle< Geometry > & bundle,
   const std::vector< int > & selected_attributes )
{
   PendingBuckets<
      UnstructuredBoundaryFaceRecord,
      MFEMBoundaryFaceMetadata,
      Geometry::num_faces > pending;

   ConstexprLoop< Geometry::num_faces >(
      [&] ( auto family )
      {
         AppendFilteredBoundaryFamily(
            bundle,
            selected_attributes,
            pending,
            family );
      } );

   return MaterializeBoundaryBundle< Geometry >(
      pending,
      bundle.boundary_element_ids_requested );
}

} // namespace detail
} // namespace mfem_interface

/**
 * @brief Filter an MFEM boundary connectivity bundle by boundary attribute.
 */
template < typename Geometry >
MFEMBoundaryConnectivityBundle< Geometry >
FilterMFEMBoundaryFaceConnectivityByAttributes(
   const MFEMBoundaryConnectivityBundle< Geometry > & bundle,
   const std::vector< int > & selected_boundary_attributes )
{
   return mfem_interface::detail::FilterMFEMBoundaryFaceConnectivityByAttributes(
      bundle,
      selected_boundary_attributes );
}

/**
 * @brief Build one-sided global boundary face families from a serial MFEM
 * Line/Quad/Hex mesh.
 *
 * The minus side is the real boundary cell. The plus side in the returned face
 * information is a legacy dummy side and must not be consumed.
 */
template < typename Geometry >
MFEMBoundaryConnectivityBundle< Geometry >
MakeMFEMGlobalBoundaryFaceConnectivity(
   const mfem::Mesh & mesh,
   MFEMBoundaryMetadataOptions options = {} )
{
   mfem_interface::detail::VerifyMFEMMeshGeometry< Geometry >( mesh );

   using Record = UnstructuredBoundaryFaceRecord;
   using Metadata = MFEMBoundaryFaceMetadata;

   mfem_interface::detail::PendingBuckets<
      Record,
      Metadata,
      Geometry::num_faces > pending;

   const auto & boundary_face_ids = mesh.GetFaceIndices( mfem::FaceType::Boundary );
   const auto & boundary_attributes = mesh.GetBdrFaceAttributes();
   GENDIL_VERIFY(
      boundary_face_ids.Size() == boundary_attributes.Size(),
      "MFEM boundary face and boundary attribute arrays have different sizes." );

   mfem::Array< int > face_to_boundary_element;
   if ( options.include_boundary_element_ids )
   {
      face_to_boundary_element = mesh.GetFaceToBdrElMap();
   }

   for ( int i = 0; i < boundary_face_ids.Size(); ++i )
   {
      const int source_face_id = boundary_face_ids[ i ];
      const auto face_info = mesh.GetFaceInformation( source_face_id );
      mfem_interface::detail::VerifyBoundaryFaceInformation< Geometry >(
         mesh,
         source_face_id,
         face_info );

      const Integer g0 =
         mfem_interface::detail::TranslateMFEMFaceIndex< Geometry >(
            face_info.element[0].local_face_id );

      int boundary_element_id = MFEMInvalidBoundaryElementId;
      if ( options.include_boundary_element_ids )
      {
         GENDIL_VERIFY(
            source_face_id >= 0 &&
            source_face_id < face_to_boundary_element.Size(),
            "MFEM boundary face id is outside the face-to-boundary-element map." );

         const int be = face_to_boundary_element[ source_face_id ];
         if ( be >= 0 )
         {
            GENDIL_VERIFY(
               be < mesh.GetNBE(),
               "MFEM face-to-boundary-element map contains an invalid boundary element id." );
            boundary_element_id = be;
         }
      }

      pending[ g0 ].push_back(
         mfem_interface::detail::PendingFamilyRecord< Record, Metadata >{
            source_face_id,
            g0,
            Record{ static_cast< GlobalIndex >( face_info.element[0].index ) },
            Metadata{
               source_face_id,
               boundary_attributes[ i ],
               boundary_element_id
            }
         } );
   }

   return mfem_interface::detail::MaterializeBoundaryBundle< Geometry >(
      pending,
      options.include_boundary_element_ids );
}

} // namespace gendil

#endif // GENDIL_USE_MFEM

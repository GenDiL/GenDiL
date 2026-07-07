// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include <algorithm>
#include <array>
#include <tuple>
#include <utility>
#include <vector>

#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/globalfacemetadata.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/temporaryglobalfacematerialization.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"

namespace gendil {
namespace mfem_interface {
namespace detail {

// Transitional pending/vector materialization used by the current MFEM global
// face builders. Stage B replaces this with exact-size storage and removes
// this header.
template < class Record, class Metadata >
struct PendingFamilyRecord
{
   int source_face_id = -1;
   Integer family = 0;
   Record record{};
   Metadata metadata{};
};

template < typename Metadata, Integer NumFaces >
using MetadataBuckets = std::array< std::vector< Metadata >, NumFaces >;

template < typename Record, typename Metadata, Integer NumFaces >
using PendingBuckets =
   std::array< std::vector< PendingFamilyRecord< Record, Metadata > >, NumFaces >;

template < typename Geometry, typename Metadata >
auto MakeFamilyMetadataTuple(
   const MetadataBuckets< Metadata, Geometry::num_faces > & buckets )
{
   MFEMFamilyMetadataTuple< Geometry, Metadata > metadata;
   ConstexprLoop< Geometry::num_faces >(
      [&] ( auto family )
      {
         std::get< family >( metadata ) = buckets[ family ];
      } );
   return metadata;
}

template < typename Pending >
void SortPendingFamily( std::vector< Pending > & pending )
{
   std::stable_sort(
      pending.begin(),
      pending.end(),
      [] ( const Pending & a, const Pending & b )
      {
         return a.source_face_id < b.source_face_id;
      } );
}

template < typename Geometry, typename Metadata >
auto MaterializeConformingInteriorBundle(
   PendingBuckets<
      UnstructuredInteriorFaceRecord< Geometry::geometry_dim >,
      Metadata,
      Geometry::num_faces > & pending )
{
   unstructured_global_face_detail::InteriorBuckets< Geometry > records;
   MetadataBuckets< Metadata, Geometry::num_faces > metadata;

   for ( Integer family = 0; family < Geometry::num_faces; ++family )
   {
      auto & family_pending = pending[ family ];
      SortPendingFamily( family_pending );
      for ( const auto & item : family_pending )
      {
         records[ family ].push_back( item.record );
         metadata[ family ].push_back( item.metadata );
      }
   }

   return MFEMConformingInteriorConnectivityBundle< Geometry >{
      unstructured_global_face_detail::MakeUnstructuredInteriorFaceConnectivity< Geometry >(
         records ),
      MakeFamilyMetadataTuple< Geometry >( metadata )
   };
}

template < typename Geometry, typename Metadata >
auto MaterializeNonconformingInteriorBundle(
   PendingBuckets<
      UnstructuredNonconformingInteriorFaceRecord< Geometry::geometry_dim >,
      Metadata,
      Geometry::num_faces > & pending )
{
   unstructured_global_face_detail::NonconformingInteriorBuckets< Geometry > records;
   MetadataBuckets< Metadata, Geometry::num_faces > metadata;

   for ( Integer family = 0; family < Geometry::num_faces; ++family )
   {
      auto & family_pending = pending[ family ];
      SortPendingFamily( family_pending );
      for ( const auto & item : family_pending )
      {
         records[ family ].push_back( item.record );
         metadata[ family ].push_back( item.metadata );
      }
   }

   return MFEMNonconformingInteriorConnectivityBundle< Geometry >{
      unstructured_global_face_detail::MakeUnstructuredNonconformingInteriorFaceConnectivity< Geometry >(
         records ),
      MakeFamilyMetadataTuple< Geometry >( metadata )
   };
}

template < typename Geometry, typename Metadata >
auto MaterializeBoundaryBundle(
   PendingBuckets<
      UnstructuredBoundaryFaceRecord,
      Metadata,
      Geometry::num_faces > & pending,
   bool boundary_element_ids_requested )
{
   unstructured_global_face_detail::BoundaryBuckets< Geometry > records;
   MetadataBuckets< Metadata, Geometry::num_faces > metadata;

   for ( Integer family = 0; family < Geometry::num_faces; ++family )
   {
      auto & family_pending = pending[ family ];
      SortPendingFamily( family_pending );
      for ( const auto & item : family_pending )
      {
         records[ family ].push_back( item.record );
         metadata[ family ].push_back( item.metadata );
      }
   }

   return MFEMBoundaryConnectivityBundle< Geometry >{
      unstructured_global_face_detail::MakeUnstructuredBoundaryFaceConnectivity< Geometry >(
         records ),
      MakeFamilyMetadataTuple< Geometry >( metadata ),
      boundary_element_ids_requested
   };
}

} // namespace detail
} // namespace mfem_interface
} // namespace gendil

#endif // GENDIL_USE_MFEM

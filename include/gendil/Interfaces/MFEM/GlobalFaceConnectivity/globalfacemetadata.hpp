// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief MFEM global-face metadata and public builder return bundles.
 *
 * These host-side metadata types are available only when GENDIL_USE_MFEM is
 * enabled. Metadata vectors are family-local and aligned with the record order
 * in the corresponding connectivity family.
 */

#ifdef GENDIL_USE_MFEM

#include <tuple>
#include <utility>
#include <vector>

#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/unstructuredglobalfacetuples.hpp"

namespace gendil {

template < typename Metadata, Integer >
using MFEMMetadataVectorForFace = std::vector< Metadata >;

template < typename Geometry, typename Metadata, Integer... I >
auto MFEMFamilyMetadataTupleType( std::integer_sequence< Integer, I... > )
   -> std::tuple< MFEMMetadataVectorForFace< Metadata, I >... >;

template < typename Geometry, typename Metadata >
using MFEMFamilyMetadataTuple =
   decltype( MFEMFamilyMetadataTupleType< Geometry, Metadata >(
      std::make_integer_sequence< Integer, Geometry::num_faces >{} ) );

struct MFEMBoundaryMetadataOptions
{
   bool include_boundary_element_ids = false;
};

inline constexpr int MFEMInvalidBoundaryElementId = -1;

/**
 * @brief Host metadata aligned with one MFEM conforming interior face record.
 */
struct MFEMInteriorFaceMetadata
{
   int source_face_id = -1;
};

/**
 * @brief Host metadata aligned with one MFEM nonconforming interior leaf-face
 * record.
 */
struct MFEMNonconformingInteriorFaceMetadata
{
   int source_face_id = -1;
   int fine_element_id = -1;
   int coarse_element_id = -1;
   int fine_mfem_local_face_id = -1;
   int coarse_mfem_local_face_id = -1;
   int ncface = -1;
};

/**
 * @brief Host metadata aligned with one MFEM boundary face record.
 *
 * If the enclosing bundle did not request boundary-element lookup, ignore
 * `boundary_element_id`. Otherwise, a nonnegative value is an MFEM boundary
 * element id and `MFEMInvalidBoundaryElementId` means no mapping was available.
 */
struct MFEMBoundaryFaceMetadata
{
   int source_face_id = -1;
   int boundary_attribute = -1;
   int boundary_element_id = MFEMInvalidBoundaryElementId;
};

template < typename Geometry >
/**
 * @brief Public result of the serial conforming MFEM interior face builder.
 */
struct MFEMConformingInteriorConnectivityBundle
{
   UnstructuredConformingInteriorConnectivityTuple< Geometry > connectivity;
   MFEMFamilyMetadataTuple< Geometry, MFEMInteriorFaceMetadata > metadata;
};

template < typename Geometry >
/**
 * @brief Public result of the serial MFEM nonconforming leaf-face builder.
 */
struct MFEMNonconformingInteriorConnectivityBundle
{
   UnstructuredNonconformingInteriorConnectivityTuple< Geometry > connectivity;
   MFEMFamilyMetadataTuple<
      Geometry,
      MFEMNonconformingInteriorFaceMetadata > metadata;
};

template < typename Geometry >
/**
 * @brief Public result of the serial MFEM interior face builder.
 *
 * The conforming tuple is suitable for current generic global interior
 * domains. The nonconforming tuple materializes supported leaf-face
 * construction only; generic nonconforming global-operator execution remains a
 * separate capability.
 */
struct MFEMInteriorConnectivityBundle
{
   MFEMConformingInteriorConnectivityBundle< Geometry > conforming;
   MFEMNonconformingInteriorConnectivityBundle< Geometry > nonconforming;
};

template < typename Geometry >
/**
 * @brief Public result of the serial MFEM boundary face builder.
 */
struct MFEMBoundaryConnectivityBundle
{
   UnstructuredBoundaryFaceConnectivityTuple< Geometry > connectivity;
   MFEMFamilyMetadataTuple< Geometry, MFEMBoundaryFaceMetadata > metadata;
   bool boundary_element_ids_requested = false;
};

} // namespace gendil

#endif // GENDIL_USE_MFEM

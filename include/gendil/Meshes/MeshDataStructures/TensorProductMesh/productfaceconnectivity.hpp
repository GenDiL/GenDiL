// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file productfaceconnectivity.hpp
 * @brief Lazy global-face connectivity for Cartesian-product partitions.
 *
 * This implementation header lifts a statically described source face family
 * into the reference coordinates of a Cartesian-product cell and extrudes each
 * source record across the cells of the other factor. It owns the source
 * connectivity by value and computes product records on demand; no product
 * face array is materialized.
 *
 * The header is consumed by productpartition.hpp. The public factory remains
 * MakeCartesianProductPartition, exposed through meshes.hpp.
 */

#include "gendil/prelude.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <array>
#include <type_traits>
#include <utility>

namespace gendil {

namespace cartesian_product_partition_detail {

/**
 * @brief Classifies conformity maps accepted by Cartesian-product faces.
 *
 * This implementation-local trait also supplies the source coordinate
 * dimension and distinguishes maps whose origin and size must be embedded.
 * Its primary template deliberately reports an unsupported, zero-dimensional
 * map so validation can issue the focused product-partition diagnostic.
 *
 * @tparam Map Source conformity-map type.
 */
template<class Map>
struct ProductConformityMapTraits
{
   static constexpr bool supported = false;
   static constexpr bool nonconforming = false;
   static constexpr Integer dim = 0;
};

template<Integer Dim>
struct ProductConformityMapTraits<ConformingFaceMap<Dim>>
{
   static constexpr bool supported = true;
   static constexpr bool nonconforming = false;
   static constexpr Integer dim = Dim;
};

template<Integer Dim>
struct ProductConformityMapTraits<NonconformingHyperCubeFaceMap<Dim>>
{
   static constexpr bool supported = true;
   static constexpr bool nonconforming = true;
   static constexpr Integer dim = Dim;
};

/**
 * @brief Decodes a canonical source normal into dimension, axis, and sign.
 *
 * The primary template represents an unsupported normal. The specialization
 * exposes the compile-time data needed to verify that the normal agrees with
 * the source local-face family before its axis is shifted into product
 * coordinates.
 *
 * @tparam Normal Source canonical-normal type.
 */
template<class Normal>
struct ProductCanonicalNormalTraits
{
   static constexpr bool supported = false;
   static constexpr Integer dim = 0;
   static constexpr Integer index = 0;
   static constexpr int sign = 0;
};

template<Integer Dim, Integer Index, int Sign>
struct ProductCanonicalNormalTraits<CanonicalVector<Dim, Index, Sign>>
{
   static constexpr bool supported = true;
   static constexpr Integer dim = Dim;
   static constexpr Integer index = Index;
   static constexpr int sign = Sign;
};

/**
 * @brief Extracts the compile-time local-face family exposed by a face side.
 *
 * Dynamic local-face indices are rejected because the lazy product adapter
 * synthesizes its full-dimensional FaceView type at compile time.
 *
 * @tparam LocalFaceIndex Source local-face index type.
 */
template<class LocalFaceIndex, class = void>
struct ProductStaticLocalFaceIndexTraits
{
   static constexpr bool supported = false;
   static constexpr Integer value = 0;
};

template<class LocalFaceIndex>
struct ProductStaticLocalFaceIndexTraits<
   LocalFaceIndex,
   std::void_t<decltype(LocalFaceIndex::value)>>
{
   static constexpr bool supported = true;
   static constexpr Integer value = LocalFaceIndex::value;
};

/**
 * @brief Computes the lifted conformity-map type for a product face side.
 *
 * Conforming maps remain conforming while acquiring the product dimension.
 * Nonconforming hypercube maps retain their representation and acquire the
 * product dimension. Unsupported maps instantiate the focused diagnostic.
 *
 * @tparam ProductDim Full Cartesian-product cell dimension.
 * @tparam Map Source conformity-map type.
 */
template<Integer ProductDim, class Map>
struct ProductConformityMapType
{
   static_assert(
      dependent_false_v<Map>,
      "MakeCartesianProductPartition: global face conformity maps must be "
      "ConformingFaceMap or NonconformingHyperCubeFaceMap.");
   using type = ConformingFaceMap<ProductDim>;
};

template<Integer ProductDim, Integer Dim>
struct ProductConformityMapType<ProductDim, ConformingFaceMap<Dim>>
{
   using type = ConformingFaceMap<ProductDim>;
};

template<Integer ProductDim, Integer Dim>
struct ProductConformityMapType<
   ProductDim,
   NonconformingHyperCubeFaceMap<Dim>>
{
   using type = NonconformingHyperCubeFaceMap<ProductDim>;
};

/**
 * @brief Embeds a source conformity map in a product coordinate block.
 *
 * A conforming map becomes the product-dimensional identity map. For a
 * nonconforming hypercube map, source origin and size components occupy
 * `[Offset, Offset + SourceDim)`. Every extruded coordinate receives origin
 * zero and unit size, so the map acts only in the source factor.
 *
 * @tparam ProductDim Full Cartesian-product cell dimension.
 * @tparam Offset First product coordinate belonging to the source factor.
 * @param map Source conformity map.
 * @return The corresponding product-dimensional conformity map.
 */
template<Integer ProductDim, Integer Offset, class Map>
GENDIL_HOST_DEVICE
auto LiftProductConformityMap(const Map& map)
{
   using MapType = std::remove_cvref_t<Map>;
   using Traits = ProductConformityMapTraits<MapType>;
   static_assert(
      Traits::supported,
      "MakeCartesianProductPartition: global face conformity maps must be "
      "ConformingFaceMap or NonconformingHyperCubeFaceMap.");
   static_assert(
      Offset + Traits::dim <= ProductDim,
      "MakeCartesianProductPartition: conformity-map dimension exceeds the "
      "product reference dimension.");

   if constexpr (Traits::nonconforming)
   {
      Point<ProductDim> origin{};
      std::array<Real, ProductDim> size{};
      for (Integer d = 0; d < ProductDim; ++d)
      {
         origin[d] = Real{0};
         size[d] = Real{1};
      }
      for (Integer d = 0; d < Traits::dim; ++d)
      {
         origin[Offset + d] = map.origin[d];
         size[Offset + d] = map.size[d];
      }
      return NonconformingHyperCubeFaceMap<ProductDim>{origin, size};
   }
   else
   {
      (void)map;
      return ConformingFaceMap<ProductDim>{};
   }
}

/**
 * @brief Validates the coordinate block used to lift a source orientation.
 *
 * Both the runtime-permutation and static-identity policies inherit this
 * contract so neither representation can bypass the dimension checks.
 *
 * @tparam ProductDim Full Cartesian-product cell dimension.
 * @tparam Offset First product coordinate belonging to the source factor.
 * @tparam SourceDim Source factor dimension.
 */
template<Integer ProductDim, Integer Offset, Integer SourceDim>
struct ProductOrientationLiftContract
{
   static constexpr bool offset_is_nonnegative = []
   {
      if constexpr (std::is_signed_v<Integer>)
      {
         return Offset >= Integer{0};
      }
      else
      {
         return true;
      }
   }();

   static constexpr bool block_fits = []
   {
      if constexpr (SourceDim > ProductDim)
      {
         return false;
      }
      else
      {
         return Offset <= ProductDim - SourceDim;
      }
   }();

   static_assert(
      ProductDim > 0,
      "MakeCartesianProductPartition: product orientation dimension must "
      "be positive.");
   static_assert(
      SourceDim > 0,
      "MakeCartesianProductPartition: source orientation dimension must "
      "be positive.");
   static_assert(
      offset_is_nonnegative,
      "MakeCartesianProductPartition: orientation coordinate offset must "
      "be nonnegative.");
   static_assert(
      SourceDim <= ProductDim,
      "MakeCartesianProductPartition: source orientation dimension exceeds "
      "the product dimension.");
   static_assert(
      block_fits,
      "MakeCartesianProductPartition: source orientation coordinate block "
      "exceeds the product dimension.");
};

/**
 * @brief Lifts a source orientation into the product coordinate system.
 *
 * The primary policy materializes the full product permutation. Its single
 * specialization below preserves the exact statically encoded identity.
 * Keeping the result type and value construction in one policy prevents the
 * generated FaceView type from diverging from the stored orientation value.
 * Coordinates outside the source block retain the reference permutation.
 *
 * @tparam ProductDim Full Cartesian-product cell dimension.
 * @tparam Offset First product coordinate belonging to the source factor.
 * @tparam SourceDim Source factor dimension.
 * @tparam SourceOrientation Concrete source orientation type.
 */
template<
   Integer ProductDim,
   Integer Offset,
   Integer SourceDim,
   class SourceOrientation>
struct ProductOrientationLift
   : ProductOrientationLiftContract<ProductDim, Offset, SourceDim>
{
   using type = Permutation<ProductDim>;

   GENDIL_HOST_DEVICE
   static type Apply(const SourceOrientation& source)
   {
      auto result = MakeReferencePermutation<ProductDim>();
      Set<Offset>(
         result,
         static_cast<Permutation<SourceDim>>(source));
      return result;
   }
};

template<Integer ProductDim, Integer Offset, Integer SourceDim>
struct ProductOrientationLift<
   ProductDim,
   Offset,
   SourceDim,
   IdentityOrientation<SourceDim>>
   : ProductOrientationLiftContract<ProductDim, Offset, SourceDim>
{
   using type = IdentityOrientation<ProductDim>;

   GENDIL_HOST_DEVICE
   static type Apply(const IdentityOrientation<SourceDim>&)
   {
      return {};
   }
};

/**
 * @brief Validates a source side and synthesizes its product FaceView type.
 *
 * Local-face axis, canonical-normal axis, conformity-map dimension, and
 * reference geometry must agree before the side is re-encoded at Offset.
 * Source local faces `[0, SourceDim)` are the negative families and
 * `[SourceDim, 2 * SourceDim)` are the positive families. If `a` is the
 * source axis, the product axis is `Offset + a`; the full-dimensional local
 * face is that axis for a negative face and `ProductDim + axis` for a
 * positive face. The canonical normal receives the same coordinate offset.
 *
 * @tparam ProductDim Full Cartesian-product cell dimension.
 * @tparam Offset First product coordinate belonging to the source factor.
 * @tparam Side Concrete source FaceView type.
 */
template<Integer ProductDim, Integer Offset, class Side>
struct ProductFaceSideTraits
{
   using source_side_type = std::remove_cvref_t<Side>;
   using source_geometry = typename source_side_type::geometry;
   using source_normal = typename source_side_type::normal_type;
   using source_conformity = typename source_side_type::conformity_type;
   using source_orientation_type = std::remove_cvref_t<
      typename source_side_type::orientation_type>;
   using normal_traits = ProductCanonicalNormalTraits<source_normal>;
   using conformity_traits = ProductConformityMapTraits<source_conformity>;
   using local_face_traits = ProductStaticLocalFaceIndexTraits<
      typename source_side_type::local_face_index_type>;

   static_assert(
      is_hypercube_geometry<source_geometry>::value,
      "MakeCartesianProductPartition: global face reference geometry must "
      "be a HyperCube.");

   static constexpr Integer SourceDim = source_geometry::geometry_dim;
   static_assert(
      local_face_traits::supported,
      "MakeCartesianProductPartition: global face meshes must expose a "
      "static local-face family.");
   static constexpr Integer SourceFace = local_face_traits::value;
   static_assert(
      0 <= SourceFace && SourceFace < 2 * SourceDim,
      "MakeCartesianProductPartition: global face meshes must expose a "
      "static local-face family in range.");
   static_assert(
      Offset + SourceDim <= ProductDim,
      "MakeCartesianProductPartition: global face dimension exceeds the "
      "product reference dimension.");
   static_assert(
      normal_traits::supported &&
         normal_traits::dim == SourceDim &&
         normal_traits::index == SourceFace % SourceDim &&
         normal_traits::sign == (SourceFace < SourceDim ? -1 : 1),
      "MakeCartesianProductPartition: global face normals must be canonical "
      "and match the static local-face family.");
   static_assert(
      conformity_traits::supported && conformity_traits::dim == SourceDim,
      "MakeCartesianProductPartition: global face conformity maps must be "
      "ConformingFaceMap or NonconformingHyperCubeFaceMap with the source "
      "cell dimension.");

   static constexpr Integer Axis = Offset + SourceFace % SourceDim;
   static constexpr int Sign = SourceFace < SourceDim ? -1 : 1;
   static constexpr Integer ProductFace =
      Sign < 0 ? Axis : ProductDim + Axis;

   using conformity_type =
      typename ProductConformityMapType<
         ProductDim,
         source_conformity>::type;
   using orientation_lift = ProductOrientationLift<
      ProductDim,
      Offset,
      SourceDim,
      source_orientation_type>;
   using orientation_type = typename orientation_lift::type;
   using type = FaceView<
      std::integral_constant<Integer, ProductFace>,
      HyperCube<ProductDim>,
      orientation_type,
      CanonicalVector<ProductDim, Axis, Sign>,
      conformity_type,
      bool>;
};

template<Integer ProductDim, Integer Offset, class Side>
using product_face_side_t =
   typename ProductFaceSideTraits<ProductDim, Offset, Side>::type;

/**
 * @brief Constructs one lifted product face side from a source side.
 *
 * The caller supplies the already-extruded product cell index. Static
 * local-face and normal values are synthesized by ProductFaceSideTraits;
 * orientation and conformity values are lifted into the source coordinate
 * block. The source boundary flag is preserved.
 *
 * @tparam ProductDim Full Cartesian-product cell dimension.
 * @tparam Offset First product coordinate belonging to the source factor.
 * @param source Source face side.
 * @param product_cell_index Cell index in the referenced product cell part.
 * @return A product-dimensional FaceView owning all lifted side values.
 */
template<Integer ProductDim, Integer Offset, class Side>
GENDIL_HOST_DEVICE
auto MakeProductFaceSide(
   const Side& source,
   const GlobalIndex product_cell_index)
{
   using Source = std::remove_cvref_t<Side>;
   using Traits = ProductFaceSideTraits<ProductDim, Offset, Source>;
   using ProductSide = typename Traits::type;
   using OrientationLift = typename Traits::orientation_lift;

   auto orientation = OrientationLift::Apply(source.GetOrientation());

   return ProductSide{
      product_cell_index,
      {},
      orientation,
      {},
      LiftProductConformityMap<ProductDim, Offset>(source.conformity),
      source.IsBoundary()};
}

/**
 * @brief Lazily extrudes an interior source face family across another mesh.
 *
 * The adapter stores the source connectivity by value, making it safe for a
 * product partition to outlive a temporary input partition. Its entity count
 * is `source_faces.GetNumberOfFaces() * extruded_cells`.
 *
 * When `SourceIsFirst` is true, source records vary fastest:
 * `face = source_face + source_face_count * extruded_cell`. Product cell
 * indices are `source_cell + side_stride * extruded_cell`. Otherwise,
 * extruded cells vary fastest:
 * `face = extruded_cell + extruded_cells * source_face`, and product cell
 * indices are `extruded_cell + side_stride * source_cell`.
 *
 * Minus and plus strides are intentionally distinct. This is required for an
 * interface whose referenced minus and plus cell parts have different cell
 * counts.
 *
 * @tparam SourceIsFirst Whether the source face belongs to the first factor.
 * @tparam FirstDim First-factor cell dimension.
 * @tparam SecondDim Second-factor cell dimension.
 * @tparam FaceMesh Value-owned source interior-face connectivity type.
 */
template<
   bool SourceIsFirst,
   Integer FirstDim,
   Integer SecondDim,
   class FaceMesh>
struct CartesianProductInteriorFaceConnectivity
{
   using source_face_mesh_type = FaceMesh;
   using source_face_info_type = std::remove_cvref_t<
      decltype(
         std::declval<const FaceMesh&>().GetGlobalFaceInfo(GlobalIndex{}))>;
   static constexpr Integer ProductDim = FirstDim + SecondDim;
   static constexpr Integer SourceOffset = SourceIsFirst ? 0 : FirstDim;

   using minus_side_type = product_face_side_t<
      ProductDim,
      SourceOffset,
      typename source_face_info_type::minus_side_type>;
   using plus_side_type = product_face_side_t<
      ProductDim,
      SourceOffset,
      typename source_face_info_type::plus_side_type>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   /// Value-owned source connectivity used to answer lazy product queries.
   FaceMesh source_faces;
   /// Source-cell count for the referenced minus cell part.
   GlobalIndex minus_stride = 0;
   /// Source-cell count for the referenced plus cell part.
   GlobalIndex plus_stride = 0;
   /// Number of cells in the other factor's selected cell part.
   GlobalIndex extruded_cells = 0;

   /**
    * @brief Returns the number of extruded face records.
    * @return Source face count multiplied by the extruded cell count.
    */
   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return source_faces.GetNumberOfFaces() * extruded_cells;
   }

   /**
    * @brief Materializes one full-dimensional interior-face record on demand.
    * @param face_index Product-family record index.
    * @return Lifted minus and plus sides with independently extruded cells.
    *
    * `face_index` must be smaller than GetNumberOfFaces(). In particular, this
    * method is not called for an explicitly declared zero-record family.
    */
   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex face_index) const
   {
      GlobalIndex source_face_index = 0;
      GlobalIndex extruded_cell = 0;
      if constexpr (SourceIsFirst)
      {
         const GlobalIndex source_count = source_faces.GetNumberOfFaces();
         source_face_index = face_index % source_count;
         extruded_cell = face_index / source_count;
      }
      else
      {
         extruded_cell = face_index % extruded_cells;
         source_face_index = face_index / extruded_cells;
      }

      const auto source = source_faces.GetGlobalFaceInfo(source_face_index);
      const GlobalIndex minus_cell = SourceIsFirst
         ? source.MinusSide().GetCellIndex() + minus_stride * extruded_cell
         : extruded_cell + minus_stride * source.MinusSide().GetCellIndex();
      const GlobalIndex plus_cell = SourceIsFirst
         ? source.PlusSide().GetCellIndex() + plus_stride * extruded_cell
         : extruded_cell + plus_stride * source.PlusSide().GetCellIndex();

      return face_info_type{
         MakeProductFaceSide<ProductDim, SourceOffset>(
            source.MinusSide(),
            minus_cell),
         MakeProductFaceSide<ProductDim, SourceOffset>(
            source.PlusSide(),
            plus_cell)};
   }
};

/**
 * @brief Lazily extrudes a boundary source face family across another mesh.
 *
 * Record enumeration and real-side cell indexing follow the same first- or
 * second-factor-derived formulas as
 * CartesianProductInteriorFaceConnectivity. Only the real minus side
 * represents product topology. The dummy plus side keeps its source cell
 * index verbatim because boundary execution is one-sided and that sentinel
 * must not be interpreted using product numbering.
 *
 * The source connectivity is stored by value, so nested products constructed
 * directly from temporary inner partitions retain valid boundary adapters.
 * Explicit zero-record boundary families remain present with zero entities.
 *
 * @tparam SourceIsFirst Whether the source face belongs to the first factor.
 * @tparam FirstDim First-factor cell dimension.
 * @tparam SecondDim Second-factor cell dimension.
 * @tparam FaceMesh Value-owned source boundary-face connectivity type.
 */
template<
   bool SourceIsFirst,
   Integer FirstDim,
   Integer SecondDim,
   class FaceMesh>
struct CartesianProductBoundaryFaceConnectivity
{
   using source_face_mesh_type = FaceMesh;
   using source_face_info_type = std::remove_cvref_t<
      decltype(
         std::declval<const FaceMesh&>().GetGlobalFaceInfo(GlobalIndex{}))>;
   static constexpr Integer ProductDim = FirstDim + SecondDim;
   static constexpr Integer SourceOffset = SourceIsFirst ? 0 : FirstDim;

   using minus_side_type = product_face_side_t<
      ProductDim,
      SourceOffset,
      typename source_face_info_type::minus_side_type>;
   using plus_side_type = product_face_side_t<
      ProductDim,
      SourceOffset,
      typename source_face_info_type::plus_side_type>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   /// Value-owned source connectivity used to answer lazy product queries.
   FaceMesh source_faces;
   /// Source-cell count for the referenced real (minus) cell part.
   GlobalIndex real_side_stride = 0;
   /// Number of cells in the other factor's selected cell part.
   GlobalIndex extruded_cells = 0;

   /**
    * @brief Returns the number of extruded boundary records.
    * @return Source boundary count multiplied by the extruded cell count.
    */
   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return source_faces.GetNumberOfFaces() * extruded_cells;
   }

   /**
    * @brief Materializes one full-dimensional boundary record on demand.
    * @param face_index Product-family record index.
    * @return A lifted real side and a lifted dummy side with its original cell
    * index.
    *
    * `face_index` must be smaller than GetNumberOfFaces(). In particular, this
    * method is not called for an explicitly declared zero-record family.
    */
   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex face_index) const
   {
      GlobalIndex source_face_index = 0;
      GlobalIndex extruded_cell = 0;
      if constexpr (SourceIsFirst)
      {
         const GlobalIndex source_count = source_faces.GetNumberOfFaces();
         source_face_index = face_index % source_count;
         extruded_cell = face_index / source_count;
      }
      else
      {
         extruded_cell = face_index % extruded_cells;
         source_face_index = face_index / extruded_cells;
      }

      const auto source = source_faces.GetGlobalFaceInfo(source_face_index);
      const GlobalIndex minus_cell = SourceIsFirst
         ? source.MinusSide().GetCellIndex() +
              real_side_stride * extruded_cell
         : extruded_cell +
              real_side_stride * source.MinusSide().GetCellIndex();

      return face_info_type{
         MakeProductFaceSide<ProductDim, SourceOffset>(
            source.MinusSide(),
            minus_cell),
         // Boundary execution is one-sided. Preserve the source dummy plus
         // cell index instead of interpreting it as product topology.
         MakeProductFaceSide<ProductDim, SourceOffset>(
            source.PlusSide(),
            source.PlusSide().GetCellIndex())};
   }
};

/**
 * @brief First-factor spelling of the interior product-face adapter.
 *
 * These position-specific aliases keep the underlying Boolean-specialized
 * connectivity types unchanged while making factor selection explicit at
 * construction sites.
 */
template<Integer FirstDim, Integer SecondDim, class FaceMesh>
using FirstFactorInteriorFaceConnectivity =
   CartesianProductInteriorFaceConnectivity<
      true,
      FirstDim,
      SecondDim,
      FaceMesh>;

/// Second-factor spelling of the interior product-face adapter.
template<Integer FirstDim, Integer SecondDim, class FaceMesh>
using SecondFactorInteriorFaceConnectivity =
   CartesianProductInteriorFaceConnectivity<
      false,
      FirstDim,
      SecondDim,
      FaceMesh>;

/// First-factor spelling of the boundary product-face adapter.
template<Integer FirstDim, Integer SecondDim, class FaceMesh>
using FirstFactorBoundaryFaceConnectivity =
   CartesianProductBoundaryFaceConnectivity<
      true,
      FirstDim,
      SecondDim,
      FaceMesh>;

/// Second-factor spelling of the boundary product-face adapter.
template<Integer FirstDim, Integer SecondDim, class FaceMesh>
using SecondFactorBoundaryFaceConnectivity =
   CartesianProductBoundaryFaceConnectivity<
      false,
      FirstDim,
      SecondDim,
      FaceMesh>;

} // namespace cartesian_product_partition_detail

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Borrowed scalar component views of component-vector FE spaces.
 */

#include "gendil/FiniteElementMethod/FiniteElements/vectorfiniteelement.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/vectorrestriction.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

template<size_t Component, class Space>
auto GetComponentFiniteElementSpace(const Space& space);

namespace component_finite_element_space_detail {

/**
 * A value-semantic mesh accessor for a component view.
 *
 * GenDiL mesh objects are kernel accessors: copying one does not clone
 * separately owned mesh storage. Keeping the accessor in the view is required
 * because the complete component space is captured by value in device kernels.
 */
template<class Mesh>
class BorrowedMeshView
{
   Mesh mesh_;

public:
   using cell_type = typename Mesh::cell_type;
   static constexpr Integer Dim = Mesh::Dim;

   explicit constexpr BorrowedMeshView(const Mesh& mesh) : mesh_(mesh) {}

   GENDIL_HOST_DEVICE
   constexpr GlobalIndex GetNumberOfCells() const
   {
      return mesh_.GetNumberOfCells();
   }

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetCell(const GlobalIndex cell) const
   {
      return mesh_.GetCell(cell);
   }

   template<Integer FaceIndex>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetLocalFaceInfo(
      const GlobalIndex cell,
      std::integral_constant<Integer, FaceIndex> face) const
   {
      return mesh_.GetLocalFaceInfo(cell, face);
   }
};

/** A partition facade used only to retain the parent component-space view. */
template<class Partition>
class BorrowedPartitionView
{
   const Partition& partition;

public:
   using cell_parts_type = typename Partition::cell_parts_type;
   using interior_face_parts_type = typename Partition::interior_face_parts_type;
   using boundary_face_parts_type = typename Partition::boundary_face_parts_type;

   static constexpr size_t num_cell_parts = Partition::num_cell_parts;
   static constexpr size_t num_interior_face_parts =
      Partition::num_interior_face_parts;
   static constexpr size_t num_boundary_face_parts =
      Partition::num_boundary_face_parts;

   explicit constexpr BorrowedPartitionView(const Partition& parent_partition)
      : partition(parent_partition)
   {}

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) CellParts() const
   {
      return partition.CellParts();
   }

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) InteriorFaceParts() const
   {
      return partition.InteriorFaceParts();
   }

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) BoundaryFaceParts() const
   {
      return partition.BoundaryFaceParts();
   }

   GENDIL_HOST_DEVICE
   constexpr size_t GetNumberOfCellParts() const
   {
      return num_cell_parts;
   }

   GENDIL_HOST_DEVICE
   constexpr size_t GetNumberOfInteriorFaceParts() const
   {
      return num_interior_face_parts;
   }

   GENDIL_HOST_DEVICE
   constexpr size_t GetNumberOfBoundaryFaceParts() const
   {
      return num_boundary_face_parts;
   }
};

template<size_t Component, class Mesh, class FiniteElement, class Restriction>
auto GetHomogeneousComponentFiniteElementSpace(
   const FiniteElementSpace<Mesh, FiniteElement, Restriction>& space)
{
   static_assert(
      is_vector_shape_functions_v<typename FiniteElement::shape_functions>,
      "GetComponentFiniteElementSpace requires vector shape functions." );
   static_assert(
      VectorElementDoFRestriction<Restriction>,
      "GetComponentFiniteElementSpace requires a vector restriction." );
   using MeshView = BorrowedMeshView<Mesh>;
   const auto finite_element =
      GetComponentFiniteElement<Component>(space.finite_element);
   const auto restriction =
      GetComponentRestriction<Component>(GetRestriction(space));
   return MakeFiniteElementSpace(
      MeshView{static_cast<const Mesh&>(space)},
      finite_element,
      restriction);
}

template<size_t Component,
         class CellSpacesTuple,
         class Partition,
         size_t... Part>
auto GetMixedComponentFiniteElementSpace(
   const MixedFiniteElementSpace<CellSpacesTuple, Partition>& space,
   std::index_sequence<Part...>)
{
   using PartitionView = BorrowedPartitionView<Partition>;
   auto cell_spaces = std::tuple{
      GetHomogeneousComponentFiniteElementSpace<Component>(
         space.template GetCellFiniteElementSpace<Part>())...};
   auto result = MixedFiniteElementSpace<
      decltype(cell_spaces),
      PartitionView>{
         std::move(cell_spaces),
         PartitionView{space.GetPartition()}};
   (void)GetAlgebraicDofExtent(result);
   return result;
}

template<class Space>
consteval size_t ComponentCount()
{
   if constexpr (is_mixed_finite_element_space_v<Space>)
   {
      static_assert(
         std::remove_cvref_t<Space>::num_cell_spaces > 0,
         "A mixed vector space requires at least one cell space." );
      using FirstSpace = std::remove_cvref_t<decltype(
         std::declval<const Space&>()
            .template GetCellFiniteElementSpace<0>())>;
      using ShapeFunctions =
         finite_element_space_shape_functions_t<FirstSpace>;
      static_assert(
         is_vector_shape_functions_v<ShapeFunctions>,
         "ForEachComponentFiniteElementSpace requires vector shape functions." );
      return static_cast<size_t>(ShapeFunctions::vector_dim);
   }
   else
   {
      using ShapeFunctions =
         finite_element_space_shape_functions_t<Space>;
      static_assert(
         is_vector_shape_functions_v<ShapeFunctions>,
         "ForEachComponentFiniteElementSpace requires vector shape functions." );
      return static_cast<size_t>(ShapeFunctions::vector_dim);
   }
}

template<class Space, class Function, size_t... Component>
void ForEachComponentFiniteElementSpaceImpl(
   const Space& space,
   Function&& function,
   std::index_sequence<Component...>)
{
   (std::forward<Function>(function)(
       std::integral_constant<size_t, Component>{},
       GetComponentFiniteElementSpace<Component>(space)),
    ...);
}

} // namespace component_finite_element_space_detail

template<class Partition>
struct is_partition<
   component_finite_element_space_detail::BorrowedPartitionView<Partition>>
   : std::true_type {};

/**
 * @brief Return a borrowed scalar component view of a vector FE space.
 *
 * The returned view owns only value-semantic mesh and restriction accessor
 * metadata. Partition metadata and any restriction map storage remain
 * borrowed from @p space, so the view must not outlive or survive movement of
 * its parent. Its scalar restriction retains the complete parent vector
 * algebraic extent.
 */
template<size_t Component, class Space>
auto GetComponentFiniteElementSpace(const Space& space)
{
   if constexpr (is_mixed_finite_element_space_v<Space>)
   {
      return component_finite_element_space_detail::
         GetMixedComponentFiniteElementSpace<Component>(
            space,
            std::make_index_sequence<
               std::remove_cvref_t<Space>::num_cell_spaces>{});
   }
   else
   {
      static_assert(
         is_cell_finite_element_space_v<Space>,
         "GetComponentFiniteElementSpace requires a FiniteElementSpace or "
         "MixedFiniteElementSpace." );
      return component_finite_element_space_detail::
         GetHomogeneousComponentFiniteElementSpace<Component>(space);
   }
}

/** Visit every scalar component view in compile-time component order. */
template<class Space, class Function>
void ForEachComponentFiniteElementSpace(
   const Space& space,
   Function&& function)
{
   constexpr size_t NumComponents =
      component_finite_element_space_detail::ComponentCount<Space>();
   component_finite_element_space_detail::
      ForEachComponentFiniteElementSpaceImpl(
         space,
         std::forward<Function>(function),
         std::make_index_sequence<NumComponents>{});
}

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Map-free H1 restrictions for nodal Cartesian finite elements.
 */

#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionunitweight.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/vectorrestriction.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianndmesh.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/get.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

/** @brief Request arithmetic nodal numbering on a Cartesian mesh. */
struct CartesianH1RestrictionSpecification { };

/**
 * @brief One-entry H1 restriction whose global index is computed on demand.
 *
 * Axis zero varies fastest. A nonperiodic axis containing @c ne elements of
 * polynomial order @c p has @c ne*p+1 nodal coordinates. A periodic axis has
 * @c ne*p coordinates and identifies its two end points.
 */
template<StaticDofShape DofShape, Integer Dim>
struct CartesianH1Restriction
{
   static_assert(
      Dim == static_cast<Integer>(DofShape::size()),
      "Cartesian H1 restriction rank must match its mesh dimension." );

   using dof_shape_type = DofShape;
   static constexpr Integer tensor_dim = Dim;

   std::array<GlobalIndex, Dim> element_sizes{};
   std::array<GlobalIndex, Dim> global_dof_sizes{};
   std::array<bool, Dim> periodic{};
   GlobalIndex global_offset = 0;
   GlobalIndex num_local_dofs = 0;
   GlobalIndex num_global_dofs = 0;
   GlobalIndex algebraic_dof_extent = 0;
};

namespace cartesian_h1_restriction_detail {

template<class DofShape, size_t... Axis>
consteval auto DofExtents(std::index_sequence<Axis...>)
{
   return std::array<GlobalIndex, sizeof...(Axis)>{
      static_cast<GlobalIndex>(seq_get_v<Axis, DofShape>)...};
}

template<class DofShape, Integer Dim>
auto MakeScalarRestriction(
   const CartesianMesh<Dim>& mesh,
   const GlobalIndex global_offset,
   const GlobalIndex algebraic_extent)
{
   constexpr auto local_sizes = DofExtents<DofShape>(
      std::make_index_sequence<static_cast<size_t>(Dim)>{});
   std::array<GlobalIndex, Dim> global_sizes{};
   GlobalIndex global_dofs = 1;
   for (Integer axis = 0; axis < Dim; ++axis)
   {
      GENDIL_VERIFY(
         mesh.sizes[axis] > 0,
         "Cartesian H1 restriction requires positive mesh extents." );
      GENDIL_VERIFY(
         local_sizes[axis] >= 2,
         "Cartesian H1 restriction requires nodal polynomial order at least one on every axis." );
      const GlobalIndex order = local_sizes[axis] - 1;
      const GlobalIndex interior_extent = CheckedMultiply(
         mesh.sizes[axis],
         order,
         "Cartesian H1 axis extent overflow." );
      global_sizes[axis] = mesh.is_periodic_boundary[axis]
         ? interior_extent
         : CheckedAdd(
              interior_extent,
              GlobalIndex{1},
              "Cartesian H1 axis extent overflow." );
      global_dofs = CheckedMultiply(
         global_dofs,
         global_sizes[axis],
         "Cartesian H1 global extent overflow." );
   }
   const GlobalIndex local_dofs = CheckedMultiply(
      static_cast<GlobalIndex>(mesh.GetNumberOfCells()),
      static_cast<GlobalIndex>(Product(DofShape{})),
      "Cartesian H1 local extent overflow." );
   return CartesianH1Restriction<DofShape, Dim>{
      mesh.sizes,
      global_sizes,
      mesh.is_periodic_boundary,
      global_offset,
      local_dofs,
      global_dofs,
      algebraic_extent};
}

template<class DofShape, Integer Dim>
GlobalIndex GetScalarGlobalDofCount(const CartesianMesh<Dim>& mesh)
{
   return GetNumberOfGlobalDofs(
      MakeScalarRestriction<DofShape>(mesh, 0, 0));
}

template<class ShapeFunctions, Integer Dim, size_t... Component>
auto MakeVectorRestriction(
   const CartesianMesh<Dim>& mesh,
   std::index_sequence<Component...>)
{
   const std::array<GlobalIndex, sizeof...(Component)> component_extents{
      GetScalarGlobalDofCount<
         component_dof_shape_t<ShapeFunctions, Component>>(mesh)...};
   std::array<GlobalIndex, sizeof...(Component)> component_offsets{};
   GlobalIndex vector_extent = 0;
   for (size_t component = 0; component < component_extents.size(); ++component)
   {
      component_offsets[component] = vector_extent;
      vector_extent = CheckedAdd(
         vector_extent,
         component_extents[component],
         "Cartesian vector H1 global extent overflow." );
   }

   auto components = std::tuple{
      MakeScalarRestriction<
         component_dof_shape_t<ShapeFunctions, Component>>(
            mesh,
            component_offsets[Component],
            vector_extent)...};
   const GlobalIndex num_local_dofs = CheckedElementLocalDofCount<
      ShapeFunctions>(static_cast<GlobalIndex>(mesh.GetNumberOfCells()));
   return VectorRestriction{
      std::move(components),
      num_local_dofs,
      vector_extent,
      vector_extent};
}

} // namespace cartesian_h1_restriction_detail

/** @brief Visit the unique global coordinate of one Cartesian nodal row. */
template<StaticDofShape DofShape, Integer Dim, class Visitor>
GENDIL_HOST_DEVICE
constexpr void ForEachRestrictionEntry(
   const CartesianH1Restriction<DofShape, Dim>& restriction,
   const GlobalIndex element,
   const std::array<GlobalIndex, Dim>& local_dof,
   Visitor&& visitor)
{
   GENDIL_ASSERT(
      element < Product(restriction.element_sizes),
      "Cartesian H1 element index is out of bounds." );
   GENDIL_ASSERT(
      details::DofIndexIsInBounds<DofShape>(local_dof),
      "Cartesian H1 local tensor index is out of bounds." );

   constexpr auto local_sizes =
      cartesian_h1_restriction_detail::DofExtents<DofShape>(
         std::make_index_sequence<static_cast<size_t>(Dim)>{});
   const auto element_index =
      GetStructuredSubIndices(element, restriction.element_sizes);
   GlobalIndex global = 0;
   GlobalIndex stride = 1;
   for (Integer axis = 0; axis < Dim; ++axis)
   {
      const GlobalIndex order = local_sizes[axis] - 1;
      GlobalIndex node = element_index[axis] * order + local_dof[axis];
      if (restriction.periodic[axis] &&
          node == restriction.global_dof_sizes[axis])
      {
         node = 0;
      }
      global += node * stride;
      stride *= restriction.global_dof_sizes[axis];
   }
   const GlobalIndex algebraic_dof = restriction.global_offset + global;
   GENDIL_ASSERT(
      algebraic_dof < restriction.algebraic_dof_extent,
      "Cartesian H1 global DoF is out of bounds." );
   std::forward<Visitor>(visitor)(
      algebraic_dof,
      RestrictionUnitWeight{});
}

template<StaticDofShape DofShape, Integer Dim>
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfLocalDofs(
   const CartesianH1Restriction<DofShape, Dim>& restriction)
{
   return restriction.num_local_dofs;
}

template<StaticDofShape DofShape, Integer Dim>
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfGlobalDofs(
   const CartesianH1Restriction<DofShape, Dim>& restriction)
{
   return restriction.num_global_dofs;
}

template<StaticDofShape DofShape, Integer Dim>
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetAlgebraicDofExtent(
   const CartesianH1Restriction<DofShape, Dim>& restriction)
{
   return restriction.algebraic_dof_extent;
}

template<StaticDofShape DofShape, Integer Dim>
inline constexpr size_t static_restriction_entry_count_v<
   CartesianH1Restriction<DofShape, Dim>> = 1;

template<StaticDofShape DofShape, Integer Dim>
inline constexpr bool restriction_may_share_global_dofs_v<
   CartesianH1Restriction<DofShape, Dim>> = true;

template<StaticDofShape DofShape, Integer Dim>
inline constexpr bool restriction_rows_have_unique_coordinates_v<
   CartesianH1Restriction<DofShape, Dim>> = true;

template<StaticDofShape DofShape, Integer Dim>
struct restriction_supports_element_reference_view<
   CartesianH1Restriction<DofShape, Dim>> : std::true_type { };

template<bool OnDevice, StaticDofShape DofShape, Integer Dim>
void ValidateRestrictionMemoryAccess(
   const CartesianH1Restriction<DofShape, Dim>&,
   const GlobalIndex)
{
   // The mapping is arithmetic and contains no separately resident storage.
}

/** @brief Complete a scalar Cartesian H1 restriction specification. */
template<Integer Dim, class FiniteElement>
auto MakeElementDoFRestriction(
   const CartesianMesh<Dim>& mesh,
   const FiniteElement&,
   const CartesianH1RestrictionSpecification&)
{
   using ShapeFunctions = typename FiniteElement::shape_functions;
   static_assert(
      !is_vector_shape_functions_v<ShapeFunctions>,
      "Scalar Cartesian H1 construction cannot be used with vector shape functions." );
   using DofShape = finite_element_dof_shape_t<ShapeFunctions>;
   const GlobalIndex global_extent =
      cartesian_h1_restriction_detail::GetScalarGlobalDofCount<DofShape>(mesh);
   return cartesian_h1_restriction_detail::MakeScalarRestriction<DofShape>(
      mesh,
      0,
      global_extent);
}

/** @brief Complete a component-major Cartesian vector H1 restriction. */
template<Integer Dim, class FiniteElement>
   requires is_vector_shape_functions_v<typename FiniteElement::shape_functions>
auto MakeElementDoFRestriction(
   const CartesianMesh<Dim>& mesh,
   const FiniteElement&,
   const CartesianH1RestrictionSpecification&)
{
   using ShapeFunctions = typename FiniteElement::shape_functions;
   return cartesian_h1_restriction_detail::MakeVectorRestriction<
      ShapeFunctions>(
         mesh,
         std::make_index_sequence<ShapeFunctions::vector_dim>{});
}

} // namespace gendil

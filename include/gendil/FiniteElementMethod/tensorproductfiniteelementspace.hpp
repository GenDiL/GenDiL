// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Construct tensor-product meshes, restrictions, and spaces from factors.
 */

#include "gendil/FiniteElementMethod/FiniteElements/tensorfiniteelement.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/tensorproductrestriction.hpp"
#include "gendil/Meshes/MeshDataStructures/TensorProductMesh/productmesh.hpp"
#include "gendil/Meshes/MeshDataStructures/TensorProductMesh/productpartition.hpp"
#include "gendil/Meshes/partition.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

namespace tensor_product_finite_element_space_detail {

/** Retain explicit factor topology; homogeneous factors contribute one cell part. */
template<class Space>
auto MakeFactorPartition(const Space& space)
{
   if constexpr (is_mixed_finite_element_space_v<Space>)
   {
      return space.GetPartition();
   }
   else
   {
      using Mesh = typename std::remove_cvref_t<Space>::mesh_type;
      return MakePartition(MakeCellPart(static_cast<const Mesh&>(space)));
   }
}

/** A single mixed factor retains its existing partition and face families. */
template<class Partition>
auto MakeProductPartition(const Partition& partition)
{
   return partition;
}

/** Compose partition factors left to right, preserving first-factor-fastest order. */
template<class FirstPartition, class SecondPartition, class... RemainingPartitions>
auto MakeProductPartition(
   const FirstPartition& first,
   const SecondPartition& second,
   const RemainingPartitions&... remaining)
{
   const auto product = MakeCartesianProductPartition(first, second);
   if constexpr (sizeof...(RemainingPartitions) == 0)
   {
      return product;
   }
   else
   {
      return MakeProductPartition(product, remaining...);
   }
}

template<class Space>
inline constexpr size_t num_factor_parts_v = []
{
   if constexpr (is_mixed_finite_element_space_v<Space>)
   {
      return std::remove_cvref_t<Space>::num_cell_spaces;
   }
   else
   {
      static_assert(
         is_cell_finite_element_space_v<Space>,
         "Tensor-product factors must be finite-element spaces." );
      return size_t{1};
   }
}();

template<size_t Part, class Space>
constexpr decltype(auto) GetFactorSpace(const Space& space)
{
   if constexpr (is_mixed_finite_element_space_v<Space>)
   {
      static_assert(Part < std::remove_cvref_t<Space>::num_cell_spaces);
      return space.template GetCellFiniteElementSpace<Part>();
   }
   else
   {
      static_assert(Part == 0);
      return (space);
   }
}

template<size_t FlatPart, size_t Factor, class FactorSpacesTuple>
consteval size_t TupleFactorPartIndex()
{
   constexpr size_t NumFactors =
      std::tuple_size_v<std::remove_cvref_t<FactorSpacesTuple>>;
   constexpr auto counts = []<size_t... I>(std::index_sequence<I...>)
   {
      return std::array<size_t, NumFactors>{
         num_factor_parts_v<
            std::tuple_element_t<I, std::remove_cvref_t<FactorSpacesTuple>>>...};
   }(std::make_index_sequence<NumFactors>{});
   size_t stride = 1;
   for (size_t i = 0; i < Factor; ++i)
   {
      stride *= counts[i];
   }
   return (FlatPart / stride) % counts[Factor];
}

template<size_t FlatPart,
         class ProductPartition,
         class FactorSpacesTuple,
         size_t... Factor>
auto MakeProductCellSpaceFromTuple(
   const ProductPartition& product_domain,
   const FactorSpacesTuple& factor_spaces,
   std::index_sequence<Factor...>)
{
   const auto finite_element = MakeTensorFiniteElement(
      GetFactorSpace<
         TupleFactorPartIndex<FlatPart, Factor, FactorSpacesTuple>()>(
            std::get<Factor>(factor_spaces)).finite_element...);
   const auto restriction = MakeTensorProductRestriction(
      GetFactorSpace<
         TupleFactorPartIndex<FlatPart, Factor, FactorSpacesTuple>()>(
            std::get<Factor>(factor_spaces))...);
   return MakeFiniteElementSpace(
      std::get<FlatPart>(product_domain.CellParts()).mesh,
      finite_element,
      restriction);
}

template<class ProductPartition,
         class FactorSpacesTuple,
         size_t... Part>
auto MakeProductCellSpaces(
   const ProductPartition& product_domain,
   const FactorSpacesTuple& factor_spaces,
   std::index_sequence<Part...>)
{
   constexpr size_t NumFactors =
      std::tuple_size_v<std::remove_cvref_t<FactorSpacesTuple>>;
   return std::tuple{
      MakeProductCellSpaceFromTuple<Part>(
         product_domain,
         factor_spaces,
         std::make_index_sequence<NumFactors>{})...};
}

template<class... FactorSpaces>
consteval size_t ProductCellPartCount()
{
   return (size_t{1} * ... * num_factor_parts_v<FactorSpaces>);
}

} // namespace tensor_product_finite_element_space_detail

/**
 * @brief Construct a homogeneous tensor-product space from its factor spaces.
 *
 * The Cartesian product of the factor meshes, the tensor finite element, and
 * the product restriction are constructed together in argument order. The
 * first factor varies fastest in element and algebraic coordinate numbering.
 * Existing factor sharing, weights, and field-wide algebraic extents are
 * preserved. Scalar/vector factors follow MakeTensorFiniteElement's broadcast
 * and componentwise zip conventions.
 *
 * To reuse an existing product mesh, construct the tensor finite element and
 * restriction separately and pass them to MakeFiniteElementSpace:
 * @code{.cpp}
 * auto finite_element = MakeTensorFiniteElement(
 *    x_space.finite_element, s_space.finite_element);
 * auto restriction = MakeTensorProductRestriction(x_space, s_space);
 * auto space = MakeFiniteElementSpace(product_mesh, finite_element, restriction);
 * @endcode
 *
 * @param factor_spaces Homogeneous spaces whose meshes and DoF maps are composed.
 * @return A FiniteElementSpace holding the constructed product mesh and accessors.
 */
template<class... FactorSpaces>
   requires (
      sizeof...(FactorSpaces) > 0 &&
      (is_cell_finite_element_space_v<FactorSpaces> && ...))
auto MakeTensorProductFiniteElementSpace(
   const FactorSpaces&... factor_spaces)
{
   const auto product_mesh = MakeCartesianProductMesh(
      static_cast<const typename FactorSpaces::mesh_type&>(factor_spaces)...);
   const auto finite_element =
      MakeTensorFiniteElement(factor_spaces.finite_element...);
   const auto restriction =
      MakeTensorProductRestriction(factor_spaces...);
   return MakeFiniteElementSpace(
      product_mesh,
      finite_element,
      restriction);
}

/**
 * @brief Construct a mixed tensor-product space from homogeneous and mixed factors.
 *
 * Mixed factors contribute their retained partitions. A homogeneous factor
 * contributes a single cell-only part, as in MakeMixedFiniteElementSpace(space).
 * The product extrudes only explicitly declared partition face families;
 * it does not infer global faces from a homogeneous mesh's local connectivity.
 * Represent that factor as a mixed space with an explicit one-part partition
 * when its global face families are needed. An integration domain may also
 * supply face connectivity independently of the resulting field partition.
 *
 * Partitions are composed left to right with MakeCartesianProductPartition.
 * Children are selected with first-factor-fastest ordering. Every child
 * restriction keeps its factor-wide algebraic coordinates, without part-local
 * rebasing. Mesh and restriction accessors are stored by value; externally
 * owned storage borrowed by those accessors must remain valid.
 *
 * To reuse an existing partition, form the child tensor finite elements and
 * restrictions and pass their tuples to MakeMixedFiniteElementSpace.
 *
 * @param factor_spaces Factor spaces, including at least one mixed space.
 * @return A MixedFiniteElementSpace retaining the constructed product partition.
 */
template<class... FactorSpaces>
   requires (
      sizeof...(FactorSpaces) > 0 &&
      ((is_cell_finite_element_space_v<FactorSpaces> ||
        is_mixed_finite_element_space_v<FactorSpaces>) && ...) &&
      (is_mixed_finite_element_space_v<FactorSpaces> || ...))
auto MakeTensorProductFiniteElementSpace(
   const FactorSpaces&... factor_spaces)
{
   const auto product_partition = tensor_product_finite_element_space_detail::
      MakeProductPartition(
         tensor_product_finite_element_space_detail::
            MakeFactorPartition(factor_spaces)...);
   using ProductPartition = std::remove_cvref_t<decltype(product_partition)>;
   constexpr size_t NumProductParts =
      tensor_product_finite_element_space_detail::
         ProductCellPartCount<FactorSpaces...>();
   static_assert(
      ProductPartition::num_cell_parts == NumProductParts,
      "Product partition cell-part count does not match the factor-space layout." );
   const auto factors = std::tie(factor_spaces...);
   auto cell_spaces = tensor_product_finite_element_space_detail::
      MakeProductCellSpaces(
         product_partition,
         factors,
         std::make_index_sequence<NumProductParts>{});
   auto result = MixedFiniteElementSpace<
      decltype(cell_spaces),
      ProductPartition>{
         std::move(cell_spaces),
         product_partition};
   (void)GetAlgebraicDofExtent(result);
   return result;
}

} // namespace gendil

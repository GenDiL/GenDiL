// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Partition-aware heterogeneous finite-element spaces.
 */

#include "gendil/prelude.hpp"
#include "gendil/Utilities/TupleHelperFunctions/tupletraits.hpp"
#include "gendil/Meshes/partition.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

/**
 * @brief Finite-element spaces associated with the cell parts of a partition.
 *
 * Tuple entry `I` is the homogeneous finite-element space associated with
 * cell part `I` of the partition. Finite elements, polynomial orders,
 * restrictions, and global DoF offsets may differ between entries.
 *
 * The retained partition describes the field's cell-part layout and may carry
 * field-owned face parts. Weak-form integration topology, geometry, and face
 * connectivity are supplied independently by a `PartitionIntegrationDomain`;
 * its face-part tuple need not match this field's face-part tuple.
 *
 * @tparam CellSpacesTuple Tuple containing one homogeneous
 * `FiniteElementSpace` per partition cell part.
 * @tparam Partition Partition type defining the corresponding cell-part
 * ordering and optional face parts.
 */
template<class CellSpacesTuple, class Partition>
struct MixedFiniteElementSpace
{
   using cell_spaces_type = CellSpacesTuple;
   using partition_type = Partition;
   using cell_parts_type = typename Partition::cell_parts_type;
   using interior_face_parts_type = typename Partition::interior_face_parts_type;
   using boundary_face_parts_type = typename Partition::boundary_face_parts_type;

   static constexpr size_t num_cell_spaces =
      std::tuple_size_v<std::remove_cvref_t<CellSpacesTuple>>;
   static constexpr size_t num_interior_face_parts =
      Partition::num_interior_face_parts;
   static constexpr size_t num_boundary_face_parts =
      Partition::num_boundary_face_parts;

   /** @brief Homogeneous spaces stored in partition cell-part order. */
   CellSpacesTuple cell_spaces;

   /** @brief Field partition retained by value. */
   Partition partition;

   /** @brief Return all cell finite-element spaces in cell-part order. */
   GENDIL_HOST_DEVICE
   constexpr const CellSpacesTuple& CellSpaces() const
   {
      return cell_spaces;
   }

   /** @brief Return the partition retained by this field space. */
   GENDIL_HOST_DEVICE
   constexpr const Partition& GetPartition() const
   {
      return partition;
   }

   /** @brief Return the partition cell parts. */
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) CellParts() const
   {
      return partition.CellParts();
   }

   /** @brief Return the field-owned interior face parts, if any. */
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) InteriorFaceParts() const
   {
      return partition.InteriorFaceParts();
   }

   /** @brief Return the field-owned boundary face parts, if any. */
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) BoundaryFaceParts() const
   {
      return partition.BoundaryFaceParts();
   }

   /** @brief Return the number of cell finite-element spaces. */
   GENDIL_HOST_DEVICE
   constexpr size_t GetNumberOfCellFiniteElementSpaces() const
   {
      return num_cell_spaces;
   }

   /**
    * @brief Return the finite-element space for cell part `I`.
    * @tparam I Compile-time cell-part index.
    */
   template<size_t I>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetCellFiniteElementSpace() const
   {
      return std::get<I>(cell_spaces);
   }

   /**
    * @brief Invoke a callable once for every cell finite-element space.
    * @param fn Callable accepting each heterogeneous tuple entry.
    */
   template<class Fn>
   constexpr void ForEachCellFiniteElementSpace(Fn&& fn) const
   {
      std::apply(
         [&] (const auto&... spaces)
         {
            (fn(spaces), ...);
         },
         cell_spaces);
   }

   /** @brief Return the total number of finite elements over all cell parts. */
   Integer GetNumberOfFiniteElements() const
   {
      return SumMixedFiniteElementSpaceCounts(
         cell_spaces,
         [] (const auto& space)
         {
            return space.GetNumberOfFiniteElements();
         });
   }

   /** @brief Return the sum of the cell spaces' finite-element DoF counts. */
   Integer GetNumberOfFiniteElementDofs() const
   {
      return SumMixedFiniteElementSpaceCounts(
         cell_spaces,
         [] (const auto& space)
         {
            return space.GetNumberOfFiniteElementDofs();
         });
   }

   /** @brief Return the total number of field-owned interior faces. */
   Integer GetNumberOfInteriorFaces() const
   {
      return SumMixedFiniteElementSpaceCounts(
         InteriorFaceParts(),
         [] (const auto& face_part)
         {
            return GetConcreteFaceMeshNumberOfFaces(face_part.face_mesh);
         });
   }

   /** @brief Return the total number of field-owned boundary faces. */
   Integer GetNumberOfBoundaryFaces() const
   {
      return SumMixedFiniteElementSpaceCounts(
         BoundaryFaceParts(),
         [] (const auto& face_part)
         {
            return GetConcreteFaceMeshNumberOfFaces(face_part.face_mesh);
         });
   }
};

/** @brief Detect a `MixedFiniteElementSpace` type. */
template<class T>
struct is_mixed_finite_element_space : std::false_type {};

template<class CellSpacesTuple, class Partition>
struct is_mixed_finite_element_space<
   MixedFiniteElementSpace<
      CellSpacesTuple,
      Partition>> : std::true_type {};

template<class T>
inline constexpr bool is_mixed_finite_element_space_v =
   is_mixed_finite_element_space<std::remove_cvref_t<T>>::value;

/** @brief Detect a homogeneous cell `FiniteElementSpace` type. */
template<class T>
struct is_cell_finite_element_space : std::false_type {};

template<class Mesh, class FiniteElement, class Restriction>
struct is_cell_finite_element_space<
   FiniteElementSpace<Mesh, FiniteElement, Restriction>> : std::true_type {};

template<class T>
inline constexpr bool is_cell_finite_element_space_v =
   is_cell_finite_element_space<std::remove_cvref_t<T>>::value;

/** @internal */
template<class T>
inline constexpr Integer mixed_finite_element_space_classification_count_v =
   static_cast<Integer>(is_cell_finite_element_space_v<T>);

/** @internal */
template<class T>
consteval void ValidateMixedFiniteElementSpaceArgument()
{
   constexpr Integer num_categories =
      mixed_finite_element_space_classification_count_v<T>;

   static_assert(
      num_categories > 0,
      "MakeMixedFiniteElementSpace: every argument must be a cell finite "
      "element space. Partition-owned face meshes are selected through "
      "MakeMixedFiniteElementSpace(partition, ...), not persistent face "
      "finite element spaces.");
   static_assert(
      num_categories < 2,
      "MakeMixedFiniteElementSpace: every argument must classify as exactly "
      "one supported finite element space category.");
}

/** @internal */
template<class Tuple, class CountFn>
Integer SumMixedFiniteElementSpaceCounts(const Tuple& tuple, CountFn&& count_fn)
{
   Integer total = 0;
   std::apply(
      [&] (const auto&... spaces)
      {
         ((total += count_fn(spaces)), ...);
      },
      tuple);
   return total;
}

/** @internal */
template<class FaceMesh>
Integer GetConcreteFaceMeshNumberOfFaces(const FaceMesh& face_mesh)
{
   static_assert(
      !is_tuple_v<FaceMesh>,
      "MixedFiniteElementSpace expects each partition face part to own one "
      "concrete face mesh. Tuple face meshes must be expanded by the "
      "Partition face-part factory.");
   return face_mesh.GetNumberOfFaces();
}

/** @internal */
template<class T>
constexpr auto as_mixed_cell_space_tuple(T&& arg)
{
   using Arg = std::remove_cvref_t<T>;

   if constexpr (is_tuple_v<Arg>)
   {
      return std::apply(
         [] (auto&&... entries)
         {
            return std::tuple_cat(
               as_mixed_cell_space_tuple(
                  std::forward<decltype(entries)>(entries))...);
         },
         std::forward<T>(arg));
   }
   else if constexpr (is_cell_finite_element_space_v<T>)
   {
      ValidateMixedFiniteElementSpaceArgument<Arg>();
      return std::make_tuple(std::forward<T>(arg));
   }
   else
   {
      ValidateMixedFiniteElementSpaceArgument<Arg>();
      return std::tuple{};
   }
}

namespace details {

template<class Space>
constexpr auto MakeCellPartFromFiniteElementSpace(const Space& space)
{
   using Mesh = typename std::remove_cvref_t<Space>::mesh_type;
   return MakeCellPart(static_cast<const Mesh&>(space));
}

template<class CellSpacesTuple, size_t... I>
constexpr auto MakeCellOnlyPartitionFromSpacesImpl(
   const CellSpacesTuple& cell_spaces,
   std::index_sequence<I...>)
{
   return MakePartition(
      MakeCellPartFromFiniteElementSpace(std::get<I>(cell_spaces))...);
}

template<class CellSpacesTuple>
constexpr auto MakeCellOnlyPartitionFromSpaces(
   const CellSpacesTuple& cell_spaces)
{
   return MakeCellOnlyPartitionFromSpacesImpl(
      cell_spaces,
      std::make_index_sequence<
         std::tuple_size_v<std::remove_cvref_t<CellSpacesTuple>>>{});
}

} // namespace details

/**
 * @brief Combine existing homogeneous spaces into a mixed cell space.
 *
 * Arguments may be individual homogeneous `FiniteElementSpace` values or
 * nested tuples of them. They are flattened in argument order, and a cell-only
 * partition is constructed from their meshes. The generated partition has no
 * global interior or boundary face parts; use a partition-taking overload
 * when global-face connectivity is required.
 *
 * @param spaces Homogeneous cell finite-element spaces or tuples of spaces.
 * @return A `MixedFiniteElementSpace` owning the flattened spaces and the
 * generated cell-only partition.
 * @tparam Spaces Homogeneous space or nested tuple argument types.
 */
template<class... Spaces>
constexpr auto MakeMixedFiniteElementSpace(Spaces&&... spaces)
{
   auto cell_spaces =
      std::tuple_cat(as_mixed_cell_space_tuple(std::forward<Spaces>(spaces))...);

   static_assert(
      std::tuple_size_v<std::remove_cvref_t<decltype(cell_spaces)>> > 0,
      "MakeMixedFiniteElementSpace: at least one cell finite element space is "
      "required.");

   using CellSpaces = std::remove_cvref_t<decltype(cell_spaces)>;
   auto partition =
      details::MakeCellOnlyPartitionFromSpaces(cell_spaces);
   using Partition = std::remove_cvref_t<decltype(partition)>;

   return MixedFiniteElementSpace<
      CellSpaces,
      Partition>{
         std::move(cell_spaces),
         std::move(partition) };
}

/**
 * @brief Request contiguous discontinuous-Galerkin direct-sum numbering.
 *
 * The corresponding factory overload creates one `L2Restriction` per cell
 * part. Offsets begin at zero and advance by each preceding cell space's DoF
 * count.
 */
struct DGDirectSumNumbering {};

namespace details {

template<
   size_t I,
   class CellPartsTuple,
   class FiniteElementsTuple,
   class RestrictionsTuple>
constexpr auto MakePartitionCellFiniteElementSpaceTuple(
   const CellPartsTuple& cell_parts,
   const FiniteElementsTuple& finite_elements,
   const RestrictionsTuple& restrictions)
{
   constexpr size_t NumCellParts =
      std::tuple_size_v<std::remove_cvref_t<CellPartsTuple>>;

   if constexpr (I == NumCellParts)
   {
      return std::tuple{};
   }
   else
   {
      const auto& cell_part = std::get<I>(cell_parts);
      const auto& finite_element = std::get<I>(finite_elements);
      const auto& restriction = std::get<I>(restrictions);
      auto finite_element_space =
         MakeFiniteElementSpace(
            cell_part.mesh,
            finite_element,
            restriction);

      return std::tuple_cat(
         std::tuple{ finite_element_space },
         MakePartitionCellFiniteElementSpaceTuple<I + 1>(
            cell_parts,
            finite_elements,
            restrictions));
   }
}

template<size_t I, class CellPartsTuple, class FiniteElementsTuple>
constexpr auto MakeDGDirectSumRestrictionTuple(
   const CellPartsTuple& cell_parts,
   const FiniteElementsTuple& finite_elements,
   GlobalIndex shift)
{
   constexpr size_t NumCellParts =
      std::tuple_size_v<std::remove_cvref_t<CellPartsTuple>>;

   if constexpr (I == NumCellParts)
   {
      return std::tuple{};
   }
   else
   {
      const auto& cell_part = std::get<I>(cell_parts);
      const auto& finite_element = std::get<I>(finite_elements);
      L2Restriction restriction{ shift };
      auto finite_element_space =
         MakeFiniteElementSpace(
            cell_part.mesh,
            finite_element,
            restriction);
      const GlobalIndex next_shift =
         shift + finite_element_space.GetNumberOfFiniteElementDofs();

      return std::tuple_cat(
         std::tuple{ restriction },
         MakeDGDirectSumRestrictionTuple<I + 1>(
            cell_parts,
            finite_elements,
            next_shift));
   }
}

} // namespace details

/**
 * @brief Construct a mixed space from a partition and explicit restrictions.
 *
 * Tuple entry `I` of `finite_elements` and `restrictions` is combined with the
 * mesh of partition cell part `I` to construct one homogeneous
 * `FiniteElementSpace`. Both tuples must have exactly one entry per cell part.
 *
 * @param partition Partition whose cell-part ordering defines the field
 * layout.
 * @param finite_elements Tuple containing one finite element per cell part.
 * @param restrictions Tuple containing one restriction per cell part.
 * @return A `MixedFiniteElementSpace` containing the constructed homogeneous
 * spaces and a retained partition value.
 * @tparam PartitionType Partition type.
 * @tparam FiniteElementsTuple Finite-element tuple type.
 * @tparam RestrictionsTuple Restriction tuple type.
 */
template<class PartitionType, class FiniteElementsTuple, class RestrictionsTuple>
   requires (
      is_partition_v<PartitionType> &&
      !std::is_same_v<
         std::remove_cvref_t<RestrictionsTuple>,
         DGDirectSumNumbering>)
constexpr auto MakeMixedFiniteElementSpace(
   PartitionType&& partition,
   FiniteElementsTuple&& finite_elements,
   RestrictionsTuple&& restrictions)
{
   using Partition = std::remove_cvref_t<PartitionType>;
   using FiniteElements = std::remove_cvref_t<FiniteElementsTuple>;
   using Restrictions = std::remove_cvref_t<RestrictionsTuple>;

   static_assert(
      is_tuple_v<FiniteElements>,
      "MakeMixedFiniteElementSpace(partition, finite_elements, "
      "restrictions): finite_elements must be a std::tuple.");

   static_assert(
      is_tuple_v<Restrictions>,
      "MakeMixedFiniteElementSpace(partition, finite_elements, "
      "restrictions): restrictions must be a std::tuple.");

   static_assert(
      tuple_size_or_zero_v<FiniteElements> ==
         Partition::num_cell_parts,
      "MakeMixedFiniteElementSpace(partition, finite_elements, "
      "restrictions): finite element tuple size must equal the number of "
      "CellParts.");

   static_assert(
      tuple_size_or_zero_v<Restrictions> ==
         Partition::num_cell_parts,
      "MakeMixedFiniteElementSpace(partition, finite_elements, "
      "restrictions): restriction tuple size must equal the number of "
      "CellParts.");

   if constexpr (
      is_tuple_v<FiniteElements> &&
      is_tuple_v<Restrictions> &&
      tuple_size_or_zero_v<FiniteElements> ==
         Partition::num_cell_parts &&
      tuple_size_or_zero_v<Restrictions> ==
         Partition::num_cell_parts)
   {
      auto cell_fes_tuple =
         details::MakePartitionCellFiniteElementSpaceTuple<0>(
            partition.CellParts(),
            finite_elements,
            restrictions);

      return MixedFiniteElementSpace<
         std::remove_cvref_t<decltype(cell_fes_tuple)>,
         Partition>{
            std::move(cell_fes_tuple),
            Partition{ partition } };
   }
   else
   {
      return MixedFiniteElementSpace<std::tuple<>, Partition>{
         {},
         Partition{ partition } };
   }
}

/**
 * @brief Construct a DG mixed space with contiguous direct-sum DoF numbering.
 *
 * Creates an `L2Restriction` for every partition cell part. The first part
 * starts at global DoF zero, and each subsequent part starts after all DoFs of
 * the preceding parts.
 *
 * @param partition Partition whose cell-part ordering defines the field
 * layout.
 * @param finite_elements Tuple containing one finite element per cell part.
 * @param numbering Tag selecting automatic DG direct-sum numbering.
 * @return A `MixedFiniteElementSpace` with automatically generated
 * restrictions.
 * @tparam PartitionType Partition type.
 * @tparam FiniteElementsTuple Finite-element tuple type.
 */
template<class PartitionType, class FiniteElementsTuple>
   requires is_partition_v<PartitionType>
constexpr auto MakeMixedFiniteElementSpace(
   PartitionType&& partition,
   FiniteElementsTuple&& finite_elements,
   DGDirectSumNumbering numbering)
{
   (void)numbering;
   using Partition = std::remove_cvref_t<PartitionType>;
   using FiniteElements = std::remove_cvref_t<FiniteElementsTuple>;

   static_assert(
      is_tuple_v<FiniteElements>,
      "MakeMixedFiniteElementSpace(partition, finite_elements, "
      "DGDirectSumNumbering): finite_elements must be a std::tuple.");

   static_assert(
      tuple_size_or_zero_v<FiniteElements> ==
         Partition::num_cell_parts,
      "MakeMixedFiniteElementSpace(partition, finite_elements, "
      "DGDirectSumNumbering): finite element tuple size must equal the "
      "number of CellParts.");

   if constexpr (
      is_tuple_v<FiniteElements> &&
      tuple_size_or_zero_v<FiniteElements> ==
         Partition::num_cell_parts)
   {
      auto restrictions =
         details::MakeDGDirectSumRestrictionTuple<0>(
            partition.CellParts(),
            finite_elements,
            GlobalIndex{0});

      return MakeMixedFiniteElementSpace(
         std::forward<PartitionType>(partition),
         std::forward<FiniteElementsTuple>(finite_elements),
         restrictions);
   }
   else
   {
      return MixedFiniteElementSpace<std::tuple<>, Partition>{
         {},
         Partition{ partition } };
   }
}

} // namespace gendil

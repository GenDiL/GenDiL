// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/TupleHelperFunctions/tupletraits.hpp"
#include "gendil/Meshes/partition.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/globalfacefiniteelementspace.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

template<class CellSpacesTuple, class InteriorFaceSpacesTuple, class BoundaryFaceSpacesTuple>
struct MixedFiniteElementSpace
{
   using cell_spaces_type = CellSpacesTuple;
   using interior_face_spaces_type = InteriorFaceSpacesTuple;
   using boundary_face_spaces_type = BoundaryFaceSpacesTuple;

   static constexpr size_t num_cell_spaces =
      std::tuple_size_v<std::remove_cvref_t<CellSpacesTuple>>;
   static constexpr size_t num_interior_face_spaces =
      std::tuple_size_v<std::remove_cvref_t<InteriorFaceSpacesTuple>>;
   static constexpr size_t num_boundary_face_spaces =
      std::tuple_size_v<std::remove_cvref_t<BoundaryFaceSpacesTuple>>;

   CellSpacesTuple cell_spaces;
   InteriorFaceSpacesTuple interior_face_spaces;
   BoundaryFaceSpacesTuple boundary_face_spaces;

   GENDIL_HOST_DEVICE
   constexpr const CellSpacesTuple& CellSpaces() const
   {
      return cell_spaces;
   }

   GENDIL_HOST_DEVICE
   constexpr const InteriorFaceSpacesTuple& InteriorFaceSpaces() const
   {
      return interior_face_spaces;
   }

   GENDIL_HOST_DEVICE
   constexpr const BoundaryFaceSpacesTuple& BoundaryFaceSpaces() const
   {
      return boundary_face_spaces;
   }

   GENDIL_HOST_DEVICE
   constexpr size_t GetNumberOfCellFiniteElementSpaces() const
   {
      return num_cell_spaces;
   }

   GENDIL_HOST_DEVICE
   constexpr size_t GetNumberOfInteriorFaceFiniteElementSpaces() const
   {
      return num_interior_face_spaces;
   }

   GENDIL_HOST_DEVICE
   constexpr size_t GetNumberOfBoundaryFaceFiniteElementSpaces() const
   {
      return num_boundary_face_spaces;
   }

   template<size_t I>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetCellFiniteElementSpace() const
   {
      return std::get<I>(cell_spaces);
   }

   template<size_t I>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetInteriorFaceFiniteElementSpace() const
   {
      return std::get<I>(interior_face_spaces);
   }

   template<size_t I>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetBoundaryFaceFiniteElementSpace() const
   {
      return std::get<I>(boundary_face_spaces);
   }

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

   template<class Fn>
   constexpr void ForEachInteriorFaceFiniteElementSpace(Fn&& fn) const
   {
      std::apply(
         [&] (const auto&... spaces)
         {
            (fn(spaces), ...);
         },
         interior_face_spaces);
   }

   template<class Fn>
   constexpr void ForEachBoundaryFaceFiniteElementSpace(Fn&& fn) const
   {
      std::apply(
         [&] (const auto&... spaces)
         {
            (fn(spaces), ...);
         },
         boundary_face_spaces);
   }

   Integer GetNumberOfFiniteElements() const
   {
      return SumMixedFiniteElementSpaceCounts(
         cell_spaces,
         [] (const auto& space)
         {
            return space.GetNumberOfFiniteElements();
         });
   }

   Integer GetNumberOfFiniteElementDofs() const
   {
      return SumMixedFiniteElementSpaceCounts(
         cell_spaces,
         [] (const auto& space)
         {
            return space.GetNumberOfFiniteElementDofs();
         });
   }

   Integer GetNumberOfInteriorFaces() const
   {
      return SumMixedFiniteElementSpaceCounts(
         interior_face_spaces,
         [] (const auto& face_space)
         {
            return GetConcreteFaceMeshNumberOfFaces(face_space.GetFaceMesh());
         });
   }

   Integer GetNumberOfBoundaryFaces() const
   {
      return SumMixedFiniteElementSpaceCounts(
         boundary_face_spaces,
         [] (const auto& face_space)
         {
            return GetConcreteFaceMeshNumberOfFaces(face_space.GetFaceMesh());
         });
   }
};

template<class T>
struct is_mixed_finite_element_space : std::false_type {};

template<class CellSpacesTuple, class InteriorFaceSpacesTuple, class BoundaryFaceSpacesTuple>
struct is_mixed_finite_element_space<
   MixedFiniteElementSpace<
      CellSpacesTuple,
      InteriorFaceSpacesTuple,
      BoundaryFaceSpacesTuple>> : std::true_type {};

template<class T>
inline constexpr bool is_mixed_finite_element_space_v =
   is_mixed_finite_element_space<std::remove_cvref_t<T>>::value;


template<class T>
struct is_cell_finite_element_space : std::false_type {};

template<class Mesh, class FiniteElement, class Restriction>
struct is_cell_finite_element_space<
   FiniteElementSpace<Mesh, FiniteElement, Restriction>> : std::true_type {};

template<class T>
inline constexpr bool is_cell_finite_element_space_v =
   is_cell_finite_element_space<std::remove_cvref_t<T>>::value;

template<class T>
inline constexpr Integer mixed_finite_element_space_classification_count_v =
   static_cast<Integer>(is_cell_finite_element_space_v<T>) +
   static_cast<Integer>(is_interior_face_finite_element_space_v<T>) +
   static_cast<Integer>(is_boundary_face_finite_element_space_v<T>);

template<class T>
consteval void ValidateMixedFiniteElementSpaceArgument()
{
   constexpr Integer num_categories =
      mixed_finite_element_space_classification_count_v<T>;

   static_assert(
      num_categories > 0,
      "MakeMixedFiniteElementSpace: every argument must be a cell finite "
      "element space, an interior face finite element space, or a boundary "
      "face finite element space.");
   static_assert(
      num_categories < 2,
      "MakeMixedFiniteElementSpace: every argument must classify as exactly "
      "one supported finite element space category.");
}

template<class FaceMesh>
Integer GetConcreteFaceMeshNumberOfFaces(const FaceMesh& face_mesh)
{
   static_assert(
      !is_tuple_v<FaceMesh>,
      "MixedFiniteElementSpace expects each face finite element space to own "
      "one concrete face mesh. Tuple face meshes must be normalized into "
      "tuple face finite element spaces by the face FES factory and then "
      "flattened by MakeMixedFiniteElementSpace.");
   return face_mesh.GetNumberOfFaces();
}

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

template<class T>
constexpr auto as_mixed_interior_face_space_tuple(T&& arg)
{
   using Arg = std::remove_cvref_t<T>;

   if constexpr (is_tuple_v<Arg>)
   {
      return std::apply(
         [] (auto&&... entries)
         {
            return std::tuple_cat(
               as_mixed_interior_face_space_tuple(
                  std::forward<decltype(entries)>(entries))...);
         },
         std::forward<T>(arg));
   }
   else if constexpr (is_interior_face_finite_element_space_v<T>)
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

template<class T>
constexpr auto as_mixed_boundary_face_space_tuple(T&& arg)
{
   using Arg = std::remove_cvref_t<T>;

   if constexpr (is_tuple_v<Arg>)
   {
      return std::apply(
         [] (auto&&... entries)
         {
            return std::tuple_cat(
               as_mixed_boundary_face_space_tuple(
                  std::forward<decltype(entries)>(entries))...);
         },
         std::forward<T>(arg));
   }
   else if constexpr (is_boundary_face_finite_element_space_v<T>)
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

template<class... Spaces>
constexpr auto MakeMixedFiniteElementSpace(Spaces&&... spaces)
{
   auto cell_spaces =
      std::tuple_cat(as_mixed_cell_space_tuple(std::forward<Spaces>(spaces))...);
   auto interior_face_spaces =
      std::tuple_cat(
         as_mixed_interior_face_space_tuple(std::forward<Spaces>(spaces))...);
   auto boundary_face_spaces =
      std::tuple_cat(
         as_mixed_boundary_face_space_tuple(std::forward<Spaces>(spaces))...);

   static_assert(
      std::tuple_size_v<std::remove_cvref_t<decltype(cell_spaces)>> > 0,
      "MakeMixedFiniteElementSpace: at least one cell finite element space is "
      "required.");

   return MixedFiniteElementSpace<
      std::remove_cvref_t<decltype(cell_spaces)>,
      std::remove_cvref_t<decltype(interior_face_spaces)>,
      std::remove_cvref_t<decltype(boundary_face_spaces)>>{
         std::move(cell_spaces),
         std::move(interior_face_spaces),
         std::move(boundary_face_spaces) };
}

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

template<class CellFESTuple, class InteriorFacePart>
constexpr auto MakePartitionInteriorFaceFiniteElementSpace(
   const CellFESTuple& cell_fes_tuple,
   const InteriorFacePart& face_part)
{
   using FacePart = std::remove_cvref_t<InteriorFacePart>;
   constexpr size_t MinusCellI = FacePart::minus_cell_index;
   constexpr size_t PlusCellI = FacePart::plus_cell_index;

   if constexpr (MinusCellI == PlusCellI)
   {
      return MakeGlobalInteriorFaceFiniteElementSpace(
         std::get<MinusCellI>(cell_fes_tuple),
         face_part.face_mesh);
   }
   else
   {
      return MakeGlobalInteriorFaceFiniteElementSpace(
         std::get<MinusCellI>(cell_fes_tuple),
         std::get<PlusCellI>(cell_fes_tuple),
         face_part.face_mesh);
   }
}

template<class CellFESTuple, class InteriorFacePartsTuple, size_t... Is>
constexpr auto MakePartitionInteriorFaceFiniteElementSpaceTuple(
   const CellFESTuple& cell_fes_tuple,
   const InteriorFacePartsTuple& interior_face_parts,
   std::index_sequence<Is...>)
{
   return std::tuple{
      MakePartitionInteriorFaceFiniteElementSpace(
         cell_fes_tuple,
         std::get<Is>(interior_face_parts))... };
}

template<class CellFESTuple, class BoundaryFacePart>
constexpr auto MakePartitionBoundaryFaceFiniteElementSpace(
   const CellFESTuple& cell_fes_tuple,
   const BoundaryFacePart& face_part)
{
   using FacePart = std::remove_cvref_t<BoundaryFacePart>;
   constexpr size_t CellI = FacePart::cell_index;

   return MakeGlobalBoundaryFaceFiniteElementSpace(
      std::get<CellI>(cell_fes_tuple),
      face_part.face_mesh);
}

template<class CellFESTuple, class BoundaryFacePartsTuple, size_t... Is>
constexpr auto MakePartitionBoundaryFaceFiniteElementSpaceTuple(
   const CellFESTuple& cell_fes_tuple,
   const BoundaryFacePartsTuple& boundary_face_parts,
   std::index_sequence<Is...>)
{
   return std::tuple{
      MakePartitionBoundaryFaceFiniteElementSpace(
         cell_fes_tuple,
         std::get<Is>(boundary_face_parts))... };
}

} // namespace details

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

      const auto& interior_face_parts = partition.InteriorFaceParts();
      using InteriorFaceParts =
         std::remove_cvref_t<decltype(interior_face_parts)>;
      auto interior_face_fes_tuple =
         details::MakePartitionInteriorFaceFiniteElementSpaceTuple(
            cell_fes_tuple,
            interior_face_parts,
            std::make_index_sequence<
               std::tuple_size_v<InteriorFaceParts>>{});

      const auto& boundary_face_parts = partition.BoundaryFaceParts();
      using BoundaryFaceParts =
         std::remove_cvref_t<decltype(boundary_face_parts)>;
      auto boundary_face_fes_tuple =
         details::MakePartitionBoundaryFaceFiniteElementSpaceTuple(
            cell_fes_tuple,
            boundary_face_parts,
            std::make_index_sequence<
               std::tuple_size_v<BoundaryFaceParts>>{});

      return MakeMixedFiniteElementSpace(
         cell_fes_tuple,
         interior_face_fes_tuple,
         boundary_face_fes_tuple);
   }
   else
   {
      return MixedFiniteElementSpace<std::tuple<>, std::tuple<>, std::tuple<>>{
         {},
         {},
         {} };
   }
}

template<class PartitionType, class FiniteElementsTuple>
   requires is_partition_v<PartitionType>
constexpr auto MakeMixedFiniteElementSpace(
   PartitionType&& partition,
   FiniteElementsTuple&& finite_elements,
   DGDirectSumNumbering)
{
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
      return MixedFiniteElementSpace<std::tuple<>, std::tuple<>, std::tuple<>>{
         {},
         {},
         {} };
   }
}

template<class Space>
struct IntegrationDomain
{
   Space space;
};

template<class Space>
struct CellIntegrationDomain
{
   Space space;
};

template<class Space>
struct InteriorFaceIntegrationDomain
{
   Space space;
};

template<class Space>
struct BoundaryFaceIntegrationDomain
{
   Space space;
};

template<class T>
struct is_integration_domain : std::false_type {};

template<class Space>
struct is_integration_domain<IntegrationDomain<Space>> : std::true_type {};

template<class T>
inline constexpr bool is_integration_domain_v =
   is_integration_domain<std::remove_cvref_t<T>>::value;

template<class T>
struct is_cell_integration_domain : std::false_type {};

template<class Space>
struct is_cell_integration_domain<CellIntegrationDomain<Space>>
   : std::true_type {};

template<class T>
inline constexpr bool is_cell_integration_domain_v =
   is_cell_integration_domain<std::remove_cvref_t<T>>::value;

template<class T>
struct is_interior_face_integration_domain : std::false_type {};

template<class Space>
struct is_interior_face_integration_domain<
   InteriorFaceIntegrationDomain<Space>> : std::true_type {};

template<class T>
inline constexpr bool is_interior_face_integration_domain_v =
   is_interior_face_integration_domain<std::remove_cvref_t<T>>::value;

template<class T>
struct is_boundary_face_integration_domain : std::false_type {};

template<class Space>
struct is_boundary_face_integration_domain<
   BoundaryFaceIntegrationDomain<Space>> : std::true_type {};

template<class T>
inline constexpr bool is_boundary_face_integration_domain_v =
   is_boundary_face_integration_domain<std::remove_cvref_t<T>>::value;

template<class IntegrationRule, class Space>
constexpr auto MakeMeshQuadData(const CellIntegrationDomain<Space>&)
{
   using SpaceType = std::remove_cvref_t<Space>;
   if constexpr (is_mixed_finite_element_space_v<SpaceType>)
   {
      static_assert(
         dependent_false_v<SpaceType>,
         "MakeMeshQuadData requires a selected homogeneous "
         "CellIntegrationDomain<Space>. Mixed finite element spaces must be "
         "iterated and restricted to a selected cell batch before qdata "
         "construction.");
   }
   else if constexpr (is_cell_finite_element_space_v<SpaceType>)
   {
      using Mesh = typename SpaceType::mesh_type;
      using QD = typename Mesh::cell_type::template QuadData<IntegrationRule>;
      return QD{};
   }
   else
   {
      static_assert(
         dependent_false_v<SpaceType>,
         "CellIntegrationDomain requires a homogeneous cell finite element "
         "space or a MixedFiniteElementSpace.");
   }
}

template<
   class IntegrationRule,
   class CellSpacesTuple,
   class InteriorFaceSpacesTuple,
   class BoundaryFaceSpacesTuple>
constexpr auto MakeFiniteElementQuadData(
   const MixedFiniteElementSpace<
      CellSpacesTuple,
      InteriorFaceSpacesTuple,
      BoundaryFaceSpacesTuple>&)
{
   static_assert(
      dependent_false_v<
         MixedFiniteElementSpace<
            CellSpacesTuple,
            InteriorFaceSpacesTuple,
            BoundaryFaceSpacesTuple>>,
      "MakeFiniteElementQuadData requires a selected homogeneous finite "
      "element space. Mixed finite element spaces must be iterated and "
      "restricted to a selected cell batch before qdata construction.");
}

template<
   class IntegrationRule,
   class CellSpacesTuple,
   class InteriorFaceSpacesTuple,
   class BoundaryFaceSpacesTuple>
constexpr auto MakeFiniteElementFacetQuadData(
   const MixedFiniteElementSpace<
      CellSpacesTuple,
      InteriorFaceSpacesTuple,
      BoundaryFaceSpacesTuple>&)
{
   static_assert(
      dependent_false_v<
         MixedFiniteElementSpace<
            CellSpacesTuple,
            InteriorFaceSpacesTuple,
            BoundaryFaceSpacesTuple>>,
      "MakeFiniteElementFacetQuadData builds local/cell-owned facet qdata "
      "from a homogeneous volume finite element space. Mixed finite element "
      "spaces must be restricted to a selected cell batch before local facet "
      "context construction; global facet context construction uses face FES "
      "bindings and MakeGlobalFacetFiniteElementQuadData.");
}

} // namespace gendil

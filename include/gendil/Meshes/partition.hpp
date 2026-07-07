// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/TupleHelperFunctions/tupletraits.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

template<class Mesh>
struct CellPart
{
   using mesh_type = Mesh;

   Mesh mesh;
};

template<
   size_t MinusCellI,
   size_t PlusCellI,
   class FaceMesh>
struct InteriorFacePart
{
   static_assert(
      !is_tuple_v<FaceMesh>,
      "InteriorFacePart must own one concrete face mesh. Tuple face meshes "
      "must be expanded by MakeInteriorFacePart.");

   using face_mesh_type = FaceMesh;
   static constexpr size_t minus_cell_index = MinusCellI;
   static constexpr size_t plus_cell_index = PlusCellI;

   FaceMesh face_mesh;
};

template<
   size_t CellI,
   class FaceMesh>
struct BoundaryFacePart
{
   static_assert(
      !is_tuple_v<FaceMesh>,
      "BoundaryFacePart must own one concrete face mesh. Tuple face meshes "
      "must be expanded by MakeBoundaryFacePart.");

   using face_mesh_type = FaceMesh;
   static constexpr size_t cell_index = CellI;

   FaceMesh face_mesh;
};

template<
   class CellPartsTuple,
   class InteriorFacePartsTuple,
   class BoundaryFacePartsTuple>
struct Partition
{
   using cell_parts_type = CellPartsTuple;
   using interior_face_parts_type = InteriorFacePartsTuple;
   using boundary_face_parts_type = BoundaryFacePartsTuple;

   static constexpr size_t num_cell_parts =
      std::tuple_size_v<std::remove_cvref_t<CellPartsTuple>>;
   static constexpr size_t num_interior_face_parts =
      std::tuple_size_v<std::remove_cvref_t<InteriorFacePartsTuple>>;
   static constexpr size_t num_boundary_face_parts =
      std::tuple_size_v<std::remove_cvref_t<BoundaryFacePartsTuple>>;

   CellPartsTuple cell_parts;
   InteriorFacePartsTuple interior_face_parts;
   BoundaryFacePartsTuple boundary_face_parts;

   GENDIL_HOST_DEVICE
   constexpr const CellPartsTuple& CellParts() const
   {
      return cell_parts;
   }

   GENDIL_HOST_DEVICE
   constexpr const InteriorFacePartsTuple& InteriorFaceParts() const
   {
      return interior_face_parts;
   }

   GENDIL_HOST_DEVICE
   constexpr const BoundaryFacePartsTuple& BoundaryFaceParts() const
   {
      return boundary_face_parts;
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

template<class T>
struct is_cell_part : std::false_type {};

template<class Mesh>
struct is_cell_part<CellPart<Mesh>> : std::true_type {};

template<class T>
inline constexpr bool is_cell_part_v =
   is_cell_part<std::remove_cvref_t<T>>::value;

template<class T>
struct is_interior_face_part : std::false_type {};

template<size_t MinusCellI, size_t PlusCellI, class FaceMesh>
struct is_interior_face_part<
   InteriorFacePart<MinusCellI, PlusCellI, FaceMesh>> : std::true_type {};

template<class T>
inline constexpr bool is_interior_face_part_v =
   is_interior_face_part<std::remove_cvref_t<T>>::value;

template<class T>
struct is_boundary_face_part : std::false_type {};

template<size_t CellI, class FaceMesh>
struct is_boundary_face_part<BoundaryFacePart<CellI, FaceMesh>>
   : std::true_type {};

template<class T>
inline constexpr bool is_boundary_face_part_v =
   is_boundary_face_part<std::remove_cvref_t<T>>::value;

template<class T>
struct is_partition : std::false_type {};

template<
   class CellPartsTuple,
   class InteriorFacePartsTuple,
   class BoundaryFacePartsTuple>
struct is_partition<
   Partition<
      CellPartsTuple,
      InteriorFacePartsTuple,
      BoundaryFacePartsTuple>> : std::true_type {};

template<class T>
inline constexpr bool is_partition_v =
   is_partition<std::remove_cvref_t<T>>::value;

template<class T>
inline constexpr size_t partition_part_classification_count_v =
   static_cast<size_t>(is_cell_part_v<T>) +
   static_cast<size_t>(is_interior_face_part_v<T>) +
   static_cast<size_t>(is_boundary_face_part_v<T>);

namespace partition_detail {

template<class T>
inline constexpr bool is_partition_face_part_v =
   is_interior_face_part_v<T> || is_boundary_face_part_v<T>;

template<class T>
constexpr auto as_flat_partition_argument_tuple(T&& arg);

template<class Tuple>
constexpr auto flatten_partition_tuple(Tuple&& tuple)
{
   return std::apply(
      [] (auto&&... entries)
      {
         return std::tuple_cat(
            as_flat_partition_argument_tuple(
               std::forward<decltype(entries)>(entries))...);
      },
      std::forward<Tuple>(tuple));
}

template<class T>
constexpr auto as_flat_partition_argument_tuple(T&& arg)
{
   using Arg = std::remove_cvref_t<T>;
   if constexpr (is_tuple_v<Arg>)
   {
      return flatten_partition_tuple(std::forward<T>(arg));
   }
   else
   {
      return std::tuple{ std::forward<T>(arg) };
   }
}

template<class Tuple, size_t... Is>
consteval bool all_partition_arguments_valid_impl(std::index_sequence<Is...>)
{
   return (true && ... &&
      (partition_part_classification_count_v<
         std::tuple_element_t<Is, Tuple>> == 1));
}

template<class Tuple>
inline constexpr bool all_partition_arguments_valid_v =
   all_partition_arguments_valid_impl<std::remove_cvref_t<Tuple>>(
      std::make_index_sequence<
         std::tuple_size_v<std::remove_cvref_t<Tuple>>>{});

template<class Tuple, size_t... Is>
consteval size_t count_cell_parts_impl(std::index_sequence<Is...>)
{
   return (size_t{0} + ... +
      static_cast<size_t>(
         is_cell_part_v<std::tuple_element_t<Is, Tuple>>));
}

template<class Tuple>
inline constexpr size_t count_cell_parts_v =
   count_cell_parts_impl<std::remove_cvref_t<Tuple>>(
      std::make_index_sequence<
         std::tuple_size_v<std::remove_cvref_t<Tuple>>>{});

template<class Tuple, size_t... Is>
consteval bool cell_parts_are_prefix_impl(std::index_sequence<Is...>)
{
   bool seen_face_part = false;
   bool valid = true;
   ((
      valid =
         valid &&
         !(is_cell_part_v<std::tuple_element_t<Is, Tuple>> && seen_face_part),
      seen_face_part =
         seen_face_part ||
         is_partition_face_part_v<std::tuple_element_t<Is, Tuple>>
   ), ...);
   return valid;
}

template<class Tuple>
inline constexpr bool cell_parts_are_prefix_v =
   cell_parts_are_prefix_impl<std::remove_cvref_t<Tuple>>(
      std::make_index_sequence<
         std::tuple_size_v<std::remove_cvref_t<Tuple>>>{});

template<size_t NumCellParts, class T>
inline constexpr bool partition_relation_indices_in_range_v = true;

template<
   size_t NumCellParts,
   size_t MinusCellI,
   size_t PlusCellI,
   class FaceMesh>
inline constexpr bool partition_relation_indices_in_range_v<
   NumCellParts,
   InteriorFacePart<MinusCellI, PlusCellI, FaceMesh>> =
      MinusCellI < NumCellParts && PlusCellI < NumCellParts;

template<size_t NumCellParts, size_t CellI, class FaceMesh>
inline constexpr bool partition_relation_indices_in_range_v<
   NumCellParts,
   BoundaryFacePart<CellI, FaceMesh>> = CellI < NumCellParts;

template<size_t NumCellParts, class Tuple, size_t... Is>
consteval bool all_partition_relation_indices_in_range_impl(
   std::index_sequence<Is...>)
{
   return (true && ... &&
      partition_relation_indices_in_range_v<
         NumCellParts,
         std::tuple_element_t<Is, Tuple>>);
}

template<size_t NumCellParts, class Tuple>
inline constexpr bool all_partition_relation_indices_in_range_v =
   all_partition_relation_indices_in_range_impl<
      NumCellParts,
      std::remove_cvref_t<Tuple>>(
         std::make_index_sequence<
            std::tuple_size_v<std::remove_cvref_t<Tuple>>>{});

template<class T>
constexpr auto as_partition_cell_part_tuple(const T& arg)
{
   if constexpr (is_cell_part_v<T>)
   {
      return std::tuple{ arg };
   }
   else
   {
      return std::tuple{};
   }
}

template<class T>
constexpr auto as_partition_interior_face_part_tuple(const T& arg)
{
   if constexpr (is_interior_face_part_v<T>)
   {
      return std::tuple{ arg };
   }
   else
   {
      return std::tuple{};
   }
}

template<class T>
constexpr auto as_partition_boundary_face_part_tuple(const T& arg)
{
   if constexpr (is_boundary_face_part_v<T>)
   {
      return std::tuple{ arg };
   }
   else
   {
      return std::tuple{};
   }
}

template<class FlatTuple>
constexpr auto make_partition_cell_parts_tuple(const FlatTuple& flat_tuple)
{
   return std::apply(
      [] (const auto&... entries)
      {
         return std::tuple_cat(as_partition_cell_part_tuple(entries)...);
      },
      flat_tuple);
}

template<class FlatTuple>
constexpr auto make_partition_interior_face_parts_tuple(
   const FlatTuple& flat_tuple)
{
   return std::apply(
      [] (const auto&... entries)
      {
         return std::tuple_cat(
            as_partition_interior_face_part_tuple(entries)...);
      },
      flat_tuple);
}

template<class FlatTuple>
constexpr auto make_partition_boundary_face_parts_tuple(
   const FlatTuple& flat_tuple)
{
   return std::apply(
      [] (const auto&... entries)
      {
         return std::tuple_cat(
            as_partition_boundary_face_part_tuple(entries)...);
      },
      flat_tuple);
}

} // namespace partition_detail

template<class Mesh>
constexpr auto MakeCellPart(Mesh&& mesh)
{
   using MeshType = std::remove_cvref_t<Mesh>;
   return CellPart<MeshType>{ static_cast<MeshType>(std::forward<Mesh>(mesh)) };
}

template<size_t MinusCellI, size_t PlusCellI, class FaceMesh>
constexpr auto MakeInteriorFacePart(FaceMesh&& face_mesh)
{
   using FaceMeshType = std::remove_cvref_t<FaceMesh>;
   if constexpr (is_tuple_v<FaceMeshType>)
   {
      return std::apply(
         [] (auto&&... concrete_face_meshes)
         {
            return std::tuple{
               MakeInteriorFacePart<MinusCellI, PlusCellI>(
                  std::forward<decltype(concrete_face_meshes)>(
                     concrete_face_meshes))... };
         },
         std::forward<FaceMesh>(face_mesh));
   }
   else
   {
      return InteriorFacePart<
         MinusCellI,
         PlusCellI,
         FaceMeshType>{
            static_cast<FaceMeshType>(std::forward<FaceMesh>(face_mesh)) };
   }
}

template<size_t CellI, class FaceMesh>
constexpr auto MakeBoundaryFacePart(FaceMesh&& face_mesh)
{
   using FaceMeshType = std::remove_cvref_t<FaceMesh>;
   if constexpr (is_tuple_v<FaceMeshType>)
   {
      return std::apply(
         [] (auto&&... concrete_face_meshes)
         {
            return std::tuple{
               MakeBoundaryFacePart<CellI>(
                  std::forward<decltype(concrete_face_meshes)>(
                     concrete_face_meshes))... };
         },
         std::forward<FaceMesh>(face_mesh));
   }
   else
   {
      return BoundaryFacePart<
         CellI,
         FaceMeshType>{
            static_cast<FaceMeshType>(std::forward<FaceMesh>(face_mesh)) };
   }
}

template<class... Parts>
constexpr auto MakePartition(Parts&&... parts)
{
   auto flat_parts =
      std::tuple_cat(
         partition_detail::as_flat_partition_argument_tuple(
            std::forward<Parts>(parts))...);
   using FlatParts = std::remove_cvref_t<decltype(flat_parts)>;

   static_assert(
      partition_detail::all_partition_arguments_valid_v<FlatParts>,
      "MakePartition: every argument must be a CellPart, an "
      "InteriorFacePart, or a BoundaryFacePart. Tuple arguments are accepted "
      "only as tuples of supported Partition parts.");

   constexpr size_t NumCellParts =
      partition_detail::count_cell_parts_v<FlatParts>;

   static_assert(
      NumCellParts > 0,
      "MakePartition: at least one CellPart is required.");

   static_assert(
      partition_detail::cell_parts_are_prefix_v<FlatParts>,
      "MakePartition: all CellPart arguments must appear before face part "
      "arguments.");

   static_assert(
      partition_detail::all_partition_relation_indices_in_range_v<
         NumCellParts,
         FlatParts>,
      "MakePartition: face part relation indices must be valid CellPart "
      "tuple indices.");

   auto cell_parts =
      partition_detail::make_partition_cell_parts_tuple(flat_parts);
   auto interior_face_parts =
      partition_detail::make_partition_interior_face_parts_tuple(flat_parts);
   auto boundary_face_parts =
      partition_detail::make_partition_boundary_face_parts_tuple(flat_parts);

   return Partition<
      std::remove_cvref_t<decltype(cell_parts)>,
      std::remove_cvref_t<decltype(interior_face_parts)>,
      std::remove_cvref_t<decltype(boundary_face_parts)>>{
         std::move(cell_parts),
         std::move(interior_face_parts),
         std::move(boundary_face_parts) };
}

} // namespace gendil

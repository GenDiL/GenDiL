// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
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
      !is_std_tuple_v<FaceMesh>,
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

   if constexpr (is_std_tuple_v<Arg>)
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

   if constexpr (is_std_tuple_v<Arg>)
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

   if constexpr (is_std_tuple_v<Arg>)
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

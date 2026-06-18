// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

#include <tuple>
#include <type_traits>

namespace gendil
{

template<class T>
struct is_std_tuple : std::false_type {};

template<class... Ts>
struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

template<class T>
inline constexpr bool is_std_tuple_v =
   is_std_tuple<std::remove_cvref_t<T>>::value;

// Same-space/two-space classification is determined by the construction
// category, not by CXX type equality. A two-space face FES constructed from
// minus/plus spaces of the same type still requires the two-space taxonomy.
template<class FiniteElementSpace, class InteriorFaceMesh>
struct GlobalSameSpaceInteriorFaceFiniteElementSpace
{
   static_assert(
      !is_std_tuple_v<InteriorFaceMesh>,
      "A global interior face finite element space must own exactly one "
      "concrete face mesh. If a face mesh factory returns a tuple, "
      "MakeGlobalInteriorFaceFiniteElementSpace maps it to a tuple of face "
      "finite element spaces.");

   using minus_finite_element_space_type = FiniteElementSpace;
   using plus_finite_element_space_type = FiniteElementSpace;
   using face_mesh_type = InteriorFaceMesh;
   static constexpr bool is_same_space_batch = true;

   FiniteElementSpace finite_element_space;
   InteriorFaceMesh face_mesh;

   GENDIL_HOST_DEVICE
   constexpr const FiniteElementSpace& GetMinusFiniteElementSpace() const
   {
      return finite_element_space;
   }

   GENDIL_HOST_DEVICE
   constexpr const FiniteElementSpace& GetPlusFiniteElementSpace() const
   {
      return finite_element_space;
   }

   GENDIL_HOST_DEVICE
   constexpr const InteriorFaceMesh& GetFaceMesh() const
   {
      return face_mesh;
   }
};

template<
   class MinusFiniteElementSpace,
   class PlusFiniteElementSpace,
   class InteriorFaceMesh>
struct GlobalTwoSpaceInteriorFaceFiniteElementSpace
{
   static_assert(
      !is_std_tuple_v<InteriorFaceMesh>,
      "A global interior face finite element space must own exactly one "
      "concrete face mesh. If a face mesh factory returns a tuple, "
      "MakeGlobalInteriorFaceFiniteElementSpace maps it to a tuple of face "
      "finite element spaces.");

   using minus_finite_element_space_type = MinusFiniteElementSpace;
   using plus_finite_element_space_type = PlusFiniteElementSpace;
   using face_mesh_type = InteriorFaceMesh;
   static constexpr bool is_same_space_batch = false;

   MinusFiniteElementSpace finite_element_space;
   PlusFiniteElementSpace plus_finite_element_space;
   InteriorFaceMesh face_mesh;

   GENDIL_HOST_DEVICE
   constexpr const MinusFiniteElementSpace& GetMinusFiniteElementSpace() const
   {
      return finite_element_space;
   }

   GENDIL_HOST_DEVICE
   constexpr const PlusFiniteElementSpace& GetPlusFiniteElementSpace() const
   {
      return plus_finite_element_space;
   }

   GENDIL_HOST_DEVICE
   constexpr const InteriorFaceMesh& GetFaceMesh() const
   {
      return face_mesh;
   }
};

template<class FiniteElementSpace, class InteriorFaceMesh>
constexpr auto MakeGlobalInteriorFaceFiniteElementSpace(
   const FiniteElementSpace& finite_element_space,
   const InteriorFaceMesh& face_mesh)
{
   if constexpr (is_std_tuple_v<InteriorFaceMesh>)
   {
      return std::apply(
         [&] (const auto&... concrete_face_meshes)
         {
            return std::tuple{
               MakeGlobalInteriorFaceFiniteElementSpace(
                  finite_element_space,
                  concrete_face_meshes)... };
         },
         face_mesh);
   }
   else
   {
      return GlobalSameSpaceInteriorFaceFiniteElementSpace<
         FiniteElementSpace,
         InteriorFaceMesh>{ finite_element_space, face_mesh };
   }
}

template<
   class MinusFiniteElementSpace,
   class PlusFiniteElementSpace,
   class InteriorFaceMesh>
constexpr auto MakeGlobalInteriorFaceFiniteElementSpace(
   const MinusFiniteElementSpace& minus_finite_element_space,
   const PlusFiniteElementSpace& plus_finite_element_space,
   const InteriorFaceMesh& face_mesh)
{
   if constexpr (is_std_tuple_v<InteriorFaceMesh>)
   {
      return std::apply(
         [&] (const auto&... concrete_face_meshes)
         {
            return std::tuple{
               MakeGlobalInteriorFaceFiniteElementSpace(
                  minus_finite_element_space,
                  plus_finite_element_space,
                  concrete_face_meshes)... };
         },
         face_mesh);
   }
   else
   {
      return GlobalTwoSpaceInteriorFaceFiniteElementSpace<
         MinusFiniteElementSpace,
         PlusFiniteElementSpace,
         InteriorFaceMesh>{
            minus_finite_element_space,
            plus_finite_element_space,
            face_mesh };
   }
}

template<class FiniteElementSpace, class BoundaryFaceMesh>
struct GlobalBoundaryFaceFiniteElementSpace
{
   static_assert(
      !is_std_tuple_v<BoundaryFaceMesh>,
      "A global boundary face finite element space must own exactly one "
      "concrete face mesh. If a face mesh factory returns a tuple, "
      "MakeGlobalBoundaryFaceFiniteElementSpace maps it to a tuple of face "
      "finite element spaces.");

   using finite_element_space_type = FiniteElementSpace;
   using face_mesh_type = BoundaryFaceMesh;

   FiniteElementSpace finite_element_space;
   BoundaryFaceMesh face_mesh;

   GENDIL_HOST_DEVICE
   constexpr const FiniteElementSpace& GetMinusFiniteElementSpace() const
   {
      return finite_element_space;
   }

   GENDIL_HOST_DEVICE
   constexpr const BoundaryFaceMesh& GetFaceMesh() const
   {
      return face_mesh;
   }
};

template<class FiniteElementSpace, class BoundaryFaceMesh>
constexpr auto MakeGlobalBoundaryFaceFiniteElementSpace(
   const FiniteElementSpace& finite_element_space,
   const BoundaryFaceMesh& face_mesh)
{
   if constexpr (is_std_tuple_v<BoundaryFaceMesh>)
   {
      return std::apply(
         [&] (const auto&... concrete_face_meshes)
         {
            return std::tuple{
               MakeGlobalBoundaryFaceFiniteElementSpace(
                  finite_element_space,
                  concrete_face_meshes)... };
         },
         face_mesh);
   }
   else
   {
      return GlobalBoundaryFaceFiniteElementSpace<
         FiniteElementSpace,
         BoundaryFaceMesh>{ finite_element_space, face_mesh };
   }
}

template<class T>
struct is_interior_face_finite_element_space : std::false_type {};

template<class Space, class InteriorFaceMesh>
struct is_interior_face_finite_element_space<
   GlobalSameSpaceInteriorFaceFiniteElementSpace<
      Space,
      InteriorFaceMesh>> : std::true_type {};

template<
   class MinusSpace,
   class PlusSpace,
   class InteriorFaceMesh>
struct is_interior_face_finite_element_space<
   GlobalTwoSpaceInteriorFaceFiniteElementSpace<
      MinusSpace,
      PlusSpace,
      InteriorFaceMesh>> : std::true_type {};

template<class T>
inline constexpr bool is_interior_face_finite_element_space_v =
   is_interior_face_finite_element_space<std::remove_cvref_t<T>>::value;

template<class T>
struct is_boundary_face_finite_element_space : std::false_type {};

template<class Space, class BoundaryFaceMesh>
struct is_boundary_face_finite_element_space<
   GlobalBoundaryFaceFiniteElementSpace<Space, BoundaryFaceMesh>> : std::true_type {};

template<class T>
inline constexpr bool is_boundary_face_finite_element_space_v =
   is_boundary_face_finite_element_space<std::remove_cvref_t<T>>::value;

template<class T>
struct is_face_finite_element_space
   : std::bool_constant<
        is_interior_face_finite_element_space_v<T> ||
        is_boundary_face_finite_element_space_v<T>> {};

template<class T>
inline constexpr bool is_face_finite_element_space_v =
   is_face_finite_element_space<std::remove_cvref_t<T>>::value;

template<class T>
struct is_same_space_interior_face_finite_element_space : std::false_type {};

template<class Space, class InteriorFaceMesh>
struct is_same_space_interior_face_finite_element_space<
   GlobalSameSpaceInteriorFaceFiniteElementSpace<
      Space,
      InteriorFaceMesh>> : std::true_type {};

template<
   class MinusSpace,
   class PlusSpace,
   class InteriorFaceMesh>
struct is_same_space_interior_face_finite_element_space<
   GlobalTwoSpaceInteriorFaceFiniteElementSpace<
      MinusSpace,
      PlusSpace,
      InteriorFaceMesh>> : std::false_type {};

template<class T>
inline constexpr bool is_same_space_interior_face_finite_element_space_v =
   is_same_space_interior_face_finite_element_space<std::remove_cvref_t<T>>::value;

template<class T>
struct is_two_space_interior_face_finite_element_space : std::false_type {};

template<class Space, class InteriorFaceMesh>
struct is_two_space_interior_face_finite_element_space<
   GlobalSameSpaceInteriorFaceFiniteElementSpace<
      Space,
      InteriorFaceMesh>> : std::false_type {};

template<
   class MinusSpace,
   class PlusSpace,
   class InteriorFaceMesh>
struct is_two_space_interior_face_finite_element_space<
   GlobalTwoSpaceInteriorFaceFiniteElementSpace<
      MinusSpace,
      PlusSpace,
      InteriorFaceMesh>> : std::true_type {};

template<class T>
inline constexpr bool is_two_space_interior_face_finite_element_space_v =
   is_two_space_interior_face_finite_element_space<std::remove_cvref_t<T>>::value;

template<class T>
struct requires_two_sided_face_qdata
   : is_two_space_interior_face_finite_element_space<std::remove_cvref_t<T>> {};

template<class T>
inline constexpr bool requires_two_sided_face_qdata_v =
   requires_two_sided_face_qdata<std::remove_cvref_t<T>>::value;

template<class T>
inline constexpr bool supports_one_sided_face_qdata_v =
   !requires_two_sided_face_qdata_v<std::remove_cvref_t<T>>;

} // namespace gendil

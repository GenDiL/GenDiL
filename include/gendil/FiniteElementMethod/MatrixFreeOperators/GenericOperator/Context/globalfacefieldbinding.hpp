// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

#include <type_traits>

namespace gendil
{

template<class FacePart, class MinusFieldCellSpace, class PlusFieldCellSpace>
struct InteriorFaceFieldBinding
{
   using face_part_type = FacePart;
   using face_mesh_type = typename FacePart::face_mesh_type;
   using minus_finite_element_space_type = MinusFieldCellSpace;
   using plus_finite_element_space_type = PlusFieldCellSpace;

   FacePart face_part;
   MinusFieldCellSpace minus_space;
   PlusFieldCellSpace plus_space;

   GENDIL_HOST_DEVICE
   constexpr const FacePart& GetInteriorFacePart() const
   {
      return face_part;
   }

   GENDIL_HOST_DEVICE
   constexpr const face_mesh_type& GetFaceMesh() const
   {
      return face_part.face_mesh;
   }

   GENDIL_HOST_DEVICE
   constexpr const MinusFieldCellSpace& GetMinusFiniteElementSpace() const
   {
      return minus_space;
   }

   GENDIL_HOST_DEVICE
   constexpr const PlusFieldCellSpace& GetPlusFiniteElementSpace() const
   {
      return plus_space;
   }
};

template<class FacePart, class FieldCellSpace>
struct BoundaryFaceFieldBinding
{
   using face_part_type = FacePart;
   using face_mesh_type = typename FacePart::face_mesh_type;
   using finite_element_space_type = FieldCellSpace;

   FacePart face_part;
   FieldCellSpace cell_space;

   GENDIL_HOST_DEVICE
   constexpr const FacePart& GetBoundaryFacePart() const
   {
      return face_part;
   }

   GENDIL_HOST_DEVICE
   constexpr const face_mesh_type& GetFaceMesh() const
   {
      return face_part.face_mesh;
   }

   GENDIL_HOST_DEVICE
   constexpr const FieldCellSpace& GetMinusFiniteElementSpace() const
   {
      return cell_space;
   }
};

template<class T>
struct is_interior_face_field_binding : std::false_type {};

template<class FacePart, class MinusSpace, class PlusSpace>
struct is_interior_face_field_binding<
   InteriorFaceFieldBinding<FacePart, MinusSpace, PlusSpace>>
   : std::true_type {};

template<class T>
inline constexpr bool is_interior_face_field_binding_v =
   is_interior_face_field_binding<std::remove_cvref_t<T>>::value;

template<class T>
struct is_boundary_face_field_binding : std::false_type {};

template<class FacePart, class Space>
struct is_boundary_face_field_binding<
   BoundaryFaceFieldBinding<FacePart, Space>> : std::true_type {};

template<class T>
inline constexpr bool is_boundary_face_field_binding_v =
   is_boundary_face_field_binding<std::remove_cvref_t<T>>::value;

template<class T>
inline constexpr bool is_face_field_binding_v =
   is_interior_face_field_binding_v<T> ||
   is_boundary_face_field_binding_v<T>;

template<class T>
struct is_same_space_interior_face_field_binding : std::false_type {};

template<class FacePart, class MinusSpace, class PlusSpace>
struct is_same_space_interior_face_field_binding<
   InteriorFaceFieldBinding<FacePart, MinusSpace, PlusSpace>>
   : std::bool_constant<std::is_same_v<MinusSpace, PlusSpace>> {};

template<class T>
inline constexpr bool is_same_space_interior_face_field_binding_v =
   is_same_space_interior_face_field_binding<std::remove_cvref_t<T>>::value;

template<class T>
struct is_two_space_interior_face_field_binding : std::false_type {};

template<class FacePart, class MinusSpace, class PlusSpace>
struct is_two_space_interior_face_field_binding<
   InteriorFaceFieldBinding<FacePart, MinusSpace, PlusSpace>>
   : std::bool_constant<!std::is_same_v<MinusSpace, PlusSpace>> {};

template<class T>
inline constexpr bool is_two_space_interior_face_field_binding_v =
   is_two_space_interior_face_field_binding<std::remove_cvref_t<T>>::value;

template<class T>
struct requires_two_sided_face_qdata
   : is_two_space_interior_face_field_binding<std::remove_cvref_t<T>> {};

template<class T>
inline constexpr bool requires_two_sided_face_qdata_v =
   requires_two_sided_face_qdata<std::remove_cvref_t<T>>::value;

template<class T>
inline constexpr bool supports_one_sided_face_qdata_v =
   !requires_two_sided_face_qdata_v<std::remove_cvref_t<T>>;

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "orientation.hpp"

namespace gendil{

template<class MinusView, class PlusView>
struct GlobalFaceInfo {
   using minus_side_type = MinusView;
   using plus_side_type  = PlusView;

   GENDIL_HOST_DEVICE
   const minus_side_type& minus_side() const { return minus; }
   GENDIL_HOST_DEVICE
   const plus_side_type&  plus_side()  const { return plus;  }

   GENDIL_HOST_DEVICE
   constexpr auto is_boundary() const
   {
      return minus_side().is_boundary() || plus_side().is_boundary();
   }
   GENDIL_HOST_DEVICE
   constexpr auto is_conforming() const
   {
      return minus_side().is_conforming() && plus_side().is_conforming();
   }
   GENDIL_HOST_DEVICE
   constexpr auto get_reference_normal() const
   {
      return minus_side().get_reference_normal();
   }

   MinusView minus;
   PlusView  plus;
};

template<class CFV>
concept CellFaceView =
   requires(const CFV& v, const Point<CFV::dim>& xf) {
      std::integral_constant<Integer, CFV::dim>{};
      std::bool_constant<CFV::is_conforming>{};

      { v.get_cell_index() } -> std::convertible_to<GlobalIndex>;
      { v.get_orientation() } -> std::convertible_to<typename CFV::orientation_type>;
      { v.get_reference_normal() };
   };

template <int Dim>
struct ConformingFaceMap {
   static constexpr bool is_conforming = true;

   GENDIL_HOST_DEVICE
   Point<Dim> map_reference_to_face_coordinates(const Point<Dim>& p) const { return p; }

   GENDIL_HOST_DEVICE
   Point<Dim> map_reference_to_face_coordinates_1d(const Point<1>& p) const { return p; }

   GENDIL_HOST_DEVICE
   Real measure() const { return 1.0; }
};

template <int Dim>
struct NonconformingHyperCubeFaceMap
{
   static constexpr bool is_conforming = false;
   Point<Dim> origin{};
   std::array<Real, Dim> size{};

   GENDIL_HOST_DEVICE
   auto map_reference_to_face_coordinates(const Point<Dim>& p) const
   {
      return origin + size * p;
   }

   template < Integer d >
   GENDIL_HOST_DEVICE
   auto map_reference_to_face_coordinates_1d(const Point<1>& p) const
   {
      static_assert(
         d < Dim,
         "Dimension out of bounds"
      );
      return Point<1>{origin[d] + size[d] * p[d]};
   }

   GENDIL_HOST_DEVICE
   Real measure() const { return Product(size); }
};

template <
   typename LocalFaceIndex,
   typename Geometry,
   typename OrientationType,
   typename NormalType,
   typename ConformityType,
   typename BoundaryType = std::bool_constant< false >>
struct FaceView
{
   using local_face_index_type = LocalFaceIndex;
   using geometry = Geometry;
   using orientation_type = OrientationType;
   using normal_type = NormalType;
   using boundary_type = BoundaryType;
   using conformity_type = ConformityType;

   static constexpr Integer dim = geometry::geometry_dim;
   static constexpr bool is_conforming = conformity_type::is_conforming;

   GlobalIndex cell_index;
   local_face_index_type local_face_index;
   orientation_type orientation;
   normal_type normal;
   conformity_type conformity;
   boundary_type boundary;

   GENDIL_HOST_DEVICE
   GlobalIndex get_cell_index() const { return cell_index; }

   GENDIL_HOST_DEVICE
   const auto & get_orientation() const { return orientation; }

   GENDIL_HOST_DEVICE
   const auto & get_reference_normal() const { return normal; }

   GENDIL_HOST_DEVICE
   auto map_reference_to_face_coordinates(const Point<dim> & p) const
   {
      return conformity.map_reference_to_face_coordinates(p);
   }

   template < Integer DimIndex >
   GENDIL_HOST_DEVICE
   auto map_reference_to_face_coordinates_1d(const Point<1> & p) const
   {
      return conformity.template map_reference_to_face_coordinates_1d<DimIndex>(p);
   }

   GENDIL_HOST_DEVICE
   Real measure() const { return conformity.measure(); }
};

template < Integer Dim >
using IdentityOrientation = std::integral_constant< Permutation<Dim>, MakeReferencePermutation< Dim >() >;

template <
   typename geometry,
   typename minus_face_index,
   typename plus_face_index,
   typename plus_orientation_type,
   typename minus_normal_type,
   typename plus_normal_type,
   typename boundary_type = std::bool_constant< false >
>
using ConformingCellFaceView =
   GlobalFaceInfo<
      FaceView<
         minus_face_index,
         geometry,
         IdentityOrientation<geometry::geometry_dim>,
         minus_normal_type,
         ConformingFaceMap<geometry::geometry_dim>,
         boundary_type
      >,
      FaceView<
         plus_face_index,
         geometry,
         plus_orientation_type,
         plus_normal_type,
         ConformingFaceMap<geometry::geometry_dim>,
         boundary_type
      >
   >;


template < typename FaceInfo >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNeighborIndex( const FaceInfo & face_info )
{
   return face_info.neighbor_index;
}

template < typename FaceInfo >
GENDIL_HOST_DEVICE
constexpr bool IsBoundaryFace( const FaceInfo & face_info )
{
   return face_info.minus.boundary || face_info.plus.boundary;
}

template < typename FaceInfo >
GENDIL_HOST_DEVICE
constexpr auto GetReferenceNormal( const FaceInfo & face_info )
{
   return face_info.get_reference_normal();
}

}

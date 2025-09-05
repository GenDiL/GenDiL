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
   const minus_side_type& minus_side() const { return minus_; }
   GENDIL_HOST_DEVICE
   const plus_side_type&  plus_side()  const { return plus_;  }

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

   MinusView minus_;
   PlusView  plus_;
};

template<class CFV>
concept CellFaceView =
   requires(const CFV& v, const Point<CFV::dim>& xf) {
      requires std::is_same_v<decltype(CFV::dim), const Integer>;
      requires std::is_same_v<decltype(CFV::local_face_index), const Integer>;
      requires std::is_same_v<decltype(CFV::is_boundary),   const bool>;
      requires std::is_same_v<decltype(CFV::is_conforming), const bool>;
      std::integral_constant<Integer, CFV::dim>{};
      std::integral_constant<Integer, CFV::local_face_index>{};
      std::bool_constant<CFV::is_boundary>{};
      std::bool_constant<CFV::is_conforming>{};

      { v.get_cell_index() } -> std::convertible_to<GlobalIndex>;
      { v.get_orientation() } -> std::convertible_to<typename CFV::orientation_type>;
      { v.get_reference_normal() };
      { v.map_reference_to_face_coordinates(xf) } -> std::convertible_to<Point<CFV::dim>>;
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
   typename ConformityType >
struct FaceView
{
   using geometry = Geometry;
   using orientation_type = OrientationType;
   using normal_type = NormalType;
   using conformity_type = ConformityType;

   static constexpr Integer dim = geometry::geometry_dim;
   static constexpr Integer local_face_index = LocalFaceIndex::value;
   static constexpr bool is_boundary = false;
   static constexpr bool is_conforming = conformity_type::is_conforming;

   GlobalIndex cell_index;
   orientation_type orientation;
   normal_type normal;
   conformity_type conformity;

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
   typename GEOMETRY,
   typename LHS_FACE_INDEX,
   typename RHS_FACE_INDEX,
   typename RHS_ORIENTATION_TYPE,
   typename LHS_NORMAL_TYPE,
   typename RHS_NORMAL_TYPE
>
using ConformingCellFaceView =
   GlobalFaceInfo<
      FaceView<
         LHS_FACE_INDEX,
         GEOMETRY,
         IdentityOrientation<GEOMETRY::geometry_dim>,
         LHS_NORMAL_TYPE,
         ConformingFaceMap<GEOMETRY::geometry_dim>
      >,
      FaceView<
         RHS_FACE_INDEX,
         GEOMETRY,
         RHS_ORIENTATION_TYPE,
         RHS_NORMAL_TYPE,
         ConformingFaceMap<GEOMETRY::geometry_dim>
      >
   >;

// TODO Remove
// !FIXME: This assumes conforming and same topologies on each side
template <
   Integer LocalFaceIndex,
   typename Geometry,
   typename ConformityType,
   typename OrientationType,
   typename BoundaryType,
   typename NormalType >
struct FaceConnectivity
{
   using geometry = Geometry;
   using conformity_type = ConformityType;
   using orientation_type = OrientationType;
   using boundary_type = BoundaryType;
   using normal_type = NormalType;

   static constexpr Integer dim = geometry::geometry_dim;
   static constexpr Integer local_face_index = LocalFaceIndex;
   static constexpr Integer neighbor_local_face_index = LocalFaceIndex < dim ? LocalFaceIndex + dim : LocalFaceIndex - dim; // TODO: This feels like magic / Only true for hypercubes => Should be Geometry function

   GlobalIndex neighbor_index;
   conformity_type conformity;
   orientation_type orientation;
   boundary_type boundary;
   normal_type normal;
};

// TODO: Specialize for FaceConnectivity?
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
   return face_info.boundary;
}

template < typename FaceInfo >
GENDIL_HOST_DEVICE
constexpr auto GetReferenceNormal( const FaceInfo & face_info )
{
   return face_info.normal;
}

}

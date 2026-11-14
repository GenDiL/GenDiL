// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <array>
#include <concepts>
#include <type_traits>
#include <utility>

#include "Orientations/identityorientation.hpp"
#include "gendil/Meshes/Geometries/point.hpp"
#include "gendil/Utilities/debug.hpp"

namespace gendil{

template<class MinusView, class PlusView,
         class NormalOrientation = std::integral_constant<int, 1>>
struct GlobalFaceInfo {
   using minus_side_type = MinusView;
   using plus_side_type  = PlusView;
   using normal_orientation_type = NormalOrientation;

   GENDIL_HOST_DEVICE
   const minus_side_type& MinusSide() const { return minus; }
   GENDIL_HOST_DEVICE
   const plus_side_type&  PlusSide()  const { return plus;  }

   GENDIL_HOST_DEVICE
   constexpr auto IsBoundary() const
   {
      return MinusSide().IsBoundary() || PlusSide().IsBoundary();
   }
   GENDIL_HOST_DEVICE
   constexpr auto IsConforming() const
   {
      return MinusSide().IsConforming() && PlusSide().IsConforming();
   }
   GENDIL_HOST_DEVICE
   constexpr auto GetReferenceNormal() const
   {
      return MinusSide().GetReferenceNormal();
   }

   /// Global records store the canonical trace sides and use s = +1:
   /// the shared normal is MinusSide()'s outward normal.
   GENDIL_HOST_DEVICE
   constexpr auto GetNormalOrientation() const
   {
      return normal_orientation;
   }

   MinusView minus;
   PlusView  plus;
   [[no_unique_address]] NormalOrientation normal_orientation{};
};

/**
 * Cell-owned face information keeps current/neighbor storage independent of
 * the shared facet orientation. Canonical minus/plus access is available only
 * when that orientation is known at compile time.
 */
template<class CurrentView, class NeighborView,
         class NormalOrientation = std::integral_constant<int, 1>>
struct LocalFaceInfo
{
   using current_side_type = CurrentView;
   using neighbor_side_type = NeighborView;
   using normal_orientation_type = NormalOrientation;

   GENDIL_HOST_DEVICE
   constexpr const current_side_type& CurrentSide() const { return current; }
   GENDIL_HOST_DEVICE
   constexpr const neighbor_side_type& NeighborSide() const { return neighbor; }

   GENDIL_HOST_DEVICE
   constexpr auto IsBoundary() const
   {
      return CurrentSide().IsBoundary() || NeighborSide().IsBoundary();
   }
   GENDIL_HOST_DEVICE
   constexpr auto IsConforming() const
   {
      return CurrentSide().IsConforming() && NeighborSide().IsConforming();
   }
   GENDIL_HOST_DEVICE
   constexpr auto GetReferenceNormal() const
   {
      return CurrentSide().GetReferenceNormal();
   }

   /// n_F = s * n_current; the stored endpoints retain their identities.
   GENDIL_HOST_DEVICE
   constexpr auto GetNormalOrientation() const { return normal_orientation; }

   CurrentView current;
   NeighborView neighbor;
   [[no_unique_address]] NormalOrientation normal_orientation{};
};

// Keep the same direct members and aggregate initialization as the runtime
// record. An alias-bearing base would add an aggregate initialization element.
template<class CurrentView, class NeighborView, class NormalOrientation>
   requires requires { std::integral_constant<int, NormalOrientation::value>{}; }
struct LocalFaceInfo<CurrentView, NeighborView, NormalOrientation>
{
   static_assert(NormalOrientation::value == 1 || NormalOrientation::value == -1,
      "A facet normal orientation must be +1 or -1.");
   using current_side_type = CurrentView;
   using neighbor_side_type = NeighborView;
   using normal_orientation_type = NormalOrientation;
   using minus_side_type = std::conditional_t<NormalOrientation::value == 1,
      CurrentView, NeighborView>;
   using plus_side_type = std::conditional_t<NormalOrientation::value == 1,
      NeighborView, CurrentView>;

   GENDIL_HOST_DEVICE
   constexpr const current_side_type& CurrentSide() const { return current; }
   GENDIL_HOST_DEVICE
   constexpr const neighbor_side_type& NeighborSide() const { return neighbor; }
   GENDIL_HOST_DEVICE
   constexpr const minus_side_type& MinusSide() const
   {
      if constexpr (NormalOrientation::value == 1) return current;
      else return neighbor;
   }
   GENDIL_HOST_DEVICE
   constexpr const plus_side_type& PlusSide() const
   {
      if constexpr (NormalOrientation::value == 1) return neighbor;
      else return current;
   }

   GENDIL_HOST_DEVICE
   constexpr auto IsBoundary() const
   {
      return CurrentSide().IsBoundary() || NeighborSide().IsBoundary();
   }
   GENDIL_HOST_DEVICE
   constexpr auto IsConforming() const
   {
      return CurrentSide().IsConforming() && NeighborSide().IsConforming();
   }
   GENDIL_HOST_DEVICE
   constexpr auto GetReferenceNormal() const
   {
      return CurrentSide().GetReferenceNormal();
   }

   /// n_F = s * n_current; canonical access does not remap reference geometry.
   GENDIL_HOST_DEVICE
   constexpr auto GetNormalOrientation() const { return normal_orientation; }

   CurrentView current;
   NeighborView neighbor;
   [[no_unique_address]] NormalOrientation normal_orientation{};
};

// Numerical local-facet helpers receive the face after its runtime sign has
// been dispatched. Derive the sign from that face instead of a separate argument.
template<class Face>
inline constexpr int static_face_normal_orientation_v = [] {
   using Orientation = std::remove_cvref_t<decltype(
      std::declval<const Face&>().GetNormalOrientation())>;
   constexpr bool is_static = requires {
      std::integral_constant<int, Orientation::value>{};
   };
   static_assert(is_static,
      "Local facet evaluation requires a compile-time normal orientation. "
      "Dispatch the face with WithLocalFacetNormalOrientation first.");
   if constexpr (is_static)
   {
      static_assert(Orientation::value == 1 || Orientation::value == -1,
         "A facet normal orientation must be +1 or -1.");
      return int(Orientation::value);
   }
   else
      return 0;
}();

// Copy the endpoint descriptors with the sign selected by local dispatch.
// Require the exact built-in type: a derived face may carry additional metadata
// and must provide its own overload in its namespace to preserve that data.
template<int Sign, class Face>
   requires std::same_as<Face, LocalFaceInfo<
      typename Face::current_side_type, typename Face::neighbor_side_type,
      typename Face::normal_orientation_type>>
GENDIL_HOST_DEVICE
constexpr auto WithStaticNormalOrientation(const Face& face)
{
   static_assert(Sign == 1 || Sign == -1,
      "A facet normal orientation must be +1 or -1.");
   return LocalFaceInfo<typename Face::current_side_type,
      typename Face::neighbor_side_type, std::integral_constant<int, Sign>>{
         face.CurrentSide(), face.NeighborSide(), {}};
}

// Invoke the callback with a face whose normal orientation is a static sign.
// Stored endpoints remain current/neighbor. Runtime faces are rebound before
// interpolation and quadrature; already-static faces pass through by reference.
// The callback and any views borrowing the rebound face must finish here.
template<class Face, class Function>
GENDIL_HOST_DEVICE
void WithLocalFacetNormalOrientation(const Face& face, Function&& function)
{
   const auto orientation = face.GetNormalOrientation();
   using Orientation = std::remove_cvref_t<decltype(orientation)>;
   if constexpr (requires { Orientation::value; })
   {
      static_assert(static_face_normal_orientation_v<Face> == 1 ||
                    static_face_normal_orientation_v<Face> == -1);
      function(face);
   }
   else
   {
      GENDIL_ASSERT(orientation == 1 || orientation == -1,
         "A facet normal orientation must be +1 or -1.");
      constexpr bool can_rebind = requires {
         WithStaticNormalOrientation<1>(face);
         WithStaticNormalOrientation<-1>(face);
      };
      static_assert(can_rebind,
         "Runtime local face information must support "
         "WithStaticNormalOrientation<Sign>(face). Custom face types must "
         "provide an overload in their namespace that preserves their data.");
      if constexpr (can_rebind)
      {
         if (orientation == 1)
         {
            const auto& static_face = WithStaticNormalOrientation<1>(face);
            static_assert(static_face_normal_orientation_v<decltype(static_face)> == 1,
               "Rebound face normal orientation must match the selected sign.");
            function(static_face);
         }
         else
         {
            const auto& static_face = WithStaticNormalOrientation<-1>(face);
            static_assert(static_face_normal_orientation_v<decltype(static_face)> == -1,
               "Rebound face normal orientation must match the selected sign.");
            function(static_face);
         }
      }
   }
}

template<class CFV>
concept CellFaceView =
   requires(const CFV& v, const Point<CFV::dim>& xf) {
      std::integral_constant<Integer, CFV::dim>{};
      std::bool_constant<CFV::is_conforming>{};

      { v.GetCellIndex() } -> std::convertible_to<GlobalIndex>;
      { v.GetOrientation() } -> std::convertible_to<typename CFV::orientation_type>;
      { v.GetReferenceNormal() };
   };

template <Integer Dim>
struct ConformingFaceMap {
   static constexpr bool is_conforming = true;

   GENDIL_HOST_DEVICE
   Point<Dim> MapReferenceToFaceCoordinates(const Point<Dim>& p) const { return p; }

   GENDIL_HOST_DEVICE
   Point<Dim> MapReferenceToFaceCoordinates1d(const Point<1>& p) const { return p; }

   GENDIL_HOST_DEVICE
   Real Measure() const { return 1.0; }
};

template <Integer Dim>
struct NonconformingHyperCubeFaceMap
{
   static constexpr bool is_conforming = false;
   Point<Dim> origin{};
   std::array<Real, Dim> size{};

   GENDIL_HOST_DEVICE
   auto MapReferenceToFaceCoordinates(const Point<Dim>& p) const
   {
      return origin + size * p;
   }

   template < Integer d >
   GENDIL_HOST_DEVICE
   auto MapReferenceToFaceCoordinates1d(const Point<1>& p) const
   {
      static_assert(
         d < Dim,
         "Dimension out of bounds"
      );
      return Point<1>{origin[d] + size[d] * p[0]};
   }

   GENDIL_HOST_DEVICE
   Real Measure() const { return Product(size); }
};

template <
   typename LocalFaceIndex,
   typename Geometry,
   typename OrientationType,
   typename NormalType,
   typename ConformityType,
   typename BoundaryType = std::bool_constant< false >,
   typename NormalOrientation = Empty>
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
   // Only stored local connectivity uses this optional incidence metadata.
   [[no_unique_address]] NormalOrientation normal_orientation{};

   GENDIL_HOST_DEVICE
   GlobalIndex GetCellIndex() const { return cell_index; }

   GENDIL_HOST_DEVICE
   const auto & GetOrientation() const { return orientation; }

   GENDIL_HOST_DEVICE
   const auto & GetReferenceNormal() const { return normal; }

   GENDIL_HOST_DEVICE
   bool IsBoundary() const { return static_cast< bool >( boundary ); }

   GENDIL_HOST_DEVICE
   constexpr bool IsConforming() const { return is_conforming; }

   GENDIL_HOST_DEVICE
   auto MapReferenceToFaceCoordinates(const Point<dim> & p) const
   {
      return conformity.MapReferenceToFaceCoordinates(p);
   }

   template < Integer DimIndex >
   GENDIL_HOST_DEVICE
   auto MapReferenceToFaceCoordinates1d(const Point<1> & p) const
   {
      return conformity.template MapReferenceToFaceCoordinates1d<DimIndex>(p);
   }

   GENDIL_HOST_DEVICE
   Real Measure() const { return conformity.Measure(); }
};

template <
   typename geometry,
   typename minus_face_index,
   typename plus_face_index,
   typename plus_orientation_type,
   typename minus_normal_type,
   typename plus_normal_type,
   typename boundary_type = std::bool_constant< false >,
   typename normal_orientation_type = std::integral_constant<int, 1>
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
      >,
      normal_orientation_type
   >;

template <
   typename Geometry,
   typename CurrentFaceIndex,
   typename NeighborFaceIndex,
   typename NeighborOrientation,
   typename CurrentNormal,
   typename NeighborNormal,
   typename Boundary = std::bool_constant< false >,
   typename NormalOrientation = std::integral_constant<int, 1>
>
using LocalConformingCellFaceView = LocalFaceInfo<
   typename ConformingCellFaceView<Geometry, CurrentFaceIndex, NeighborFaceIndex,
      NeighborOrientation, CurrentNormal, NeighborNormal, Boundary>::minus_side_type,
   typename ConformingCellFaceView<Geometry, CurrentFaceIndex, NeighborFaceIndex,
      NeighborOrientation, CurrentNormal, NeighborNormal, Boundary>::plus_side_type,
   NormalOrientation>;


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
   return face_info.IsBoundary();
}

template < typename FaceInfo >
GENDIL_HOST_DEVICE
constexpr auto GetReferenceNormal( const FaceInfo & face_info )
{
   return face_info.GetReferenceNormal();
}

}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>
#include <tuple>
#include <type_traits>

using namespace gendil;

namespace
{

struct SerialKernel
{
};

using Shape1D = GaussLobattoLegendreShapeFunctions<1>;
using Points1D = GaussLegendrePoints<2>;
using DofToQuad1D = CachedDofToQuad<Shape1D, Points1D>;
using LocalFaceQData = std::tuple<DofToQuad1D, DofToQuad1D>;
using SparseFaceQData =
   std::tuple<Empty, LocalFaceQData, Empty, Empty>;

using Geometry = HyperCube<2>;
using FaceIndex = std::integral_constant<Integer, 1>;
using Orientation = IdentityOrientation<2>;
using Normal = Point<2>;
using ConformingFace = FaceView<
   FaceIndex,
   Geometry,
   Orientation,
   Normal,
   ConformingFaceMap<2>>;
using NonconformingFace = FaceView<
   FaceIndex,
   Geometry,
   Orientation,
   Normal,
   NonconformingHyperCubeFaceMap<2>>;

bool CheckClose(
   const Real value,
   const Real reference,
   const char* label,
   const Real tolerance = 1e-14)
{
   const Real diff = std::abs(value - reference);
   const Real scale = std::max(Real{1.0}, std::abs(reference));
   if (diff > tolerance * scale)
   {
      std::cout
         << "FAILED: " << label
         << " value=" << value
         << " reference=" << reference
         << " diff=" << diff << '\n';
      return false;
   }
   return true;
}

SparseFaceQData MakeSparseFaceQData()
{
   return SparseFaceQData{
      Empty{},
      LocalFaceQData{DofToQuad1D{}, DofToQuad1D{}},
      Empty{},
      Empty{}};
}

ConformingFace MakeConformingFace()
{
   return ConformingFace{
      0,
      FaceIndex{},
      Orientation{},
      Normal{1.0, 0.0},
      ConformingFaceMap<2>{},
      typename ConformingFace::boundary_type{}};
}

NonconformingFace MakeNonconformingFace()
{
   return NonconformingFace{
      0,
      FaceIndex{},
      Orientation{},
      Normal{1.0, 0.0},
      NonconformingHyperCubeFaceMap<2>{
         Point<2>{0.25, 0.5},
         std::array<Real, 2>{0.5, 0.25}},
      typename NonconformingFace::boundary_type{}};
}

bool TestMapReferenceToFaceCoordinates1d()
{
   NonconformingHyperCubeFaceMap<2> map{
      Point<2>{0.125, 0.25},
      std::array<Real, 2>{0.5, 0.75}};

   const Point<1> p{0.4};
   const auto x = map.template MapReferenceToFaceCoordinates1d<0>(p);
   const auto y = map.template MapReferenceToFaceCoordinates1d<1>(p);

   bool success = true;
   success = CheckClose(x[0], 0.125 + 0.5 * p[0], "mapped x coordinate") &&
             success;
   success = CheckClose(y[0], 0.25 + 0.75 * p[0], "mapped y coordinate") &&
             success;
   return success;
}

bool TestConformingFacetQuadDataReference()
{
   auto face_quad_data = MakeSparseFaceQData();
   const auto& expected = std::get<1>(face_quad_data);
   auto&& selected = GetFacetQuadData(face_quad_data, MakeConformingFace());

   static_assert(
      std::is_lvalue_reference_v<decltype(selected)>,
      "Conforming facet qdata must be returned by reference.");

   return &selected == &expected;
}

bool TestEmptyFacetQuadDataTrait()
{
   static_assert(IsEmptyFacetQuadData_v<Empty>);
   static_assert(IsEmptyFacetQuadData_v<const Empty&>);
   static_assert(!IsEmptyFacetQuadData_v<LocalFaceQData>);

   using WrongSlot = std::tuple_element_t<0, SparseFaceQData>;
   using CorrectSlot = std::tuple_element_t<1, SparseFaceQData>;
   static_assert(IsEmptyFacetQuadData_v<WrongSlot>);
   static_assert(!IsEmptyFacetQuadData_v<CorrectSlot>);

   return true;
}

bool TestNonconformingFacetQuadDataMappingAndLifetime()
{
   auto face_quad_data = MakeSparseFaceQData();
   auto mapped_qd = GetFacetQuadData(face_quad_data, MakeNonconformingFace());

   using MappedQData = std::remove_cvref_t<decltype(mapped_qd)>;
   using FirstMappedDofToQuad = std::tuple_element_t<0, MappedQData>;
   static_assert(
      !std::is_reference_v<typename FirstMappedDofToQuad::face_type>,
      "Nonconforming mapped qdata face_type must be an owned value type.");
   static_assert(
      !std::is_reference_v<decltype(std::declval<FirstMappedDofToQuad>().face)>,
      "Nonconforming mapped qdata must not retain a face reference.");

   const Real q0 = Points1D::GetCoord(0);
   const Real mapped_x = 0.25 + 0.5 * q0;
   const Real mapped_y = 0.5 + 0.25 * q0;

   bool success = true;
   success =
      CheckClose(
         std::get<0>(mapped_qd).values(0, 1),
         mapped_x,
         "mapped basis value in x") &&
      success;
   success =
      CheckClose(
         std::get<1>(mapped_qd).values(0, 1),
         mapped_y,
         "mapped basis value in y") &&
      success;
   success =
      CheckClose(
         std::get<0>(mapped_qd).gradients(0, 1),
         1.0,
         "unscaled mapped reference gradient") &&
      success;
   return success;
}

bool TestInterpolateValuesUsesFacetQuadData()
{
   auto face_quad_data = MakeSparseFaceQData();
   auto face = MakeNonconformingFace();

   SerialRecursiveArray<Real, 2, 2> dofs{};
   dofs(0, 0) = 0.0;
   dofs(1, 0) = 1.0;
   dofs(0, 1) = 10.0;
   dofs(1, 1) = 11.0;

   SerialKernel kernel{};
   auto values = InterpolateValues(kernel, face, face_quad_data, dofs);

   const Real q0 = Points1D::GetCoord(0);
   const Real mapped_x = 0.25 + 0.5 * q0;
   const Real mapped_y = 0.5 + 0.25 * q0;
   const Real expected = mapped_x + 10.0 * mapped_y;

   return CheckClose(values(0, 0), expected, "InterpolateValues mapped trace");
}

} // namespace

int main()
{
   bool success = true;
   success = TestMapReferenceToFaceCoordinates1d() && success;
   success = TestConformingFacetQuadDataReference() && success;
   success = TestEmptyFacetQuadDataTrait() && success;
   success = TestNonconformingFacetQuadDataMappingAndLifetime() && success;
   success = TestInterpolateValuesUsesFacetQuadData() && success;

   if (!success)
   {
      return 1;
   }

   std::cout << "All facet qdata accessor tests passed.\n";
   return 0;
}

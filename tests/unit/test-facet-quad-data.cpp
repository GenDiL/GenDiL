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
using LocalFaceQData =
   TensorProductDofToQuad<DofToQuad1D, DofToQuad1D>;
using SparseFaceQData =
   std::tuple<Empty, LocalFaceQData, Empty, Empty>;
using StaticSquareFaceQData0 =
   TensorProductData<ZeroPoint, Points1D>;
using StaticSquareFaceQData1 =
   TensorProductData<Points1D, OnePoint>;
using StaticCubeFaceQData0 =
   TensorProductData<ZeroPoint, Points1D, Points1D>;
using StaticCubeFaceQData2 =
   TensorProductData<Points1D, Points1D, OnePoint>;

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
      MakeTensorProductData(
         DofToQuad1D{},
         DofToQuad1D{}),
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
   using Actual = decltype(GetFacetQuadData(
      std::declval<const SparseFaceQData&>(),
      std::declval<const ConformingFace&>()));
   static_assert(std::same_as<Actual, const LocalFaceQData&>);

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
   using FirstMappedDofToQuad =
      tensor_product_entry_t<0, MappedQData>;
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
         GetTensorProductEntry<0>(mapped_qd).values(0, 1),
         mapped_x,
         "mapped basis value in x") &&
      success;
   success =
      CheckClose(
         GetTensorProductEntry<1>(mapped_qd).values(0, 1),
         mapped_y,
         "mapped basis value in y") &&
      success;
   success =
      CheckClose(
         GetTensorProductEntry<0>(mapped_qd).gradients(0, 1),
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

NonconformingFace MakeEmbeddedMapFace(Point<2> origin, std::array<Real, 2> size)
{
   return NonconformingFace{
      0,
      FaceIndex{},
      Orientation{},
      Normal{1.0, 0.0},
      NonconformingHyperCubeFaceMap<2>{origin, size},
      typename NonconformingFace::boundary_type{}};
}

template<class LocalQData>
auto MakeSparseStaticQData(LocalQData qdata)
{
   return std::tuple{Empty{}, qdata, Empty{}, Empty{}};
}

bool TestAffineMappedQDataFixedCoordinateFirst()
{
   auto face_quad_data =
      MakeSparseStaticQData(
         MakeTensorProductData(ZeroPoint{}, Points1D{}));
   auto face =
      MakeEmbeddedMapFace(
         Point<2>{0.0, 0.25},
         std::array<Real, 2>{1.0, 0.5});
   auto mapped_qd = GetFacetQuadData(face_quad_data, face);

   const Integer q = 1;
   const Real y_ref = Points1D::GetCoord(q);
   const TensorIndex<2> qi{GlobalIndex{0}, static_cast<GlobalIndex>(q)};

   bool success = true;
   success =
      CheckClose(
         GetCoord<0>(mapped_qd, 0),
         0.0,
         "fixed ZeroPoint coordinate remains unchanged") &&
      success;
   success =
      CheckClose(
         GetCoord<1>(mapped_qd, q),
         0.25 + 0.5 * y_ref,
         "mapped tangential coordinate with fixed coordinate first") &&
      success;
   success =
      CheckClose(
         GetWeight(qi, mapped_qd),
         Points1D::GetWeight(q),
         "mapped affine qdata base weight with fixed coordinate first") &&
      success;
   return success;
}

bool TestAffineMappedQDataFixedCoordinateLast()
{
   auto face_quad_data =
      MakeSparseStaticQData(
         MakeTensorProductData(Points1D{}, OnePoint{}));
   auto face =
      MakeEmbeddedMapFace(
         Point<2>{0.125, 0.0},
         std::array<Real, 2>{0.25, 1.0});
   auto mapped_qd = GetFacetQuadData(face_quad_data, face);

   const Integer q = 0;
   const Real x_ref = Points1D::GetCoord(q);
   const TensorIndex<2> qi{static_cast<GlobalIndex>(q), GlobalIndex{0}};

   bool success = true;
   success =
      CheckClose(
         GetCoord<0>(mapped_qd, q),
         0.125 + 0.25 * x_ref,
         "mapped tangential coordinate with fixed coordinate last") &&
      success;
   success =
      CheckClose(
         GetCoord<1>(mapped_qd, 0),
         1.0,
         "fixed OnePoint coordinate remains unchanged") &&
      success;
   success =
      CheckClose(
         GetWeight(qi, mapped_qd),
         Points1D::GetWeight(q),
         "mapped affine qdata base weight with fixed coordinate last") &&
      success;
   return success;
}

bool TestAffineMappedQData3D()
{
   using Geometry3D = HyperCube<3>;
   using FaceIndex3D = std::integral_constant<Integer, 1>;
   using Orientation3D = IdentityOrientation<3>;
   using Normal3D = Point<3>;
   using NonconformingFace3D = FaceView<
      FaceIndex3D,
      Geometry3D,
      Orientation3D,
      Normal3D,
      NonconformingHyperCubeFaceMap<3>>;

   auto make_face =
      [] (Point<3> origin, std::array<Real, 3> size)
      {
         return NonconformingFace3D{
            0,
            FaceIndex3D{},
            Orientation3D{},
            Normal3D{1.0, 0.0, 0.0},
            NonconformingHyperCubeFaceMap<3>{origin, size},
            typename NonconformingFace3D::boundary_type{}};
      };

   auto qd0 = MakeSparseStaticQData(
      MakeTensorProductData(
         ZeroPoint{},
         Points1D{},
         Points1D{}));
   auto face0 =
      make_face(
         Point<3>{0.0, 0.1, 0.2},
         std::array<Real, 3>{1.0, 0.25, 0.5});
   auto mapped0 = GetFacetQuadData(qd0, face0);

   auto qd2 = MakeSparseStaticQData(
      MakeTensorProductData(
         Points1D{},
         Points1D{},
         OnePoint{}));
   auto face2 =
      make_face(
         Point<3>{0.2, 0.3, 0.0},
         std::array<Real, 3>{0.5, 0.25, 1.0});
   auto mapped2 = GetFacetQuadData(qd2, face2);

   const Integer q0 = 0;
   const Integer q1 = 1;
   const TensorIndex<3> qi{
      GlobalIndex{0},
      static_cast<GlobalIndex>(q0),
      static_cast<GlobalIndex>(q1)};

   bool success = true;
   success =
      CheckClose(
         GetCoord<0>(mapped0, 0),
         0.0,
         "3D fixed ZeroPoint coordinate remains unchanged") &&
      success;
   success =
      CheckClose(
         GetCoord<1>(mapped0, q0),
         0.1 + 0.25 * Points1D::GetCoord(q0),
         "3D first tangential coordinate maps") &&
      success;
   success =
      CheckClose(
         GetCoord<2>(mapped0, q1),
         0.2 + 0.5 * Points1D::GetCoord(q1),
         "3D second tangential coordinate maps") &&
      success;
   success =
      CheckClose(
         GetWeight(qi, mapped0),
         Points1D::GetWeight(q0) * Points1D::GetWeight(q1),
         "3D mapped qdata base weight") &&
      success;
   success =
      CheckClose(
         GetCoord<2>(mapped2, 0),
         1.0,
         "3D fixed OnePoint coordinate remains unchanged") &&
      success;
   return success;
}

bool TestNestedNonconformingQDataRootOffset()
{
   using Geometry3D = HyperCube<3>;
   using FaceIndex3D = std::integral_constant<Integer, 1>;
   using Orientation3D = IdentityOrientation<3>;
   using Normal3D = Point<3>;
   using NonconformingFace3D = FaceView<
      FaceIndex3D,
      Geometry3D,
      Orientation3D,
      Normal3D,
      NonconformingHyperCubeFaceMap<3>>;

   const NonconformingFace3D face{
      0,
      FaceIndex3D{},
      Orientation3D{},
      Normal3D{1.0, 0.0, 0.0},
      NonconformingHyperCubeFaceMap<3>{
         Point<3>{0.1, 0.2, 0.3},
         std::array<Real, 3>{0.25, 0.5, 0.75}},
      typename NonconformingFace3D::boundary_type{}};

   const auto nested_dof_qdata = MakeTensorProductData(
      MakeTensorProductData(DofToQuad1D{}),
      MakeTensorProductData(DofToQuad1D{}, DofToQuad1D{}));
   using ExpectedDofQData = TensorProductData<
      TensorProductData<
         NonconformingDofToQuad<
            Shape1D,
            Points1D,
            NonconformingFace3D,
            0>>,
      TensorProductData<
         NonconformingDofToQuad<
            Shape1D,
            Points1D,
            NonconformingFace3D,
            1>,
         NonconformingDofToQuad<
            Shape1D,
            Points1D,
            NonconformingFace3D,
            2>>>;
   using ActualDofQData = decltype(
      MakeNestedNonconformingDofToQuadData(face, nested_dof_qdata));
   static_assert(std::same_as<ActualDofQData, ExpectedDofQData>);
   const auto mapped_dof_qdata =
      MakeNestedNonconformingDofToQuadData(face, nested_dof_qdata);

   using MappedDofQData = decltype(mapped_dof_qdata);
   using FirstDofGroup = tensor_product_entry_t<0, MappedDofQData>;
   using SecondDofGroup = tensor_product_entry_t<1, MappedDofQData>;
   static_assert(is_tensor_product_data_v<FirstDofGroup>);
   static_assert(is_tensor_product_data_v<SecondDofGroup>);
   static_assert(tensor_product_entry_count_v<FirstDofGroup> == 1);
   static_assert(tensor_product_entry_count_v<SecondDofGroup> == 2);
   static_assert(std::is_same_v<
      tensor_product_entry_t<0, FirstDofGroup>,
      NonconformingDofToQuad<
         Shape1D,
         Points1D,
         NonconformingFace3D,
         0>>);
   static_assert(std::is_same_v<
      tensor_product_entry_t<0, SecondDofGroup>,
      NonconformingDofToQuad<
         Shape1D,
         Points1D,
         NonconformingFace3D,
         1>>);
   static_assert(std::is_same_v<
      tensor_product_entry_t<1, SecondDofGroup>,
      NonconformingDofToQuad<
         Shape1D,
         Points1D,
         NonconformingFace3D,
         2>>);
   static_assert(!std::is_reference_v<
      typename tensor_product_entry_t<
         0,
         FirstDofGroup>::face_type>);

   const auto nested_point_qdata = MakeTensorProductData(
      MakeTensorProductData(Points1D{}),
      MakeTensorProductData(Points1D{}, Points1D{}));
   using ExpectedPointQData = TensorProductData<
      TensorProductData<
         NonconformingMappedPointSet1D<
            Points1D,
            NonconformingFace3D,
            0>>,
      TensorProductData<
         NonconformingMappedPointSet1D<
            Points1D,
            NonconformingFace3D,
            1>,
         NonconformingMappedPointSet1D<
            Points1D,
            NonconformingFace3D,
            2>>>;
   using ActualPointQData = decltype(
      MakeNestedNonconformingMappedPointSetData(
         face,
         nested_point_qdata));
   static_assert(std::same_as<ActualPointQData, ExpectedPointQData>);
   const auto mapped_point_qdata =
      MakeNestedNonconformingMappedPointSetData(face, nested_point_qdata);

   using MappedPointQData = decltype(mapped_point_qdata);
   using FirstPointGroup = tensor_product_entry_t<0, MappedPointQData>;
   using SecondPointGroup = tensor_product_entry_t<1, MappedPointQData>;
   static_assert(is_tensor_product_data_v<FirstPointGroup>);
   static_assert(is_tensor_product_data_v<SecondPointGroup>);
   static_assert(tensor_product_entry_count_v<FirstPointGroup> == 1);
   static_assert(tensor_product_entry_count_v<SecondPointGroup> == 2);
   static_assert(std::is_same_v<
      tensor_product_entry_t<0, FirstPointGroup>,
      NonconformingMappedPointSet1D<
         Points1D,
         NonconformingFace3D,
         0>>);
   static_assert(std::is_same_v<
      tensor_product_entry_t<0, SecondPointGroup>,
      NonconformingMappedPointSet1D<
         Points1D,
         NonconformingFace3D,
         1>>);
   static_assert(std::is_same_v<
      tensor_product_entry_t<1, SecondPointGroup>,
      NonconformingMappedPointSet1D<
         Points1D,
         NonconformingFace3D,
         2>>);
   static_assert(!std::is_reference_v<decltype(
      std::declval<tensor_product_entry_t<
         0,
         FirstPointGroup>>().face)>);

   const Integer q = 1;
   const Real reference_coord = Points1D::GetCoord(q);
   const Real reference_weight = Points1D::GetWeight(q);
   const auto& mapped_dof_x = GetTensorProductEntry<0>(
      GetTensorProductEntry<0>(mapped_dof_qdata));
   const auto& mapped_dof_y = GetTensorProductEntry<0>(
      GetTensorProductEntry<1>(mapped_dof_qdata));
   const auto& mapped_dof_z = GetTensorProductEntry<1>(
      GetTensorProductEntry<1>(mapped_dof_qdata));
   const auto& mapped_point_x = GetTensorProductEntry<0>(
      GetTensorProductEntry<0>(mapped_point_qdata));
   const auto& mapped_point_y = GetTensorProductEntry<0>(
      GetTensorProductEntry<1>(mapped_point_qdata));
   const auto& mapped_point_z = GetTensorProductEntry<1>(
      GetTensorProductEntry<1>(mapped_point_qdata));

   bool success = true;
   success =
      CheckClose(
         mapped_dof_x.values(q, 1),
         0.1 + 0.25 * reference_coord,
         "nested mapped dof x coordinate") &&
      success;
   success =
      CheckClose(
         mapped_dof_y.values(q, 1),
         0.2 + 0.5 * reference_coord,
         "nested mapped dof y coordinate") &&
      success;
   success =
      CheckClose(
         mapped_dof_z.values(q, 1),
         0.3 + 0.75 * reference_coord,
         "nested mapped dof z coordinate") &&
      success;
   success =
      CheckClose(
         mapped_point_x.coord(q),
         0.1 + 0.25 * reference_coord,
         "nested mapped point x coordinate") &&
      success;
   success =
      CheckClose(
         mapped_point_y.coord(q),
         0.2 + 0.5 * reference_coord,
         "nested mapped point y coordinate") &&
      success;
   success =
      CheckClose(
         mapped_point_z.coord(q),
         0.3 + 0.75 * reference_coord,
         "nested mapped point z coordinate") &&
      success;
   success =
      CheckClose(
         mapped_dof_z.weights(q),
         reference_weight,
         "nested mapped dof weight") &&
      success;
   success =
      CheckClose(
         mapped_point_z.weights(q),
         reference_weight,
         "nested mapped point weight") &&
      success;

   const TensorIndex<3> qi{
      static_cast<GlobalIndex>(q),
      static_cast<GlobalIndex>(q),
      static_cast<GlobalIndex>(q)};
   const Real expected_weight =
      reference_weight * reference_weight * reference_weight;
   success =
      CheckClose(
         GetWeight(qi, mapped_dof_qdata),
         expected_weight,
         "nested mapped dof product weight") &&
      success;
   success =
      CheckClose(
         GetWeight(qi, mapped_point_qdata),
         expected_weight,
         "nested mapped point product weight") &&
      success;
   return success;
}

bool TestTensorProductWeight()
{
   using FlatMeshQData =
      TensorProductData<
         Points1D,
         Points1D,
         Points1D>;
   using NestedMeshQData =
      TensorProductData<
         TensorProductData<Points1D>,
         TensorProductData<
            TensorProductData<Points1D>,
            TensorProductData<Points1D>>>;

   const TensorIndex<3> qi{
      GlobalIndex{0},
      GlobalIndex{1},
      GlobalIndex{0}};
   const Real expected =
      Points1D::GetWeight(0) *
      Points1D::GetWeight(1) *
      Points1D::GetWeight(0);

   bool success = true;
   success =
      CheckClose(
         GetWeight(qi, FlatMeshQData{}),
         expected,
         "flat mesh qdata weight") &&
      success;
   success =
      CheckClose(
         GetWeight(qi, NestedMeshQData{}),
         expected,
         "nested product mesh qdata weight") &&
      success;
   return success;
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
   success = TestAffineMappedQDataFixedCoordinateFirst() && success;
   success = TestAffineMappedQDataFixedCoordinateLast() && success;
   success = TestAffineMappedQData3D() && success;
   success = TestNestedNonconformingQDataRootOffset() && success;
   success = TestTensorProductWeight() && success;

   if (!success)
   {
      return 1;
   }

   std::cout << "All facet qdata accessor tests passed.\n";
   return 0;
}

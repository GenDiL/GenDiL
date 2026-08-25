// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "cartesian-product-partition-test-helpers.hpp"

#include <array>
#include <concepts>
#include <tuple>
#include <type_traits>

using namespace gendil;
using namespace gendil::test;

namespace
{

template<class Orientation, size_t Dim>
bool CheckPermutation(
   const Orientation& orientation,
   const std::array<LocalIndex, Dim>& expected,
   const std::string& label)
{
   static_assert(orientation_dimension_v<Orientation> == Dim);
   const auto got = FlattenOrientation(orientation);
   bool success = true;
   for (size_t d = 0; d < Dim; ++d)
   {
      success = Check(got(d) == expected[d], label) && success;
   }
   return success;
}

template<Integer Dim, Integer LocalFaceIndex, size_t NumRecords>
struct MaterializedConformingInteriorConnectivity
{
   static constexpr Integer axis =
      HyperCube<Dim>::GetNormalDimensionIndex(LocalFaceIndex);
   static constexpr int sign =
      HyperCube<Dim>::GetNormalSign(LocalFaceIndex);
   static constexpr Integer plus_face =
      HyperCube<Dim>::GetOppositeFaceIndex(LocalFaceIndex);
   using geometry = HyperCube<Dim>;
   using face_info_type = ConformingCellFaceView<
      geometry,
      std::integral_constant<Integer, LocalFaceIndex>,
      std::integral_constant<Integer, plus_face>,
      Permutation<Dim>,
      CanonicalVector<Dim, axis, sign>,
      CanonicalVector<Dim, axis, -sign>>;

   struct Record
   {
      GlobalIndex minus_cell;
      GlobalIndex plus_cell;
      Permutation<Dim> plus_orientation;
   };

   std::array<Record, NumRecords> records{};

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return static_cast<GlobalIndex>(NumRecords);
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex face) const
   {
      const auto record = records[face];
      return face_info_type{
         {record.minus_cell},
         {record.plus_cell, {}, record.plus_orientation}};
   }
};

struct StaticIdentityMiddleFactorConnectivity
{
   using geometry = HyperCube<2>;
   using minus_side_type = FaceView<
      std::integral_constant<Integer, 2>,
      geometry,
      IdentityOrientation<2>,
      CanonicalVector<2, 0, 1>,
      ConformingFaceMap<2>>;
   using plus_side_type = FaceView<
      std::integral_constant<Integer, 0>,
      geometry,
      IdentityOrientation<2>,
      CanonicalVector<2, 0, -1>,
      ConformingFaceMap<2>>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return 1;
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(GlobalIndex) const
   {
      return face_info_type{
         {1, {}, {}, {}, {}, {}},
         {2, {}, {}, {}, {}, {}}};
   }
};

auto MakeStaticIdentityMiddleFactor()
{
   Cartesian2DMesh mesh(
      0.5, 0.5, 2, 2, Point<2>{0.25, -0.5});
   return MakePartition(
      MakeCellPart(mesh),
      MakeInteriorFacePart<0, 0>(
         StaticIdentityMiddleFactorConnectivity{}));
}

auto MakeAsymmetricFacedPartition()
{
   Cartesian2DMesh minus_mesh(
      0.5, 0.5, 2, 2, Point<2>{0.0, 0.0});
   Cartesian2DMesh plus_mesh(
      1.0 / 3.0, 0.5, 3, 2, Point<2>{1.0, 0.0});

   using Positive = MaterializedConformingInteriorConnectivity<2, 2, 2>;
   Positive positive{{
      typename Positive::Record{1, 2, Permutation<2>{{1, -2}}},
      typename Positive::Record{3, 5, Permutation<2>{{2, 1}}}}};

   using Negative = MaterializedConformingInteriorConnectivity<2, 1, 1>;
   Negative negative{{
      typename Negative::Record{2, 0, Permutation<2>{{-1, 2}}}}};

   using ZeroInterior =
      MaterializedConformingInteriorConnectivity<2, 3, 0>;
   using Boundary = MaterializedBoundaryConnectivity<2, 0, 2>;
   Boundary boundary{{
      typename Boundary::Record{0, 97},
      typename Boundary::Record{3, 98}}};
   using ZeroBoundary = MaterializedBoundaryConnectivity<2, 3, 0>;

   return MakePartition(
      MakeCellPart(minus_mesh),
      MakeCellPart(plus_mesh),
      MakeInteriorFacePart<0, 1>(positive),
      MakeInteriorFacePart<0, 0>(negative),
      MakeInteriorFacePart<1, 1>(ZeroInterior{}),
      MakeBoundaryFacePart<0>(boundary),
      MakeBoundaryFacePart<1>(ZeroBoundary{}));
}

bool TestAsymmetricStructure()
{
   const auto faced = MakeAsymmetricFacedPartition();
   const Cartesian1DMesh line_mesh(0.2, 5);
   const auto cell_only = MakePartition(MakeCellPart(line_mesh));

   const auto faced_first =
      MakeCartesianProductPartition(faced, cell_only);
   const auto faced_second =
      MakeCartesianProductPartition(cell_only, faced);

   using First = std::remove_cvref_t<decltype(faced_first)>;
   using Second = std::remove_cvref_t<decltype(faced_second)>;
   static_assert(First::num_cell_parts == 2);
   static_assert(Second::num_cell_parts == 2);
   static_assert(First::num_interior_face_parts == 3);
   static_assert(Second::num_interior_face_parts == 3);
   static_assert(First::num_boundary_face_parts == 2);
   static_assert(Second::num_boundary_face_parts == 2);

   using FirstPositive = std::tuple_element_t<
      0,
      typename First::interior_face_parts_type>;
   using SecondPositive = std::tuple_element_t<
      0,
      typename Second::interior_face_parts_type>;
   static_assert(
      FirstPositive::minus_cell_index == 0 &&
      FirstPositive::plus_cell_index == 1);
   static_assert(
      SecondPositive::minus_cell_index == 0 &&
      SecondPositive::plus_cell_index == 1);

   bool success = true;
   success = Check(
      std::get<0>(faced_first.CellParts()).mesh.GetNumberOfCells() == 20 &&
      std::get<1>(faced_first.CellParts()).mesh.GetNumberOfCells() == 30,
      "faced-first product cell counts use asymmetric source sizes") && success;
   success = Check(
      std::get<0>(faced_second.CellParts()).mesh.GetNumberOfCells() == 20 &&
      std::get<1>(faced_second.CellParts()).mesh.GetNumberOfCells() == 30,
      "faced-second product cell counts use asymmetric source sizes") && success;

   const auto& first_positive =
      std::get<0>(faced_first.InteriorFaceParts()).face_mesh;
   const auto& second_positive =
      std::get<0>(faced_second.InteriorFaceParts()).face_mesh;
   static_assert(std::is_copy_constructible_v<
      std::remove_cvref_t<decltype(first_positive)>>);
   static_assert(std::is_copy_constructible_v<
      std::remove_cvref_t<decltype(second_positive)>>);
   success = Check(
      first_positive.GetNumberOfFaces() == 10 &&
      second_positive.GetNumberOfFaces() == 10,
      "multiple source records are extruded over five cells") && success;

   const auto first0 = first_positive.GetGlobalFaceInfo(0);
   const auto first9 = first_positive.GetGlobalFaceInfo(9);
   success = Check(
      first0.MinusSide().GetCellIndex() == 1 &&
      first0.PlusSide().GetCellIndex() == 2 &&
      first9.MinusSide().GetCellIndex() == 19 &&
      first9.PlusSide().GetCellIndex() == 29,
      "faced-first entity ordering and side-specific strides") && success;
   success = CheckPermutation(
      first0.PlusSide().GetOrientation(),
      std::array<LocalIndex, 3>{1, -2, 3},
      "faced-first signed orientation") && success;
   using FirstMinus = typename decltype(first0)::minus_side_type;
   static_assert(std::same_as<
      typename FirstMinus::orientation_type,
      IdentityOrientation<3>>);
   static_assert(std::same_as<
      typename decltype(first0)::plus_side_type::orientation_type,
      TensorProductOrientation<Permutation<2>, IdentityOrientation<1>>>);
   static_assert(FirstMinus::local_face_index_type::value == 3);
   static_assert(FirstMinus::normal_type::index == 0);
   static_assert(FirstMinus::normal_type::sign == 1);

   const auto second0 = second_positive.GetGlobalFaceInfo(0);
   const auto second9 = second_positive.GetGlobalFaceInfo(9);
   success = Check(
      second0.MinusSide().GetCellIndex() == 5 &&
      second0.PlusSide().GetCellIndex() == 10 &&
      second9.MinusSide().GetCellIndex() == 19 &&
      second9.PlusSide().GetCellIndex() == 29,
      "faced-second entity ordering and first-factor-fastest strides") && success;
   success = CheckPermutation(
      second0.PlusSide().GetOrientation(),
      std::array<LocalIndex, 3>{1, 2, -3},
      "faced-second signed orientation") && success;
   using SecondMinus = typename decltype(second0)::minus_side_type;
   static_assert(std::same_as<
      typename SecondMinus::orientation_type,
      IdentityOrientation<3>>);
   static_assert(std::same_as<
      typename decltype(second0)::plus_side_type::orientation_type,
      TensorProductOrientation<IdentityOrientation<1>, Permutation<2>>>);
   static_assert(SecondMinus::local_face_index_type::value == 4);
   static_assert(SecondMinus::normal_type::index == 1);
   static_assert(SecondMinus::normal_type::sign == 1);

   using FirstNegativeInfo = std::remove_cvref_t<decltype(
      std::get<1>(faced_first.InteriorFaceParts()).face_mesh.
         GetGlobalFaceInfo(0))>;
   using SecondNegativeInfo = std::remove_cvref_t<decltype(
      std::get<1>(faced_second.InteriorFaceParts()).face_mesh.
         GetGlobalFaceInfo(0))>;
   static_assert(
      FirstNegativeInfo::minus_side_type::local_face_index_type::value == 1);
   static_assert(
      SecondNegativeInfo::minus_side_type::local_face_index_type::value == 2);

   success = Check(
      std::get<2>(faced_first.InteriorFaceParts()).face_mesh.
         GetNumberOfFaces() == 0 &&
      std::get<2>(faced_second.InteriorFaceParts()).face_mesh.
         GetNumberOfFaces() == 0 &&
      std::get<1>(faced_first.BoundaryFaceParts()).face_mesh.
         GetNumberOfFaces() == 0 &&
      std::get<1>(faced_second.BoundaryFaceParts()).face_mesh.
         GetNumberOfFaces() == 0,
      "explicit zero-face families remain declared") && success;

   const auto first_boundary =
      std::get<0>(faced_first.BoundaryFaceParts()).face_mesh.
         GetGlobalFaceInfo(8);
   const auto second_boundary0 =
      std::get<0>(faced_second.BoundaryFaceParts()).face_mesh.
         GetGlobalFaceInfo(4);
   const auto second_boundary1 =
      std::get<0>(faced_second.BoundaryFaceParts()).face_mesh.
         GetGlobalFaceInfo(5);
   success = Check(
      first_boundary.MinusSide().GetCellIndex() == 16 &&
      first_boundary.PlusSide().GetCellIndex() == 97 &&
      second_boundary0.MinusSide().GetCellIndex() == 4 &&
      second_boundary0.PlusSide().GetCellIndex() == 97 &&
      second_boundary1.MinusSide().GetCellIndex() == 15 &&
      second_boundary1.PlusSide().GetCellIndex() == 98,
      "boundary extrusion maps only the real minus side") && success;

   return success;
}

bool TestNestedMiddleFactorLifting()
{
   const Cartesian1DMesh a_mesh(0.5, 2);
   const Cartesian1DMesh c_mesh(1.0 / 3.0, 3);
   const auto a = MakePartition(MakeCellPart(a_mesh));
   const auto b = MakeNonconformingFactor();
   const auto c = MakePartition(MakeCellPart(c_mesh));

   // Construct the outer product directly from a temporary inner product to
   // exercise the value ownership required by lazy nested connectivity.
   const auto left_associated = MakeCartesianProductPartition(
      MakeCartesianProductPartition(a, b),
      c);
   const auto right_associated = MakeCartesianProductPartition(
      a,
      MakeCartesianProductPartition(b, c));

   const auto& left_faces =
      std::get<0>(left_associated.InteriorFaceParts()).face_mesh;
   const auto& right_faces =
      std::get<0>(right_associated.InteriorFaceParts()).face_mesh;
   const auto left_info = left_faces.GetGlobalFaceInfo(4);
   const auto right_info = right_faces.GetGlobalFaceInfo(4);
   using LeftInfo = std::remove_cvref_t<decltype(left_info)>;
   using RightInfo = std::remove_cvref_t<decltype(right_info)>;
   using LeftMinus = typename LeftInfo::minus_side_type;
   using RightMinus = typename RightInfo::minus_side_type;
   static_assert(LeftMinus::local_face_index_type::value == 5);
   static_assert(RightMinus::local_face_index_type::value == 5);
   static_assert(LeftMinus::normal_type::index == 1);
   static_assert(RightMinus::normal_type::index == 1);
   static_assert(LeftMinus::normal_type::sign == 1);
   static_assert(RightMinus::normal_type::sign == 1);
   static_assert(std::same_as<
      typename LeftInfo::plus_side_type::orientation_type,
      TensorProductOrientation<
         TensorProductOrientation<IdentityOrientation<1>, Permutation<2>>,
         IdentityOrientation<1>>>);
   static_assert(std::same_as<
      typename RightInfo::plus_side_type::orientation_type,
      TensorProductOrientation<
         IdentityOrientation<1>,
         TensorProductOrientation<Permutation<2>, IdentityOrientation<1>>>>);

   bool success = Check(
      left_info.MinusSide().GetCellIndex() == 18 &&
      left_info.PlusSide().GetCellIndex() == 20 &&
      right_info.MinusSide().GetCellIndex() == 18 &&
      right_info.PlusSide().GetCellIndex() == 20,
      "nested middle-factor product cell indices agree by association");
   success = CheckPermutation(
      left_info.PlusSide().GetOrientation(),
      std::array<LocalIndex, 4>{1, 2, -3, 4},
      "left-associated middle-factor signed orientation") && success;
   success = CheckPermutation(
      right_info.PlusSide().GetOrientation(),
      std::array<LocalIndex, 4>{1, 2, -3, 4},
      "right-associated middle-factor signed orientation") && success;
   success = Check(
      FlattenOrientation(left_info.PlusSide().GetOrientation()) ==
         FlattenOrientation(right_info.PlusSide().GetOrientation()),
      "middle-factor orientation values agree by association") && success;
   const auto& left_map = left_info.MinusSide().conformity;
   const auto& right_map = right_info.MinusSide().conformity;
   const std::array<Real, 4> expected_origin{0.0, 0.0, 0.2, 0.0};
   const std::array<Real, 4> expected_size{1.0, 1.0, 0.3, 1.0};
   for (Integer d = 0; d < 4; ++d)
   {
      success = Check(
         std::abs(left_map.origin[d] - expected_origin[d]) <
            product_partition_tolerance &&
         std::abs(right_map.origin[d] - expected_origin[d]) <
            product_partition_tolerance &&
         std::abs(left_map.size[d] - expected_size[d]) <
            product_partition_tolerance &&
         std::abs(right_map.size[d] - expected_size[d]) <
            product_partition_tolerance,
         "nested middle-factor conformity-map coordinate block agrees by "
         "association") && success;
   }
   return success;
}

bool TestNestedStaticIdentityByAssociation()
{
   const Cartesian1DMesh a_mesh(0.5, 2, -0.25);
   const Cartesian1DMesh c_mesh(1.0 / 3.0, 3, 0.75);
   const auto a = MakePartition(MakeCellPart(a_mesh));
   const auto b = MakeStaticIdentityMiddleFactor();
   const auto c = MakePartition(MakeCellPart(c_mesh));

   const auto left_associated = MakeCartesianProductPartition(
      MakeCartesianProductPartition(a, b),
      c);
   const auto right_associated = MakeCartesianProductPartition(
      a,
      MakeCartesianProductPartition(b, c));

   const auto left_info =
      std::get<0>(left_associated.InteriorFaceParts()).face_mesh.
         GetGlobalFaceInfo(4);
   const auto right_info =
      std::get<0>(right_associated.InteriorFaceParts()).face_mesh.
         GetGlobalFaceInfo(4);
   using LeftInfo = std::remove_cvref_t<decltype(left_info)>;
   using RightInfo = std::remove_cvref_t<decltype(right_info)>;
   static_assert(std::same_as<
      typename LeftInfo::minus_side_type::orientation_type,
      IdentityOrientation<4>>);
   static_assert(std::same_as<
      typename LeftInfo::plus_side_type::orientation_type,
      IdentityOrientation<4>>);
   static_assert(std::same_as<
      typename RightInfo::minus_side_type::orientation_type,
      IdentityOrientation<4>>);
   static_assert(std::same_as<
      typename RightInfo::plus_side_type::orientation_type,
      IdentityOrientation<4>>);

   return Check(
             left_info.MinusSide().GetCellIndex() == 18 &&
             left_info.PlusSide().GetCellIndex() == 20 &&
             right_info.MinusSide().GetCellIndex() == 18 &&
             right_info.PlusSide().GetCellIndex() == 20,
             "nested static-identity cell numbering agrees by association") &&
          Check(
             static_cast<Permutation<4>>(
                left_info.PlusSide().GetOrientation()) ==
                static_cast<Permutation<4>>(
                   right_info.PlusSide().GetOrientation()),
             "nested static-identity values agree by association");
}

} // namespace

int main()
{
   bool success = true;
   success = TestAsymmetricStructure() && success;
   success = TestNestedMiddleFactorLifting() && success;
   success = TestNestedStaticIdentityByAssociation() && success;
   return success ? 0 : 1;
}

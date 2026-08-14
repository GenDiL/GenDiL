// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <string>

namespace gendil::test
{

inline constexpr Real product_partition_tolerance = 1.0e-11;

inline bool Check(const bool condition, const std::string& message)
{
   if (!condition)
   {
      std::cerr << "FAILED: " << message << '\n';
   }
   return condition;
}

template<Integer Dim, Integer LocalFaceIndex, size_t NumRecords>
struct MaterializedBoundaryConnectivity
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
      CanonicalVector<Dim, axis, -sign>,
      bool>;

   struct Record
   {
      GlobalIndex cell;
      GlobalIndex dummy_plus_cell;
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
         {record.cell, {}, {}, {}, {}, true},
         {record.dummy_plus_cell, {}, {}, {}, {}, true}};
   }
};

template<class Operator>
Vector Apply(const Operator& op, const Vector& input)
{
   Vector output(input.Size());
   output = 0.0;
   op(input, output);
   return output;
}

inline Real DifferenceNorm(const Vector& a, const Vector& b)
{
   const Real* a_data = a.ReadHostData();
   const Real* b_data = b.ReadHostData();
   Real norm2 = 0.0;
   for (Integer i = 0; i < a.Size(); ++i)
   {
      const Real difference = a_data[i] - b_data[i];
      norm2 += difference * difference;
   }
   return std::sqrt(norm2);
}

inline Real VectorNorm(const Vector& vector)
{
   const Real* data = vector.ReadHostData();
   Real norm2 = 0.0;
   for (Integer i = 0; i < vector.Size(); ++i)
   {
      norm2 += data[i] * data[i];
   }
   return std::sqrt(norm2);
}

inline bool CheckVectorClose(
   const std::string& label,
   const Vector& got,
   const Vector& expected)
{
   const Real error = DifferenceNorm(got, expected);
   const Real scale = std::max(Real{1}, VectorNorm(expected));
   return Check(
      error <= product_partition_tolerance * scale,
      label);
}

struct NonconformingFactorConnectivity
{
   using geometry = HyperCube<2>;
   using minus_side_type = FaceView<
      std::integral_constant<Integer, 2>,
      geometry,
      Permutation<2>,
      CanonicalVector<2, 0, 1>,
      NonconformingHyperCubeFaceMap<2>>;
   using plus_side_type = FaceView<
      std::integral_constant<Integer, 0>,
      geometry,
      Permutation<2>,
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
         minus_side_type{
            1,
            {},
            MakeReferencePermutation<2>(),
            {},
            {Point<2>{0.0, 0.2}, std::array<Real, 2>{1.0, 0.3}},
            {}},
         plus_side_type{
            2,
            {},
            Permutation<2>{{1, -2}},
            {},
            {},
            {}}};
   }
};

inline auto MakeNonconformingFactor()
{
   Cartesian2DMesh mesh(0.5, 0.5, 2, 2, Point<2>{0.0, 0.0});
   return MakePartition(
      MakeCellPart(mesh),
      MakeInteriorFacePart<0, 0>(NonconformingFactorConnectivity{}));
}

} // namespace gendil::test

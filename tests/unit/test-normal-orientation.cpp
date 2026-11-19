// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/globalfacesfromlocalconnectivity.hpp>
#ifdef GENDIL_USE_MFEM
#include <gendil/Interfaces/MFEM/GlobalFaceConnectivity/conformingglobalfacebuilder.hpp>
#endif

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace gendil;

namespace normal_orientation_test
{
using Side = FaceView<Integer, HyperCube<2>, Permutation<2>,
   std::array<Real, 2>, NonconformingHyperCubeFaceMap<2>, bool, int>;
using RuntimeFace = LocalFaceInfo<Side, Side, int>;

static_assert(!face_connectivity_detail::HasOrientationValue<int>);
static_assert(!face_connectivity_detail::StaticOrientation<int>);
static_assert(face_connectivity_detail::HasOrientationValue<std::integral_constant<int, -1>>);
static_assert(face_connectivity_detail::StaticOrientation<std::integral_constant<int, -1>>);

template<class Face>
concept HasCanonicalSides = requires(const Face& face)
{
   face.MinusSide();
   face.PlusSide();
};

template<class Face>
concept HasCanonicalSideTypes = requires
{
   typename Face::minus_side_type;
   typename Face::plus_side_type;
};

template<class Face>
concept HasLocalSides = requires(const Face& face)
{
   face.CurrentSide();
   face.NeighborSide();
};

using DistinctRuntimeFace = LocalFaceInfo<Side, Empty, int>;
using DistinctPositiveFace = LocalFaceInfo<Side, Empty, std::integral_constant<int, 1>>;
using DistinctNegativeFace = LocalFaceInfo<Side, Empty, std::integral_constant<int, -1>>;
using DistinctGlobalFace = GlobalFaceInfo<Side, Empty>;
using DistinctGlobalRuntimeStorage = GlobalFaceInfo<Side, Empty, int>;
static_assert(HasLocalSides<RuntimeFace> && !HasCanonicalSides<RuntimeFace>);
static_assert(!HasCanonicalSideTypes<RuntimeFace>);
static_assert(HasCanonicalSides<DistinctPositiveFace> && HasCanonicalSides<DistinctNegativeFace>);
static_assert(std::same_as<DistinctPositiveFace::minus_side_type, Side>);
static_assert(std::same_as<DistinctPositiveFace::plus_side_type, Empty>);
static_assert(std::same_as<DistinctNegativeFace::minus_side_type, Empty>);
static_assert(std::same_as<DistinctNegativeFace::plus_side_type, Side>);
static_assert(HasCanonicalSides<DistinctGlobalFace> && !HasLocalSides<DistinctGlobalFace>);
static_assert(std::same_as<DistinctGlobalFace::normal_orientation_type, std::integral_constant<int, 1>>);
static_assert(sizeof(DistinctPositiveFace) == sizeof(DistinctGlobalFace));
static_assert(sizeof(DistinctNegativeFace) == sizeof(DistinctGlobalFace));
static_assert(alignof(DistinctNegativeFace) == alignof(DistinctGlobalFace));
static_assert(sizeof(DistinctRuntimeFace) == sizeof(GlobalFaceInfo<Side, Empty, int>));
static_assert(alignof(DistinctRuntimeFace) == alignof(GlobalFaceInfo<Side, Empty, int>));
static_assert(std::is_standard_layout_v<DistinctRuntimeFace>);
static_assert(std::is_standard_layout_v<DistinctNegativeFace>);
static_assert(std::is_aggregate_v<DistinctRuntimeFace>);
static_assert(std::is_aggregate_v<DistinctNegativeFace>);
static_assert(std::is_trivially_copyable_v<DistinctRuntimeFace>);
static_assert(std::is_trivially_copyable_v<DistinctNegativeFace>);
static_assert(offsetof(DistinctNegativeFace, current) == offsetof(DistinctGlobalFace, minus));
static_assert(offsetof(DistinctNegativeFace, neighbor) == offsetof(DistinctGlobalFace, plus));
static_assert(offsetof(DistinctRuntimeFace, normal_orientation) ==
   offsetof(DistinctGlobalRuntimeStorage, normal_orientation));
static_assert([]
{
   const DistinctNegativeFace face{};
   return &face.MinusSide() == &face.NeighborSide() &&
      &face.PlusSide() == &face.CurrentSide();
}());

// Derived/custom records must explicitly preserve their additional metadata.
struct UnadaptedFace : RuntimeFace
{
   int attribute;
};

template<class Orientation>
struct CustomFace : LocalFaceInfo<Side, Side, Orientation>
{
   int attribute;
};

template<int Sign, class Orientation>
   requires std::same_as<Orientation, int>
GENDIL_HOST_DEVICE
constexpr auto WithStaticNormalOrientation(const CustomFace<Orientation>& face)
{
   return CustomFace<std::integral_constant<int, Sign>>{
      {face.CurrentSide(), face.NeighborSide(), {}}, face.attribute};
}

template<class Face>
concept HasBuiltinRebind = requires(const Face& face)
{
   gendil::WithStaticNormalOrientation<1>(face);
};

static_assert(HasBuiltinRebind<RuntimeFace>);
static_assert(!HasBuiltinRebind<UnadaptedFace>);
static_assert(!HasBuiltinRebind<CustomFace<int>>);
} // namespace normal_orientation_test

namespace
{
template<class Side>
struct OriginalSideStorage
{
   GlobalIndex cell_index;
   typename Side::local_face_index_type local_face_index;
   typename Side::orientation_type orientation;
   typename Side::normal_type normal;
   typename Side::conformity_type conformity;
   typename Side::boundary_type boundary;
};

template<class Face>
struct OriginalFaceStorage
{
   typename Face::minus_side_type minus;
   typename Face::plus_side_type plus;
};

template<class Face>
struct OriginalLocalFaceStorage
{
   typename Face::current_side_type current;
   typename Face::neighbor_side_type neighbor;
};

GENDIL_HOST_DEVICE
normal_orientation_test::RuntimeFace MakeDispatchFace(int direction)
{
   return {
      {17, 2, {{-2, 1}}, {.6, .8}, {{.125, .25}, {.5, .75}}, false, 73},
      {29, 1, {{2, -1}}, {-.6, -.8}, {{.375, .5}, {.25, .125}}, true, 91},
      direction};
}

GENDIL_HOST_DEVICE
bool SameSide(const normal_orientation_test::Side& actual,
              const normal_orientation_test::Side& expected)
{
   bool ok = actual.GetCellIndex() == expected.GetCellIndex() &&
      actual.local_face_index == expected.local_face_index &&
      actual.IsBoundary() == expected.IsBoundary() &&
      actual.IsConforming() == expected.IsConforming() &&
      actual.normal_orientation == expected.normal_orientation;
   for (int axis = 0; axis < 2; ++axis)
      ok = ok && actual.GetOrientation()(axis) == expected.GetOrientation()(axis) &&
         actual.GetReferenceNormal()[axis] == expected.GetReferenceNormal()[axis] &&
         actual.conformity.origin[axis] == expected.conformity.origin[axis] &&
         actual.conformity.size[axis] == expected.conformity.size[axis];
   return ok;
}

template<int Sign>
GENDIL_HOST_DEVICE
bool CheckStaticRebind(const normal_orientation_test::RuntimeFace& source)
{
   const auto fixed = gendil::WithStaticNormalOrientation<Sign>(source);
   using Face = std::remove_cvref_t<decltype(fixed)>;
   static_assert(std::is_same_v<Face, LocalFaceInfo<normal_orientation_test::Side,
      normal_orientation_test::Side, std::integral_constant<int, Sign>>>);
   static_assert(static_face_normal_orientation_v<const Face&> == Sign);
   static_assert(sizeof(Face) == sizeof(OriginalLocalFaceStorage<Face>));
   static_assert(alignof(Face) == alignof(OriginalLocalFaceStorage<Face>));
   static_assert(std::is_trivially_copyable_v<Face>);
   bool ok = SameSide(fixed.CurrentSide(), source.CurrentSide()) &&
      SameSide(fixed.NeighborSide(), source.NeighborSide());
   if constexpr (Sign == 1)
      ok = ok && &fixed.MinusSide() == &fixed.CurrentSide() &&
         &fixed.PlusSide() == &fixed.NeighborSide();
   else
      ok = ok && &fixed.MinusSide() == &fixed.NeighborSide() &&
         &fixed.PlusSide() == &fixed.CurrentSide();
   // Local geometry always uses the current outward reference normal.
   for (int axis = 0; axis < 2; ++axis)
      ok = ok && fixed.GetReferenceNormal()[axis] == fixed.CurrentSide().GetReferenceNormal()[axis];
   int calls = 0;
   WithLocalFacetNormalOrientation(fixed, [&](const auto& face)
   {
      static_assert(static_face_normal_orientation_v<decltype(face)> == Sign);
      ok = ok && &face == &fixed;
      ++calls;
   });
   return ok && calls == 1;
}

GENDIL_HOST_DEVICE
bool CheckDistinctSideDispatch(int direction)
{
   const auto source = MakeDispatchFace(direction);
   const normal_orientation_test::DistinctRuntimeFace local{source.CurrentSide(), {}, direction};
   bool ok = true;
   WithLocalFacetNormalOrientation(local, [&](const auto& face)
   {
      constexpr int sign = static_face_normal_orientation_v<decltype(face)>;
      using Face = std::remove_cvref_t<decltype(face)>;
      using Current = typename Face::current_side_type;
      using Neighbor = typename Face::neighbor_side_type;
      if constexpr (sign == 1)
      {
         static_assert(std::same_as<decltype(face.MinusSide()), const Current&>);
         static_assert(std::same_as<decltype(face.PlusSide()), const Neighbor&>);
         ok = ok && &face.MinusSide() == &face.CurrentSide() &&
            &face.PlusSide() == &face.NeighborSide();
      }
      else
      {
         static_assert(std::same_as<decltype(face.MinusSide()), const Neighbor&>);
         static_assert(std::same_as<decltype(face.PlusSide()), const Current&>);
         ok = ok && &face.MinusSide() == &face.NeighborSide() &&
            &face.PlusSide() == &face.CurrentSide();
      }
      ok = ok && face.CurrentSide().GetCellIndex() == 17 && sign == direction;
   });
   return ok && local.CurrentSide().GetCellIndex() == 17;
}

GENDIL_HOST_DEVICE
bool CheckReboundDispatch(int direction)
{
   const auto source = MakeDispatchFace(direction);
   bool ok = CheckStaticRebind<1>(source) && CheckStaticRebind<-1>(source);
   int calls = 0;
   WithLocalFacetNormalOrientation(source, [&](const auto& face)
   {
      constexpr int sign = static_face_normal_orientation_v<decltype(face)>;
      ok = ok && sign == direction &&
         SameSide(face.CurrentSide(), source.CurrentSide()) &&
         SameSide(face.NeighborSide(), source.NeighborSide()) &&
         face.MinusSide().GetCellIndex() == (sign == 1 ? 17 : 29) &&
         face.PlusSide().GetCellIndex() == (sign == 1 ? 29 : 17);
      ++calls;
   });
   const auto expected = MakeDispatchFace(direction);
   return ok && calls == 1 && source.GetNormalOrientation() == direction &&
      SameSide(source.CurrentSide(), expected.CurrentSide()) &&
      SameSide(source.NeighborSide(), expected.NeighborSide());
}

GENDIL_HOST_DEVICE
bool CheckCustomDispatch(int direction)
{
   const auto source = MakeDispatchFace(direction);
   const normal_orientation_test::CustomFace<int> custom{source, 137};
   bool ok = true;
   int calls = 0;
   WithLocalFacetNormalOrientation(custom, [&](const auto& face)
   {
      constexpr int sign = static_face_normal_orientation_v<decltype(face)>;
      using Face = std::remove_cvref_t<decltype(face)>;
      static_assert(std::is_same_v<Face,
         normal_orientation_test::CustomFace<std::integral_constant<int, sign>>>);
      ok = ok && sign == direction && face.attribute == 137 &&
         SameSide(face.CurrentSide(), source.CurrentSide()) &&
         SameSide(face.NeighborSide(), source.NeighborSide()) &&
         face.MinusSide().GetCellIndex() == (sign == 1 ? 17 : 29) &&
         face.PlusSide().GetCellIndex() == (sign == 1 ? 29 : 17);
      ++calls;
      WithLocalFacetNormalOrientation(face, [&](const auto& unchanged)
      {
         ok = ok && &unchanged == &face && unchanged.attribute == 137;
      });
   });
   return ok && calls == 1 && custom.attribute == 137 &&
      custom.GetNormalOrientation() == direction &&
      SameSide(custom.CurrentSide(), source.CurrentSide()) &&
      SameSide(custom.NeighborSide(), source.NeighborSide());
}

template<Integer Dim>
bool CheckCartesian()
{
   std::array<GlobalIndex, Dim> sizes{};
   std::array<Real, Dim> spacing{};
   sizes.fill(2);
   spacing.fill(1.0);
   const CartesianMesh<Dim> mesh(sizes, spacing, Point<Dim>{}, false);
   bool ok = true;
   ConstexprLoop<2*Dim>([&](auto face_index)
   {
      constexpr int expected = -HyperCube<Dim>::GetNormalSign(face_index);
      const auto face = mesh.GetLocalFaceInfo(0, face_index);
      using Face = decltype(face);
      using Side = typename Face::current_side_type;
      static_assert(decltype(face.GetNormalOrientation())::value == expected);
      static_assert(sizeof(Face) == sizeof(OriginalLocalFaceStorage<Face>));
      static_assert(alignof(Face) == alignof(OriginalLocalFaceStorage<Face>));
      static_assert(sizeof(Side) == sizeof(OriginalSideStorage<Side>));
      using StoredSide = typename ConformingCellConnectivity<HyperCube<Dim>>::FaceInfo;
      static_assert(sizeof(StoredSide) == sizeof(OriginalSideStorage<StoredSide>));
      ok = ok && face.GetNormalOrientation() == expected &&
         face.CurrentSide().GetCellIndex() == 0;
   });

   const auto global = MakeCartesianInteriorFaceConnectivity<Dim>(sizes);
   std::apply([&](const auto&... family)
   {
      const auto check = [&](const auto& faces)
      {
         for (GlobalIndex i = 0; i < faces.GetNumberOfFaces(); ++i)
         {
            const auto face = faces.GetGlobalFaceInfo(i);
            static_assert(decltype(face.GetNormalOrientation())::value == 1);
            static_assert(sizeof(face) == sizeof(OriginalFaceStorage<decltype(face)>));
            using Index = typename decltype(face)::minus_side_type::local_face_index_type;
            const auto local = mesh.GetLocalFaceInfo(face.MinusSide().GetCellIndex(), Index{});
            ok = ok && local.GetNormalOrientation() == 1 &&
               local.NeighborSide().GetCellIndex() == face.PlusSide().GetCellIndex();
         }
      };
      (check(family), ...);
   }, global);
   return ok;
}

GENDIL_HOST_DEVICE
bool CheckLegacyAndPeriodic(const Cartesian1DMesh& line,
                            const PeriodicCartesian1DMesh& periodic)
{
   const auto left = line.GetLocalFaceInfo(1, std::integral_constant<Integer, 0>{});
   const auto right = line.GetLocalFaceInfo(0, std::integral_constant<Integer, 1>{});
   const auto wrap_left = periodic.GetLocalFaceInfo(0, std::integral_constant<Integer, 0>{});
   const auto wrap_right = periodic.GetLocalFaceInfo(0, std::integral_constant<Integer, 1>{});
   static_assert(decltype(left.GetNormalOrientation())::value == 1);
   static_assert(decltype(right.GetNormalOrientation())::value == -1);
   return left.NeighborSide().GetCellIndex() == 0 && right.NeighborSide().GetCellIndex() == 1 &&
      wrap_left.NeighborSide().GetCellIndex() == 0 && wrap_right.NeighborSide().GetCellIndex() == 0 &&
      wrap_left.GetNormalOrientation() == 1 && wrap_right.GetNormalOrientation() == -1;
}

auto MakeRuntimeConnectivity(bool explicit_orientation)
{
   UnstructuredConformingConnectivity<HyperCube<1>> connectivity(2);
   for (GlobalIndex cell = 0; cell < 2; ++cell)
      for (Integer face = 0; face < 2; ++face)
         connectivity[cell].faces[face] =
            {0, {}, MakeReferencePermutation<1>(), {}, {}, true};
   connectivity[0].faces[1] = {1, {}, MakeReferencePermutation<1>(), {}, {}, false};
   connectivity[1].faces[0] = {0, {}, MakeReferencePermutation<1>(), {}, {}, false};
   if (explicit_orientation)
   {
      connectivity[0].faces[1].normal_orientation = -1;
      connectivity[1].faces[0].normal_orientation = 1;
   }
   ToDevice(2, connectivity.element_connectivities);
   return connectivity;
}

struct RuntimeLine
{
   static constexpr Integer Dim = 1;
   using cell_type = SegmentCell;
   UnstructuredConformingConnectivity<HyperCube<1>> connectivity;

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfCells() const { return 2; }

   template<Integer FaceIndex>
   GENDIL_HOST_DEVICE
   auto GetLocalFaceInfo(GlobalIndex cell, std::integral_constant<Integer, FaceIndex> face) const
   {
      return connectivity.GetLocalFaceInfo(cell, face);
   }
};

template<class Connectivity>
GENDIL_HOST_DEVICE
bool CheckRuntime(const Connectivity& connectivity, int expected)
{
   const auto right = connectivity.GetLocalFaceInfo(0, std::integral_constant<Integer, 1>{});
   const auto left = connectivity.GetLocalFaceInfo(1, std::integral_constant<Integer, 0>{});
   static_assert(std::is_same_v<decltype(right.GetNormalOrientation()), int>);
   return right.GetNormalOrientation() == expected &&
      left.GetNormalOrientation() == -expected &&
      right.CurrentSide().GetCellIndex() == 0 && right.NeighborSide().GetCellIndex() == 1;
}

template<class Product>
GENDIL_HOST_DEVICE
bool CheckProduct(const Product& product)
{
   // The runtime factor is in the middle, so these exercise both recursion paths.
   const auto x = product.GetLocalFaceInfo(0, std::integral_constant<Integer, 3>{});
   const auto y = product.GetLocalFaceInfo(0, std::integral_constant<Integer, 4>{});
   const auto z = product.GetLocalFaceInfo(0, std::integral_constant<Integer, 5>{});
   static_assert(decltype(x.GetNormalOrientation())::value == -1);
   static_assert(std::is_same_v<decltype(y.GetNormalOrientation()), int>);
   static_assert(decltype(z.GetNormalOrientation())::value == -1);
   return x.NeighborSide().GetCellIndex() == 1 && y.NeighborSide().GetCellIndex() == 2 &&
      z.NeighborSide().GetCellIndex() == 4 && y.GetNormalOrientation() == 1;
}

bool CheckFallback(bool explicit_orientation)
{
   const auto connectivity = MakeRuntimeConnectivity(explicit_orientation);
   const auto global = MakeUnstructuredInteriorFaceConnectivity(connectivity, 2);
   bool ok = true;
   GlobalIndex count = 0;
   std::apply([&](const auto&... family)
   {
      const auto check = [&](const auto& faces)
      {
         count += faces.GetNumberOfFaces();
         for (GlobalIndex i = 0; i < faces.GetNumberOfFaces(); ++i)
         {
            const auto face = faces.GetGlobalFaceInfo(i);
            const GlobalIndex expected = explicit_orientation ? 1 : 0;
            ok = ok && face.MinusSide().GetCellIndex() == expected &&
               face.PlusSide().GetCellIndex() == 1-expected;
         }
      };
      (check(family), ...);
   }, global);
   return ok && count == 1;
}

#ifdef GENDIL_USE_MFEM
bool CheckMFEMImport()
{
   auto mesh = mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::QUADRILATERAL, true);
   const auto local = MakeMeshLocalConnectivity<2>(mesh);
   const auto direct = MakeMFEMConformingGlobalInteriorFaceConnectivity<HyperCube<2>>(mesh);
   const auto fallback = MakeUnstructuredInteriorFaceConnectivity(local, mesh.GetNE());
   bool ok = true;
   using Pair = std::array<GlobalIndex, 2>;
   const auto collect = [&](const auto& families)
   {
      std::vector<Pair> result;
      std::apply([&](const auto&... family)
      {
         const auto read = [&](const auto& faces)
         {
            for (GlobalIndex i = 0; i < faces.GetNumberOfFaces(); ++i)
            {
               const auto face = faces.GetGlobalFaceInfo(i);
               using Index = typename decltype(face)::minus_side_type::local_face_index_type;
               const auto current = local.GetLocalFaceInfo(face.MinusSide().GetCellIndex(), Index{});
               ok = ok && current.GetNormalOrientation() == 1;
               result.push_back({face.MinusSide().GetCellIndex(), face.PlusSide().GetCellIndex()});
            }
         };
         (read(family), ...);
      }, families);
      std::sort(result.begin(), result.end());
      return result;
   };
   const auto expected = collect(direct.connectivity);
   const auto actual = collect(fallback);
   return ok && expected == actual && expected.size() == 4;
}
#endif
} // namespace

int main()
{
   const Cartesian1DMesh line(1., 2);
   const PeriodicCartesian1DMesh periodic(1., 1);
   const auto native = MakeRuntimeConnectivity(false);
   const auto imported = MakeRuntimeConnectivity(true);
   const auto product = MakeCartesianProductMesh(line, RuntimeLine{native}, line);
   bool ok = CheckCartesian<1>() && CheckCartesian<2>() && CheckCartesian<3>() &&
      CheckLegacyAndPeriodic(line, periodic) && CheckRuntime(native, 1) &&
      CheckRuntime(imported, -1) && CheckProduct(product) &&
      CheckFallback(false) && CheckFallback(true) &&
      CheckReboundDispatch(1) && CheckReboundDispatch(-1) &&
      CheckCustomDispatch(1) && CheckCustomDispatch(-1) &&
      CheckDistinctSideDispatch(1) && CheckDistinctSideDispatch(-1);
#ifdef GENDIL_USE_MFEM
   ok = CheckMFEMImport() && ok;
#endif
#ifdef GENDIL_USE_DEVICE
   using Policy = DeviceKernelConfiguration<ThreadBlockLayout<1>, 1, 4>;
   // Six items with a batch size of four also exercise an incomplete final batch.
   Vector input(2), results(6);
   auto* input_host = input.WriteHostData();
   input_host[0] = 1.;
   input_host[1] = -1.;
   const auto* directions = input.ReadDeviceData();
   auto* values = results.WriteDeviceData();
   Policy::CandidateBlockLoop(6, [=] GENDIL_DEVICE ()
   {
      const auto i = Policy::WorkItemIndex();
      if (Policy::IsActive(6))
      {
         if (i == 0) values[i] = CheckLegacyAndPeriodic(line, periodic);
         if (i == 1) values[i] = CheckRuntime(native, 1);
         if (i == 2) values[i] = CheckRuntime(imported, -1);
         if (i == 3) values[i] = CheckProduct(product);
         if (i >= 4)
         {
            const int direction = static_cast<int>(directions[i - 4]);
            values[i] = CheckReboundDispatch(direction) && CheckCustomDispatch(direction) &&
               CheckDistinctSideDispatch(direction);
         }
      }
   });
   GENDIL_DEVICE_SYNC;
   const auto* values_host = results.ReadHostData();
   for (Integer i = 0; i < 6; ++i) ok = ok && values_host[i] == 1.;
#endif
   std::cout << (ok ? "PASS" : "FAIL") << ": facet normal orientation\n";
   return ok ? 0 : 1;
}

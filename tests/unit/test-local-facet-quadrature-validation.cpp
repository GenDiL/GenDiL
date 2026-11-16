// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/FiniteElementMethod/OperatorValidation/operatorvalidation.hpp>
#include <gendil/gendil.hpp>
#include <gendil/Meshes/MeshDataStructures/UnstructuredMesh/quadmesh.hpp>
#include <gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/globalfacesfromlocalconnectivity.hpp>

#include <array>
#include <atomic>
#include <iostream>
#include <string_view>

using namespace gendil;
namespace validation = gendil::local_facet_quadrature_detail;

// Keep fixture-dependent template instantiations externally linked:
// AppleClang 21's OpenMP liveness analysis scales poorly for internal definitions.
namespace local_facet_quadrature_validation_test
{

struct QuadratureChecks : OperatorChecks
{
   static constexpr bool local_facet_quadrature = true;
};

// A disabled policy must not instantiate any form, context, or rule machinery.
struct NoValidationMembers {};

struct ReorderedGL2
{
   static constexpr Integer GetNumPoints() { return 2; }
   static constexpr Real GetCoord(Integer q) { return GaussLegendrePoints<2>::GetCoord(1 - q); }
   static constexpr Real GetWeight(Integer q) { return GaussLegendrePoints<2>::GetWeight(1 - q); }
};

struct AsymmetricPoints
{
   static constexpr Integer GetNumPoints() { return 2; }
   static constexpr Real GetCoord(Integer q) { return q == 0 ? Real{0.2} : Real{0.7}; }
   static constexpr Real GetWeight(Integer q) { return q == 0 ? Real{0.4} : Real{0.6}; }
};

struct RuntimeGL2
{
   static inline std::array<Real, 2> coords{
      GaussLegendrePoints<2>::GetCoord(0), GaussLegendrePoints<2>::GetCoord(1)};
   static inline Integer coordinate_reads = 0;
   static constexpr Integer GetNumPoints() { return 2; }
   static Real GetCoord(Integer q) { ++coordinate_reads; return coords[q]; }
   static Real GetWeight(Integer q) { return GaussLegendrePoints<2>::GetWeight(q); }
};

using CustomSymmetricRule = IntegrationRule<HyperCube<2>,
   TensorPoints<GaussLegendrePoints<2>, ReorderedGL2>>;
using RuntimeRule = IntegrationRule<HyperCube<2>,
   TensorPoints<GaussLegendrePoints<3>, RuntimeGL2>>;
using AnisotropicRule = decltype(MakeIntegrationRule(IntegrationRuleNumPoints<3, 4>{}));
using UniformRule = decltype(MakeIntegrationRule(IntegrationRuleNumPoints<3, 3>{}));
using MixedRule = IntegrationRule<HyperCube<2>,
   TensorPoints<GaussLegendrePoints<2>, GaussLobattoLegendrePoints<2>>>;
using AsymmetricRule = IntegrationRule<HyperCube<2>,
   TensorPoints<GaussLegendrePoints<3>, AsymmetricPoints>>;

static_assert(validation::PointSetsMatch<GaussLegendrePoints<2>, ReorderedGL2>(false));
static_assert(validation::PointSetsMatch<GaussLegendrePoints<2>, ReorderedGL2>(true));
static_assert(validation::PointSetsMatch<GaussLegendrePoints<1>, GaussLobattoLegendrePoints<1>>(true));
static_assert(!validation::PointSetsMatch<GaussLegendrePoints<2>, GaussLobattoLegendrePoints<2>>(false));
static_assert(!validation::PointSetsMatch<GaussLegendrePoints<3>, GaussLegendrePoints<4>>(false));
static_assert(!validation::PointSetsMatch<AsymmetricPoints, AsymmetricPoints>(true));
static_assert(validation::PointSetsMatch<AsymmetricPoints, AsymmetricPoints>(false));

template<Integer Dim, Integer FaceIndex = 0, class Orientation>
auto MakeFace(const Orientation& orientation)
{
   using Geometry = HyperCube<Dim>;
   using Face = LocalConformingCellFaceView<Geometry,
      std::integral_constant<Integer, FaceIndex>,
      std::integral_constant<Integer, Geometry::GetOppositeFaceIndex(FaceIndex)>,
      Orientation,
      CanonicalVector<Dim, Geometry::GetNormalDimensionIndex(FaceIndex), -1>,
      CanonicalVector<Dim, Geometry::GetNormalDimensionIndex(FaceIndex), 1>>;
   return Face{{0}, {1, {}, orientation}};
}

bool Check(bool condition, const char* label)
{
   if (!condition) std::cerr << "FAILED: " << label << '\n';
   return condition;
}

struct CountedQuadMesh : QuadMesh<1>
{
   std::atomic<Integer>* face_reads;

   template<Integer FaceIndex>
   GENDIL_HOST_DEVICE
   auto GetLocalFaceInfo(GlobalIndex cell, std::integral_constant<Integer, FaceIndex> face) const
   {
#ifndef GENDIL_DEVICE_CODE
      // Host cell traversal can run concurrently when OpenMP is enabled.
      face_reads->fetch_add(1, std::memory_order_relaxed);
#endif
      return QuadMesh<1>::GetLocalFaceInfo(cell, face);
   }
};

struct Fixture
{
   // Geometry is needed only by the accepted operator-invocation check.
   const std::array<Real, 16> nodes{0, 0, 1, 0, 0, 1, 1, 1,
                              1, 0, 2, 0, 1, 1, 2, 1};
   const std::array<int, 8> indices{0, 1, 2, 3, 4, 5, 6, 7};
   UnstructuredConformingConnectivity<HyperCube<2>> connectivity{2};
   std::atomic<Integer> face_reads{0};

   explicit Fixture(bool rotated)
   {
      for (Integer cell = 0; cell < 2; ++cell)
         for (Integer face = 0; face < 4; ++face)
            connectivity[cell].faces[face] =
               {0, {}, MakeReferencePermutation<2>(), {}, {}, true};
      if (rotated)
      {
         connectivity[0].faces[2] = {1, {}, Permutation<2>{{-2, 1}}, {}, {}, false};
         connectivity[1].faces[1] = {0, {}, Permutation<2>{{2, -1}}, {}, {}, false};
      }
      else
      {
         connectivity[0].faces[2] = {1, {}, MakeReferencePermutation<2>(), {}, {}, false};
         connectivity[1].faces[0] = {0, {}, MakeReferencePermutation<2>(), {}, {}, false};
      }
   }

   auto Mesh()
   {
      HostDevicePointer<const int> pointer;
      pointer.host_pointer = indices.data();
      return CountedQuadMesh{QuadMesh<1>{
         MakeFIFOView(nodes.data(), Integer(2), Integer(8)),
         MakeFIFOView(pointer, Integer(2), Integer(2), Integer(2)), connectivity, 2}, &face_reads};
   }
};

bool CheckCompatibility()
{
   bool ok = true;
   ok = Check(validation::FaceQuadratureIsCompatible<AnisotropicRule>(
      MakeFace<2>(IdentityOrientation<2>{})), "anisotropic static identity") && ok;
   ok = Check(validation::FaceQuadratureIsCompatible<AnisotropicRule>(
      MakeFace<2>(Permutation<2>{{1, -2}})), "anisotropic tangential reflection") && ok;
   ok = Check(!validation::FaceQuadratureIsCompatible<AnisotropicRule>(
      MakeFace<2>(Permutation<2>{{-2, 1}})), "unequal counts under axis swap") && ok;
   ok = Check(!validation::FaceQuadratureIsCompatible<MixedRule>(
      MakeFace<2>(Permutation<2>{{-2, 1}})), "equal counts with different weighted points") && ok;
   ok = Check(!validation::FaceQuadratureIsCompatible<AsymmetricRule>(
      MakeFace<2>(Permutation<2>{{1, -2}})), "asymmetric reflected rule") && ok;
   ok = Check(validation::FaceQuadratureIsCompatible<AsymmetricRule>(
      MakeFace<2>(Permutation<2>{{-1, 2}})), "normal reflection does not constrain tangential points") && ok;
   using ProductRule = decltype(MakeIntegrationRule(IntegrationRuleNumPoints<3, 3, 5>{}));
   using ProductOrientation = TensorProductOrientation<Permutation<2>, IdentityOrientation<1>>;
   ConstexprLoop<6>([](auto face_index)
   {
      using ProductFace = decltype(MakeFace<3, decltype(face_index)::value>(ProductOrientation{}));
      using UnrestrictedFace = decltype(MakeFace<3, decltype(face_index)::value>(Permutation<3>{}));
      static_assert(validation::FaceTypePreservesQuadrature<ProductRule, ProductFace>(
         validation::MakeQuadratureCompatibility<ProductRule>()));
      static_assert(!validation::FaceTypePreservesQuadrature<ProductRule, UnrestrictedFace>(
         validation::MakeQuadratureCompatibility<ProductRule>()));
   });
   ok = Check(validation::FaceQuadratureIsCompatible<ProductRule>(MakeFace<3>(
      MakeTensorProductOrientation(Permutation<2>{{-2, 1}}, IdentityOrientation<1>{}))),
      "product orientation preserves the independent anisotropic axis") && ok;

   Fixture rotated(true);
   validation::ValidateMesh<UniformRule>(rotated.Mesh());
   ok = Check(rotated.face_reads.load(std::memory_order_relaxed) == 0,
      "uniform built-in quadrature skips connectivity scan") && ok;
   validation::ValidateMesh<CustomSymmetricRule>(rotated.Mesh());
   ok = Check(rotated.face_reads.load(std::memory_order_relaxed) == 0,
      "equivalent custom symmetric quadrature skips connectivity scan") && ok;
   const auto reflected = MakeFace<2>(Permutation<2>{{1, -2}});
   ok = Check(validation::FaceQuadratureIsCompatible<RuntimeRule>(reflected),
      "non-constexpr custom quadrature supports reflection") && ok;
   const Real saved = RuntimeGL2::coords[0];
   RuntimeGL2::coords[0] = Real{0.1};
   ok = Check(!validation::FaceQuadratureIsCompatible<RuntimeRule>(reflected),
      "custom runtime point values are rechecked after mutation") && ok;
   RuntimeGL2::coords[0] = saved;
   Fixture aligned(false);
   auto empty_mesh = aligned.Mesh();
   empty_mesh.num_elems = 0;
   RuntimeGL2::coordinate_reads = 0;
   validation::ValidateMesh<RuntimeRule>(empty_mesh);
   const auto table_reads = RuntimeGL2::coordinate_reads;
   RuntimeGL2::coordinate_reads = 0;
   validation::ValidateMesh<RuntimeRule>(aligned.Mesh());
   ok = Check(table_reads > 0 && RuntimeGL2::coordinate_reads == table_reads,
      "runtime quadrature comparisons are computed once before face traversal") && ok;
   aligned.face_reads.store(0, std::memory_order_relaxed);
   validation::ValidateMesh<AnisotropicRule>(aligned.Mesh());
   ok = Check(aligned.face_reads.load(std::memory_order_relaxed) == 8,
      "anisotropic unstructured rules inspect actual faces") && ok;
   return ok;
}

bool CheckConstructionAndInvocation()
{
   Fixture fixture(false);
   const auto mesh = fixture.Mesh();
   const auto space = MakeFiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{}));
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space), MakeIntegrationDomain<"m">(mesh));
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   const auto form = integrate(InteriorFacets<"m">{}, jump(u) * jump(v));
   using KP = SerialKernelConfiguration;
   const auto op = MakeGenericOperator<KP>(form, context, AnisotropicRule{});
   const auto pullback_op = MakePullbackGenericOperator<KP>(form, context, AnisotropicRule{});
   const auto residual_op = MakeWeakFormResidualOperator<KP>(form, context, AnisotropicRule{});
   bool ok = Check(fixture.face_reads.load(std::memory_order_relaxed) == 0,
      "operator factories do not scan quadrature compatibility");
   OperatorChecks disabled;
   QuadratureChecks enabled; // The object itself does not have to be constexpr.
   ValidateOperator(NoValidationMembers{}, NoValidationMembers{}, NoValidationMembers{}, disabled);
   ValidateOperator(op, disabled);
   ValidateOperator(form, context, AnisotropicRule{}, disabled);
   ok = Check(fixture.face_reads.load(std::memory_order_relaxed) == 0,
      "default policy does not scan or instantiate check machinery") && ok;
   ValidateOperator(op, enabled);
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == 8,
      "derived policy enables explicit operator validation") && ok;
   ValidateOperator(pullback_op, enabled);
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == 8,
      "pullback factory operator supports validation") && ok;
   ValidateOperator(residual_op, enabled);
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == 8,
      "residual factory operator supports validation") && ok;
   ValidateOperator(form, context, AnisotropicRule{}, enabled);
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == 8,
      "form overload explicitly validates current connectivity") && ok;
   Vector x(space.GetNumberOfFiniteElementDofs()), y(space.GetNumberOfFiniteElementDofs());
   x = 1.0;
   op(x, y);
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == 8,
      "operator invocation visits numerical faces only") && ok;
   op.ApplyAdd(x, y);
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == 8,
      "additive operator invocation visits numerical faces only") && ok;
   GenericExplicitOperator<KP>(form, context, AnisotropicRule{}, x, y);
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == 8,
      "direct input execution visits numerical faces only") && ok;
   GenericExplicitOperator<KP>(integrate(InteriorFacets<"m">{}, jump(v)),
      context, AnisotropicRule{}, y);
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == 8,
      "direct output execution visits numerical faces only") && ok;

   // Measure pattern construction and numerical insertion separately; public
   // assembly entry points must perform exactly those traversals, without scans.
   auto target = MakeDGBSRPattern(mesh, space, space, HostBSRBackend<>{});
   const auto pattern_reads = fixture.face_reads.exchange(0, std::memory_order_relaxed);
   auto view = GetHostValuesReadWriteView(target);
   AssembleSparseTarget<KP>(form, context, AnisotropicRule{}, view);
   const auto insertion_reads = fixture.face_reads.exchange(0, std::memory_order_relaxed);
   GenericAssembly<KP>(form, context, AnisotropicRule{}, view);
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == insertion_reads,
      "direct sparse assembly does not scan quadrature compatibility") && ok;
   (void)GenericBSRAssembly<KP>(form, context, AnisotropicRule{}, HostBSRBackend<>{});
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == pattern_reads + insertion_reads,
      "BSR preparation does not scan quadrature compatibility") && ok;
   (void)GenericSGBSRAssembly<KP>(form, context, AnisotropicRule{}, HostBSRBackend<>{});
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == pattern_reads + insertion_reads,
      "SGBSR preparation does not scan quadrature compatibility") && ok;
   AssembleSparseTarget<KP, SparseAssemblyMode::ElementBlockDiagonal>(form, context, AnisotropicRule{}, view);
   const auto diagonal_reads = fixture.face_reads.exchange(0, std::memory_order_relaxed);
   (void)GenericBSRElementBlockDiagonalAssembly<KP>(form, context, AnisotropicRule{}, HostBSRBackend<>{});
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == diagonal_reads,
      "diagonal BSR preparation does not scan quadrature compatibility") && ok;
   (void)MakeRawCOOAssemblyLayout<false, false, true>(mesh, 16);
   const auto raw_layout_reads = fixture.face_reads.exchange(0, std::memory_order_relaxed);
   (void)GenericRawCOOAssembly<KP>(form, context, AnisotropicRule{});
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == raw_layout_reads + insertion_reads,
      "RawCOO preparation does not scan quadrature compatibility") && ok;
   (void)MakeRawCOOElementBlockDiagonalAssemblyLayout<false, false, true>(mesh, 16);
   const auto raw_diagonal_layout_reads = fixture.face_reads.exchange(0, std::memory_order_relaxed);
   (void)GenericRawCOOElementBlockDiagonalAssembly<KP>(form, context, AnisotropicRule{});
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == raw_diagonal_layout_reads + diagonal_reads,
      "diagonal RawCOO preparation does not scan quadrature compatibility") && ok;

#if defined(GENDIL_USE_DEVICE)
   // Validation reads host connectivity; it does not launch this GPU operator.
   using DevicePolicy = DeviceKernelConfiguration<ThreadBlockLayout<4, 4>, 2, 1>;
   const auto device_op = MakeGenericOperator<DevicePolicy>(form, context, AnisotropicRule{});
   ok = Check(fixture.face_reads.load(std::memory_order_relaxed) == 0,
      "GPU operator construction does not scan quadrature compatibility") && ok;
   ValidateOperator(device_op, enabled);
   ok = Check(fixture.face_reads.exchange(0, std::memory_order_relaxed) == 8,
      "GPU-configured operator validates host connectivity explicitly") && ok;
#endif
   return ok;
}

bool CheckInapplicableForms()
{
   Fixture fixture(true);
   const auto mesh = fixture.Mesh();
   const auto space = MakeFiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{}));
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space), MakeIntegrationDomain<"m">(mesh));
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   QuadratureChecks enabled;
   const auto form = integrate(InteriorFacets<"m">{}, jump(u) * jump(v));
   const auto zero = rhs(form);
   static_assert(is_zero<decltype(zero)>);
   ValidateOperator(zero, context, AnisotropicRule{}, enabled);
   ValidateOperator(integrate(Cells<"m">{}, u * v), context, AnisotropicRule{}, enabled);
   ValidateOperator(integrate(BoundaryFacets<"m">{}, u * v), context, AnisotropicRule{}, enabled);
   const auto faces = MakeUnstructuredInteriorFaceConnectivity(mesh);
   const auto partition = MakePartition(MakeCellPart(mesh), MakeInteriorFacePart<0, 0>(faces));
   const auto global_context = MakeWeakFormContext(
      MakeTrialField<"u">(space), MakeIntegrationDomain<"m">(partition));
   ValidateOperator(form, global_context, AnisotropicRule{}, enabled);
   return Check(fixture.face_reads.load(std::memory_order_relaxed) == 0,
      "zero, cell-only, boundary-only and global-facet forms need no local check");
}

void Reject(std::string_view mode)
{
   Fixture fixture(true);
   const auto mesh = fixture.Mesh();
   const auto space = MakeFiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{}));
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space), MakeIntegrationDomain<"m">(mesh));
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   const auto form = integrate(InteriorFacets<"m">{}, jump(u) * jump(v));
   QuadratureChecks enabled;
   if (mode == "reject-operator")
   {
      const auto op = MakeGenericOperator<SerialKernelConfiguration>(form, context, AnisotropicRule{});
      ValidateOperator(op, enabled);
   }
   else if (mode == "reject-form")
      ValidateOperator(form, context, AnisotropicRule{}, enabled);
#if defined(GENDIL_USE_DEVICE)
   else if (mode == "reject-device-operator")
   {
      using DevicePolicy = DeviceKernelConfiguration<ThreadBlockLayout<4, 4>, 2, 1>;
      const auto op = MakeGenericOperator<DevicePolicy>(form, context, AnisotropicRule{});
      ValidateOperator(op, enabled);
   }
#endif
}

} // namespace local_facet_quadrature_validation_test

int main(int argc, char** argv)
{
   using namespace local_facet_quadrature_validation_test;
   if (argc == 2)
   {
      Reject(argv[1]);
      std::cerr << "FAILED: incompatible quadrature was not rejected\n";
      return 1;
   }
   return CheckCompatibility() && CheckConstructionAndInvocation() && CheckInapplicableForms() ? 0 : 1;
}

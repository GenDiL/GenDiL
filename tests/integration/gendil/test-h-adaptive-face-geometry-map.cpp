// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>

using namespace gendil;

namespace
{

constexpr Real tol = 1.0e-11;

bool CheckNear(const std::string& label, const Real got, const Real expected)
{
   const Real err = std::abs(got - expected);
   if (err > tol)
   {
      std::cerr << "FAILED: " << label
                << " got " << got
                << ", expected " << expected
                << ", err " << err << "\n";
      return false;
   }
   return true;
}

bool CheckFar(const std::string& label, const Real a, const Real b)
{
   const Real diff = std::abs(a - b);
   if (diff <= 100.0 * tol)
   {
      std::cerr << "FAILED: " << label
                << " values unexpectedly matched: " << a
                << " vs " << b << "\n";
      return false;
   }
   return true;
}

template <typename X>
Real LinearCoefficient(const X& x)
{
   return 2.0 * x[0] + 3.0 * x[1];
}

Real TestField(const Real x, const Real y)
{
   return 100.0 * x + 10.0 * y;
}

Real MaterializedMappedY(const Real eta)
{
   return 0.25 + 0.5 * eta;
}

template <Integer LocalFaceIndex>
struct MaterializedOneFaceNonconformingConnectivity
{
   static constexpr Integer Dim = 2;
   using geometry = HyperCube<Dim>;
   using orientation_type = Permutation<Dim>;

   static constexpr Integer axis =
      HyperCube< Dim >::GetNormalDimensionIndex( LocalFaceIndex );
   static constexpr int sign =
      HyperCube< Dim >::GetNormalSign( LocalFaceIndex );
   static constexpr Integer minus_local_face_index = LocalFaceIndex;
   static constexpr Integer plus_local_face_index =
      HyperCube< Dim >::GetOppositeFaceIndex( LocalFaceIndex );

   using minus_view_type =
      FaceView<
         std::integral_constant<Integer, minus_local_face_index>,
         geometry,
         orientation_type,
         CanonicalVector<Dim, axis, sign>,
         NonconformingHyperCubeFaceMap<Dim>>;
   using plus_view_type =
      FaceView<
         std::integral_constant<Integer, plus_local_face_index>,
         geometry,
         orientation_type,
         CanonicalVector<Dim, axis, -sign>,
         ConformingFaceMap<Dim>>;
   using face_info_type = GlobalFaceInfo<minus_view_type, plus_view_type>;

   face_info_type info;

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return 1;
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex&) const
   {
      return info;
   }
};

struct TangentialReversalCase
{
   static const char* Name()
   {
      return "tangential reversal";
   }

   static Permutation<2> Orientation()
   {
      return Permutation<2>{{1, -2}};
   }

   static std::array<Real, 2> NativePoint(const Real x_ref, const Real y_ref)
   {
      return {x_ref, 1.0 - y_ref};
   }

   static std::array<LocalIndex, 2> ReferenceDofFromNative(
      const LocalIndex i_native,
      const LocalIndex j_native,
      const LocalIndex p)
   {
      return {i_native, static_cast<LocalIndex>(p - j_native)};
   }
};

struct NormalAxisSwapCase
{
   static const char* Name()
   {
      return "normal/tangential axis swap";
   }

   static Permutation<2> Orientation()
   {
      return Permutation<2>{{2, 1}};
   }

   static std::array<Real, 2> NativePoint(const Real x_ref, const Real y_ref)
   {
      return {y_ref, x_ref};
   }

   static std::array<LocalIndex, 2> ReferenceDofFromNative(
      const LocalIndex i_native,
      const LocalIndex j_native,
      const LocalIndex)
   {
      return {j_native, i_native};
   }
};

bool TestCartesianHAdaptiveGeometryMapCharacterization()
{
   constexpr Integer Dim = 2;
   constexpr Integer MeshFaceIndex = Dim; // +x face on the coarse/minus mesh.
   constexpr Integer q1d = 4;

   const GlobalIndex nxL = 2;
   const GlobalIndex nyL = 2;
   const GlobalIndex nxR = 2;
   const GlobalIndex nyR = 4;

   const Real hx = 1.0 / static_cast<Real>(nxL);
   const Real hyL = 1.0 / static_cast<Real>(nyL);
   const Real hyR = 1.0 / static_cast<Real>(nyR);

   Cartesian2DMesh meshL(hx, hyL, nxL, nyL, Point<2>{0.0, 0.0});
   Cartesian2DMesh meshR(hx, hyR, nxR, nyR, Point<2>{1.0, 0.0});

   NonconformingCartesianIntermeshFaceConnectivity<Dim, MeshFaceIndex>
      iface({nxL, nyL}, {nxR, nyR});

   auto int_rule = MakeIntegrationRule(IntegrationRuleNumPoints<q1d, q1d>{});
   auto face_int_rules = GetFaceIntegrationRules(int_rule);
   auto mesh_face_qd_L = MakeMeshFaceQuadData<decltype(meshL)>(face_int_rules);
   auto mesh_face_qd_R = MakeMeshFaceQuadData<decltype(meshR)>(face_int_rules);

   using MinusLocalFaceQD =
      std::remove_cvref_t<decltype(std::get<MeshFaceIndex>(mesh_face_qd_L))>;
   using MinusTangentialPoints = std::tuple_element_t<1, MinusLocalFaceQD>;

   bool success = true;
   Real integral_one = 0.0;
   const Real expected_interface_measure =
      static_cast<Real>(nyL) * hyL;

   for (GlobalIndex face_index = 0;
        face_index < iface.GetNumberOfFaces();
        ++face_index)
   {
      const auto face_info = iface.GetGlobalFaceInfo(face_index);
      const auto minus_face = face_info.MinusSide();
      const auto plus_face = face_info.PlusSide();
      const auto coarse_cell = meshL.GetCell(minus_face.GetCellIndex());
      auto fine_cell = meshR.GetCell(plus_face.GetCellIndex());
      ApplyOrientationToCell(plus_face.GetOrientation(), fine_cell);

      auto&& minus_qd = GetFacetQuadData(mesh_face_qd_L, minus_face);
      auto&& plus_qd = GetFacetQuadData(mesh_face_qd_R, plus_face);

      for (LocalIndex qy = 0; qy < q1d; ++qy)
      {
         const TensorIndex<Dim> qi{
            GlobalIndex{0},
            static_cast<GlobalIndex>(qy)};

         typename decltype(coarse_cell)::physical_coordinates X_coarse{};
         typename decltype(coarse_cell)::jacobian J_coarse{};
         typename decltype(fine_cell)::physical_coordinates X_fine{};
         typename decltype(fine_cell)::jacobian J_fine{};

         mesh::ComputePhysicalCoordinatesAndJacobian(
            coarse_cell,
            qi,
            minus_qd,
            X_coarse,
            J_coarse);
         mesh::ComputePhysicalCoordinatesAndJacobian(
            fine_cell,
            qi,
            plus_qd,
            X_fine,
            J_fine);

         typename decltype(coarse_cell)::jacobian inv_J_coarse{};
         typename decltype(fine_cell)::jacobian inv_J_fine{};
         const Real det_J_coarse =
            ComputeInverseAndDeterminant(J_coarse, inv_J_coarse);
         ComputeInverseAndDeterminant(J_fine, inv_J_fine);
         const auto facet_geometry_minus =
            ComputeFacetGeometry(
               inv_J_coarse,
               minus_face.GetReferenceNormal(),
               det_J_coarse);

         const Real p_leaf = MinusTangentialPoints::GetCoord(qy);
         const Real subface_origin =
            0.5 * static_cast<Real>(face_index % GlobalIndex{2});
         const Real p_coarse_mapped = subface_origin + 0.5 * p_leaf;
         const Real y_coarse_mapped =
            coarse_cell.origin[1] + coarse_cell.h_y * p_coarse_mapped;
         const Real y_coarse_unmapped =
            coarse_cell.origin[1] + coarse_cell.h_y * p_leaf;

         Point<2> X_expected_mapped{X_coarse[0], y_coarse_mapped};
         Point<2> X_expected_unmapped{X_coarse[0], y_coarse_unmapped};

         // Durable positive assertions: the nonconforming face map is a
         // full Dim-component embedded face-coordinate map.
         const Point<Dim> p_leaf_full{1.0, p_leaf};
         const auto mapped_face_point =
            minus_face.MapReferenceToFaceCoordinates(p_leaf_full);
         const auto mapped_face_point_1d =
            minus_face.template MapReferenceToFaceCoordinates1d<1>(
               Point<1>{p_leaf});
         success =
            CheckNear(
               "nonconforming map preserves the +x normal coordinate",
               mapped_face_point[0],
               1.0) &&
            success;
         success =
            CheckNear(
               "nonconforming map applies origin + size * eta",
               mapped_face_point[1],
               p_coarse_mapped) &&
            success;
         success =
            CheckNear(
               "one-dimensional nonconforming map applies origin + size * eta",
               mapped_face_point_1d[0],
               p_coarse_mapped) &&
            success;
         success =
            CheckNear(
               "nonconforming map measure is the subfacet scale",
               minus_face.Measure(),
               0.5) &&
            success;

         success =
            CheckNear(
               "coarse side is the mapped physical point",
               X_coarse[1],
               X_expected_mapped[1]) &&
            success;
         success =
            CheckNear(
               "fine side is the mapped physical point",
               X_fine[1],
               X_expected_mapped[1]) &&
            success;
         success =
            CheckNear(
               "coarse and fine physical x-coordinates match",
               X_coarse[0],
               X_fine[0]) &&
            success;
         success =
            CheckNear(
               "coarse and fine physical y-coordinates match",
               X_coarse[1],
               X_fine[1]) &&
            success;
         success =
            CheckFar(
               "mapped coarse geometry is not the old unmapped point",
               X_coarse[1],
               X_expected_unmapped[1]) &&
            success;
         success =
            CheckNear(
               "mapped coefficient equals fine-side coefficient",
               LinearCoefficient(X_coarse),
               LinearCoefficient(X_fine)) &&
            success;
         success =
            CheckNear(
               "minus physical gradient x-coordinate",
               (Real{3.0} * J_coarse[0]) * inv_J_coarse[0],
               Real{3.0}) &&
            success;
         success =
            CheckNear(
               "minus physical gradient y-coordinate",
               (Real{-5.0} * J_coarse[1]) * inv_J_coarse[1],
               Real{-5.0}) &&
            success;
         success =
            CheckNear(
               "plus physical gradient x-coordinate",
               (Real{3.0} * J_fine[0]) * inv_J_fine[0],
               Real{3.0}) &&
            success;
         success =
            CheckNear(
               "plus physical gradient y-coordinate",
               (Real{-5.0} * J_fine[1]) * inv_J_fine[1],
               Real{-5.0}) &&
            success;

         integral_one +=
            GetWeight(qi, minus_qd)
          * minus_face.Measure()
          * facet_geometry_minus.det_J_facet;
      }
   }

   success =
      CheckNear(
         "constant integration over full coarse h-adaptive interface",
         integral_one,
         expected_interface_measure) &&
      success;

   return success;
}

template <typename OrientationCase>
bool TestLowLevelOrientationAndNonconformingMapKernels()
{
   constexpr Integer Dim = 2;
   constexpr Integer LocalFaceIndex = Dim; // +x canonical minus face.
   constexpr Integer p = 2;
   constexpr Integer q1d = 4;

   Cartesian2DMesh mesh(1.0, 1, 1, Point<2>{0.0, 0.0});
   auto fe = MakeLobattoFiniteElement(FiniteElementOrders<p, p>{});
   auto fe_space = MakeFiniteElementSpace(mesh, fe, L2Restriction{0});

   using Connectivity =
      MaterializedOneFaceNonconformingConnectivity<LocalFaceIndex>;
   static_assert(mesh::GlobalFaceMeshConnectivity<Connectivity>);

   NonconformingHyperCubeFaceMap<Dim>
      coarse_subface{Point<Dim>{0.0, 0.25}, {1.0, 0.5}};

   // This diagnostic executes only this test-local GlobalFaceMeshConnectivity
   // model plus low-level finite-element helpers: ReadDofs, InterpolateValues,
   // ApplyTestFunctions, WriteAddDofs, and the geometry helper. It does not
   // execute GlobalFaceIterator, a global face finite-element space, or a
   // specialized/generic global face operator.
   typename Connectivity::minus_view_type minus_face{
      0,
      {},
      OrientationCase::Orientation(),
      {},
      coarse_subface};
   typename Connectivity::plus_view_type plus_face{
      0,
      {},
      MakeReferencePermutation<Dim>(),
      {},
      {}};
   Connectivity connectivity{{minus_face, plus_face}};

   const auto face_info = connectivity.GetGlobalFaceInfo(0);

   using KernelPolicy = SerialKernelConfiguration;
   Real shared_data[1] = {};
   KernelContext<KernelPolicy, 0> kernel(shared_data);

   auto int_rule = MakeIntegrationRule(IntegrationRuleNumPoints<q1d, q1d>{});
   auto face_int_rules = GetFaceIntegrationRules(int_rule);
   using ShapeFunctions =
      typename decltype(fe_space)::finite_element_type::shape_functions;
   auto face_qd = MakeFaceDofToQuad<ShapeFunctions, decltype(face_int_rules)>();
   auto mesh_face_qd = MakeMeshFaceQuadData<decltype(mesh)>(face_int_rules);

   Vector input(fe_space.GetNumberOfFiniteElementDofs());
   input = 0.0;
   auto input_view = MakeReadWriteElementTensorView<KernelPolicy>(
      fe_space,
      input);

   using NodalPoints = GaussLobattoLegendrePoints<p + 1>;
   for (LocalIndex j = 0; j <= p; ++j)
   {
      for (LocalIndex i = 0; i <= p; ++i)
      {
         const Real x = NodalPoints::GetCoord(i);
         const Real y = NodalPoints::GetCoord(j);
         input_view(i, j, 0) = 100.0 * x + 10.0 * y;
      }
   }

   const auto input_ro = MakeReadOnlyElementTensorView<KernelPolicy>(
      fe_space,
      input);
   const auto local_dofs = ReadDofs(
      kernel,
      fe_space,
      face_info.MinusSide(),
      input_ro);
   const auto values = InterpolateValues(
      kernel,
      face_info.MinusSide(),
      face_qd,
      local_dofs);

   using LocalFaceQD =
      std::remove_cvref_t<decltype(std::get<LocalFaceIndex>(face_qd))>;
   using TangentialQD = std::tuple_element_t<1, LocalFaceQD>;
   using Shape1D = GaussLobattoLegendreShapeFunctions<p>;

   bool success = true;
   const std::string case_label = OrientationCase::Name();

   for (const LocalIndex qy : {LocalIndex{0}, LocalIndex{1}, LocalIndex{3}})
   {
      const Real p_leaf = TangentialQD::points::GetCoord(qy);
      const Real mapped_y = MaterializedMappedY(p_leaf);
      const Point<Dim> p_leaf_full{1.0, p_leaf};
      const auto mapped_face_point =
         face_info.MinusSide().MapReferenceToFaceCoordinates(p_leaf_full);
      const auto native_mapped =
         OrientationCase::NativePoint(1.0, mapped_y);
      const auto native_unmapped =
         OrientationCase::NativePoint(1.0, p_leaf);
      const Real expected_with_orientation_and_map =
         TestField(native_mapped[0], native_mapped[1]);
      const Real expected_without_map =
         TestField(native_unmapped[0], native_unmapped[1]);

      // Durable positive assertions: the map itself is origin + size * p,
      // and interpolation evaluates the trace at that analytically mapped
      // point after applying the selected orientation.
      success =
         CheckNear(
            case_label + ": nonconforming map preserves the +x normal coordinate",
            mapped_face_point[0],
            1.0) &&
         success;
      success =
         CheckNear(
            case_label + ": nonconforming map applies y = 0.25 + 0.5 * eta",
            mapped_face_point[1],
            mapped_y) &&
         success;
      success =
         CheckNear(
            case_label + ": nonconforming map measure is 0.5",
            face_info.MinusSide().Measure(),
            0.5) &&
         success;

      success =
         CheckNear(
            case_label + ": trace interpolation uses orientation and subface map",
            values(0, qy),
            expected_with_orientation_and_map) &&
         success;
      success =
         CheckFar(
            case_label + ": trace interpolation is not the unmapped orientation-only value",
            values(0, qy),
            expected_without_map) &&
         success;
   }

   auto unit_q_values = MakeSerialRecursiveArray<Real>(
      std::index_sequence<1, q1d>{});
   for (LocalIndex qy = 0; qy < q1d; ++qy)
   {
      unit_q_values(0, qy) = 1.0;
   }

   const auto local_test_dofs = ApplyTestFunctions(
      kernel,
      face_info.MinusSide(),
      face_qd,
      unit_q_values);

   Real expected_local[p + 1][p + 1] = {};
   Real expected_local_unmapped[p + 1][p + 1] = {};
   for (LocalIndex j = 0; j <= p; ++j)
   {
      for (LocalIndex i = 0; i <= p; ++i)
      {
         for (LocalIndex qy = 0; qy < q1d; ++qy)
         {
            const Real p_leaf = TangentialQD::points::GetCoord(qy);
            const Real mapped_y = MaterializedMappedY(p_leaf);
            expected_local[i][j] +=
               Shape1D::ComputeValue(i, Point<1>{1.0}) *
               Shape1D::ComputeValue(j, Point<1>{mapped_y});
            expected_local_unmapped[i][j] +=
               Shape1D::ComputeValue(i, Point<1>{1.0}) *
               Shape1D::ComputeValue(j, Point<1>{p_leaf});
         }

         success =
            CheckNear(
               case_label + ": test-function application uses the subface map",
               local_test_dofs(i, j),
               expected_local[i][j]) &&
            success;
      }
   }

   success =
      CheckFar(
         case_label + ": test-function application is not the unmapped result",
         local_test_dofs(p, 0),
         expected_local_unmapped[p][0]) &&
      success;

   Vector output(fe_space.GetNumberOfFiniteElementDofs());
   output = 0.0;
   auto output_view = MakeReadWriteElementTensorView<KernelPolicy>(
      fe_space,
      output);
   WriteAddDofs(
      kernel,
      fe_space,
      face_info.MinusSide(),
      local_test_dofs,
      output_view);

   for (LocalIndex j_native = 0; j_native <= p; ++j_native)
   {
      for (LocalIndex i_native = 0; i_native <= p; ++i_native)
      {
         const auto reference =
            OrientationCase::ReferenceDofFromNative(
               i_native,
               j_native,
               p);
         success =
            CheckNear(
               case_label + ": residual writeback applies orientation after mapped test functions",
               output_view(i_native, j_native, 0),
               expected_local[reference[0]][reference[1]]) &&
            success;
      }
   }

   const auto cell = mesh.GetCell(face_info.MinusSide().GetCellIndex());
   auto&& mesh_minus_qd =
      GetFacetQuadData(mesh_face_qd, face_info.MinusSide());
   typename decltype(cell)::physical_coordinates X_geom{};
   typename decltype(cell)::jacobian J_geom{};
   TensorIndex<Dim> qi{GlobalIndex{0}, GlobalIndex{1}};
   mesh::ComputePhysicalCoordinatesAndJacobian(
      cell,
      qi,
      mesh_minus_qd,
      X_geom,
      J_geom);
   const Real p_leaf = TangentialQD::points::GetCoord(1);
   const Real mapped_y = MaterializedMappedY(p_leaf);

   success =
      CheckNear(
         case_label + ": materialized-face geometry path uses the subface map",
         X_geom[1],
         mapped_y) &&
      success;
   success =
      CheckFar(
         case_label + ": materialized-face geometry path is not the old unmapped point",
         X_geom[1],
         p_leaf) &&
      success;

   return success;
}

bool TestGenericOperatorP0JumpResidualCancellation()
{
   constexpr Integer Dim = 2;
   constexpr Integer MeshFaceIndex = Dim; // +x face on the coarse/minus mesh.

   const GlobalIndex nxL = 2;
   const GlobalIndex nyL = 2;
   const GlobalIndex nxR = 2;
   const GlobalIndex nyR = 4;

   const Real hx = Real{1.0} / static_cast<Real>(nxL);
   const Real hyL = Real{1.0} / static_cast<Real>(nyL);
   const Real hyR = Real{1.0} / static_cast<Real>(nyR);
   const Real expected_interface_measure =
      static_cast<Real>(nyL) * hyL;

   Cartesian2DMesh meshL(hx, hyL, nxL, nyL, Point<2>{0.0, 0.0});
   Cartesian2DMesh meshR(hx, hyR, nxR, nyR, Point<2>{1.0, 0.0});

   constexpr Integer order = 0;
   FiniteElementOrders<order, order> orders;
   auto fe = MakeLegendreFiniteElement(orders);

   auto fe_space_L = MakeFiniteElementSpace(meshL, fe, L2Restriction{0});
   const Integer ndofsL = fe_space_L.GetNumberOfFiniteElementDofs();
   auto fe_space_R = MakeFiniteElementSpace(meshR, fe, L2Restriction{ndofsL});
   const Integer ndofsR = fe_space_R.GetNumberOfFiniteElementDofs();

   NonconformingCartesianIntermeshFaceConnectivity<Dim, MeshFaceIndex>
      iface({nxL, nyL}, {nxR, nyR});
   auto face_fes =
      MakeGlobalInteriorFaceFiniteElementSpace(
         fe_space_L,
         fe_space_R,
         iface);
   auto mixed =
      MakeMixedFiniteElementSpace(
         fe_space_L,
         fe_space_R,
         face_fes);

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"mesh"> interior_facets;
   auto weak_form = integrate(interior_facets, jump(u) * jump(v));
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField<"u">(mixed),
         MakeIntegrationDomain<"mesh">(mixed));

   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<1, 1>{});
   using KernelPolicy = SerialKernelConfiguration;
   auto op =
      MakeGenericOperator<KernelPolicy>(
         weak_form,
         wf_context,
         integration_rule);

   Vector x(mixed.GetNumberOfFiniteElementDofs());
   x = 0.0;
   Real* x_data = x.WriteHostData();
   for (Integer i = 0; i < ndofsL; ++i)
   {
      x_data[i] = 1.0;
   }

   Vector y(mixed.GetNumberOfFiniteElementDofs());
   y = 0.0;
   op(x, y);

   const Real* y_data = y.ReadHostData();
   Real minus_sum = 0.0;
   Real plus_sum = 0.0;
   for (Integer i = 0; i < ndofsL; ++i)
   {
      minus_sum += y_data[i];
   }
   for (Integer i = 0; i < ndofsR; ++i)
   {
      plus_sum += y_data[ndofsL + i];
   }

   bool success = true;
   success =
      CheckNear(
         "p0 h-adaptive jump minus residual magnitude",
         minus_sum,
         expected_interface_measure) &&
      success;
   success =
      CheckNear(
         "p0 h-adaptive jump plus residual magnitude",
         plus_sum,
         -expected_interface_measure) &&
      success;
   success =
      CheckNear(
         "p0 h-adaptive jump residuals cancel globally",
         minus_sum + plus_sum,
         Real{0.0}) &&
      success;

   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestCartesianHAdaptiveGeometryMapCharacterization() && success;
   success =
      TestLowLevelOrientationAndNonconformingMapKernels<
         TangentialReversalCase>() &&
      success;
   success =
      TestLowLevelOrientationAndNonconformingMapKernels<
         NormalAxisSwapCase>() &&
      success;
   success = TestGenericOperatorP0JumpResidualCancellation() && success;

   if (!success)
   {
      return 1;
   }

   std::cout
      << "H-adaptive face geometry-map diagnostic checks passed.\n";
   return 0;
}

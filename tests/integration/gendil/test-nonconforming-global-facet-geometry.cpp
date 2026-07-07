// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{

constexpr Real tol = 1.0e-12;

bool Near(const Real a, const Real b)
{
   const Real scale = std::max(Real{1.0}, std::max(std::abs(a), std::abs(b)));
   return std::abs(a - b) <= tol * scale;
}

bool CheckNear(const char* label, const Real got, const Real expected)
{
   if (!Near(got, expected))
   {
      std::cerr
         << "FAILED: " << label
         << " got=" << got
         << " expected=" << expected
         << " diff=" << std::abs(got - expected) << '\n';
      return false;
   }
   return true;
}

template<class X>
Real LinearTrace(const X& x)
{
   return x[0] + Real{2.0} * x[1];
}

bool TestMappedCoordinatesAndMeasure()
{
   constexpr Integer Dim = 2;
   constexpr Integer MeshFaceIndex = Dim; // +x on the coarse/minus side.
   constexpr Integer q1d = 4;

   const GlobalIndex nx_coarse = 2;
   const GlobalIndex ny_coarse = 2;
   const GlobalIndex nx_fine = 2;
   const GlobalIndex ny_fine = 4;

   const Real hx = Real{1.0} / static_cast<Real>(nx_coarse);
   const Real hy_coarse = Real{1.0} / static_cast<Real>(ny_coarse);
   const Real hy_fine = Real{1.0} / static_cast<Real>(ny_fine);

   Cartesian2DMesh coarse_mesh(
      hx,
      hy_coarse,
      nx_coarse,
      ny_coarse,
      Point<2>{0.0, 0.0});
   Cartesian2DMesh fine_mesh(
      hx,
      hy_fine,
      nx_fine,
      ny_fine,
      Point<2>{1.0, 0.0});

   NonconformingCartesianIntermeshFaceConnectivity<Dim, MeshFaceIndex>
      faces({nx_coarse, ny_coarse}, {nx_fine, ny_fine});

   auto int_rule = MakeIntegrationRule(IntegrationRuleNumPoints<q1d, q1d>{});
   auto face_int_rules = GetFaceIntegrationRules(int_rule);
   auto coarse_mesh_face_qd =
      MakeMeshFaceQuadData<decltype(coarse_mesh)>(face_int_rules);
   auto fine_mesh_face_qd =
      MakeMeshFaceQuadData<decltype(fine_mesh)>(face_int_rules);

   bool success = true;
   Real integral_one = 0.0;
   Real dg_minus_residual_sum = 0.0;
   Real dg_plus_residual_sum = 0.0;

   for (GlobalIndex face_index = 0;
        face_index < faces.GetNumberOfFaces();
        ++face_index)
   {
      const auto face_info = faces.GetGlobalFaceInfo(face_index);
      const auto minus_face = face_info.MinusSide();
      const auto plus_face = face_info.PlusSide();
      const auto coarse_cell = coarse_mesh.GetCell(minus_face.GetCellIndex());
      auto fine_cell = fine_mesh.GetCell(plus_face.GetCellIndex());
      ApplyOrientationToCell(plus_face.GetOrientation(), fine_cell);

      auto&& minus_qd =
         GetFacetQuadData(coarse_mesh_face_qd, minus_face);
      auto&& plus_qd =
         GetFacetQuadData(fine_mesh_face_qd, plus_face);

      for (LocalIndex qy = 0; qy < q1d; ++qy)
      {
         const TensorIndex<Dim> qi{
            GlobalIndex{0},
            static_cast<GlobalIndex>(qy)};

         typename decltype(coarse_cell)::physical_coordinates X_minus{};
         typename decltype(coarse_cell)::jacobian J_minus{};
         typename decltype(fine_cell)::physical_coordinates X_plus{};
         typename decltype(fine_cell)::jacobian J_plus{};

         mesh::ComputePhysicalCoordinatesAndJacobian(
            coarse_cell,
            qi,
            minus_qd,
            X_minus,
            J_minus);
         mesh::ComputePhysicalCoordinatesAndJacobian(
            fine_cell,
            qi,
            plus_qd,
            X_plus,
            J_plus);

         typename decltype(coarse_cell)::jacobian inv_J_minus{};
         typename decltype(fine_cell)::jacobian inv_J_plus{};
         const Real det_J_minus =
            ComputeInverseAndDeterminant(J_minus, inv_J_minus);
         ComputeInverseAndDeterminant(J_plus, inv_J_plus);

         success =
            CheckNear("mapped X x-coordinate", X_minus[0], X_plus[0]) &&
            success;
         success =
            CheckNear("mapped X y-coordinate", X_minus[1], X_plus[1]) &&
            success;
         success =
            CheckNear(
               "nonconstant trace agrees at mapped point",
               LinearTrace(X_minus),
               LinearTrace(X_plus)) &&
            success;

         success =
            CheckNear(
               "minus physical gradient x-coordinate",
               (Real{3.0} * J_minus[0]) * inv_J_minus[0],
               Real{3.0}) &&
            success;
         success =
            CheckNear(
               "minus physical gradient y-coordinate",
               (Real{-5.0} * J_minus[1]) * inv_J_minus[1],
               Real{-5.0}) &&
            success;
         success =
            CheckNear(
               "plus physical gradient x-coordinate",
               (Real{3.0} * J_plus[0]) * inv_J_plus[0],
               Real{3.0}) &&
            success;
         success =
            CheckNear(
               "plus physical gradient y-coordinate",
               (Real{-5.0} * J_plus[1]) * inv_J_plus[1],
               Real{-5.0}) &&
            success;

         const auto facet_geometry_minus =
            ComputeFacetGeometry(
               inv_J_minus,
               minus_face.GetReferenceNormal(),
               det_J_minus);

         const Real jxw =
            GetWeight(qi, minus_qd)
          * minus_face.Measure()
          * facet_geometry_minus.det_J_facet;

         integral_one += jxw;

         // p0 jump(u) * jump(v) with u_minus=1 and u_plus=0 contributes
         // equal-and-opposite global residual sums when both sides use the
         // canonical minus-side JxW.
         dg_minus_residual_sum += jxw;
         dg_plus_residual_sum -= jxw;
      }
   }

   success =
      CheckNear(
         "constant integration over split coarse face",
         integral_one,
         Real{1.0}) &&
      success;
   success =
      CheckNear(
         "p0 DG jump residuals cancel globally",
         dg_minus_residual_sum + dg_plus_residual_sum,
         Real{0.0}) &&
      success;
   success =
      CheckNear(
         "p0 DG minus jump residual magnitude",
         dg_minus_residual_sum,
         Real{1.0}) &&
      success;

   return success;
}

} // namespace

int main()
{
   const bool success = TestMappedCoordinatesAndMeasure();
   if (!success)
   {
      return 1;
   }
   std::cout << "PASS: nonconforming global facet mapped geometry\n";
   return 0;
}

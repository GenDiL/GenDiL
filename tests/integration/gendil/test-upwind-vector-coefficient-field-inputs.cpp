// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <tuple>

using namespace gendil;

namespace
{

struct CellVectorField
{
   std::array<Real, 2> offset;
   std::array<Real, 2> grad0;
   std::array<Real, 2> grad1;
};

CellVectorField CellBetaParameters(
   const GlobalIndex ix,
   const GlobalIndex iy)
{
   const Real x_cell = static_cast<Real>(ix);
   const Real y_cell = static_cast<Real>(iy);

   return CellVectorField{
      std::array<Real, 2>{
         -0.35 + 0.95 * x_cell - 0.45 * y_cell,
          0.45 - 0.75 * x_cell + 1.10 * y_cell},
      std::array<Real, 2>{0.35, -0.20},
      std::array<Real, 2>{-0.15, 0.55}
   };
}

std::array<Real, 2> ManufacturedBeta(
   const GlobalIndex ix,
   const GlobalIndex iy,
   const Real x,
   const Real y)
{
   const auto p = CellBetaParameters(ix, iy);
   return {
      p.offset[0] + p.grad0[0] * x + p.grad0[1] * y,
      p.offset[1] + p.grad1[0] * x + p.grad1[1] * y
   };
}

Real Dot2(const std::array<Real, 2>& a, const std::array<Real, 2>& b)
{
   return a[0] * b[0] + a[1] * b[1];
}

template <Integer order, typename FiniteElementSpace>
void FillDiscontinuousBetaField(
   const FiniteElementSpace& fe_space,
   Vector& field)
{
   using DofPoints = GaussLobattoLegendrePoints<order + 1>;

   auto dofs =
      MakeWriteOnlyElementTensorView<SerialKernelConfiguration>(
         fe_space,
         field);

   const GlobalIndex nx = fe_space.sizes[0];

   for (GlobalIndex element_index = 0;
        element_index < fe_space.GetNumberOfCells();
        ++element_index)
   {
      const GlobalIndex ix = element_index % nx;
      const GlobalIndex iy = element_index / nx;

      for (GlobalIndex j = 0; j < order + 1; ++j)
      {
         for (GlobalIndex i = 0; i < order + 1; ++i)
         {
            const Real x =
               fe_space.mesh_origin[0] +
               fe_space.h[0] * (ix + DofPoints::GetCoord(i));
            const Real y =
               fe_space.mesh_origin[1] +
               fe_space.h[1] * (iy + DofPoints::GetCoord(j));

            const auto beta = ManufacturedBeta(ix, iy, x, y);
            std::get<0>(dofs)(i, j, element_index) = beta[0];
            std::get<1>(dofs)(i, j, element_index) = beta[1];
         }
      }
   }
}

template <Integer order, typename FiniteElementSpace>
void FillDiscontinuousTrialField(
   const FiniteElementSpace& fe_space,
   Vector& field)
{
   using DofPoints = GaussLobattoLegendrePoints<order + 1>;

   auto dofs =
      MakeWriteOnlyElementTensorView<SerialKernelConfiguration>(
         fe_space,
         field);

   const GlobalIndex nx = fe_space.sizes[0];

   for (GlobalIndex element_index = 0;
        element_index < fe_space.GetNumberOfCells();
        ++element_index)
   {
      const GlobalIndex ix = element_index % nx;
      const GlobalIndex iy = element_index / nx;

      for (GlobalIndex j = 0; j < order + 1; ++j)
      {
         for (GlobalIndex i = 0; i < order + 1; ++i)
         {
            const Real x =
               fe_space.mesh_origin[0] +
               fe_space.h[0] * (ix + DofPoints::GetCoord(i));
            const Real y =
               fe_space.mesh_origin[1] +
               fe_space.h[1] * (iy + DofPoints::GetCoord(j));

            dofs(i, j, element_index) =
               1.0 + 0.23 * x - 0.31 * y + 0.09 * x * y
                   + 0.21 * static_cast<Real>(ix)
                   - 0.13 * static_cast<Real>(iy)
                   + 0.04 * static_cast<Real>(ix + 2 * iy);
         }
      }
   }
}

template <Integer order>
Real ShapeValue(const GlobalIndex dof_index, const Real ref_coord)
{
   using Basis = GaussLobattoLegendreShapeFunctions<order>;
   return Basis::ComputeValue(dof_index, Point<1>{ref_coord});
}

template <Integer order>
Real ShapeGradient(const GlobalIndex dof_index, const Real ref_coord)
{
   using Basis = GaussLobattoLegendreShapeFunctions<order>;
   return Basis::ComputeGradientValue(dof_index, Point<1>{ref_coord});
}

template <Integer order, typename ElementDofs>
Real EvaluateScalarDofsAt(
   const ElementDofs& dofs,
   const GlobalIndex element_index,
   const Real xi,
   const Real eta)
{
   Real value = 0.0;

   for (GlobalIndex j = 0; j < order + 1; ++j)
   {
      for (GlobalIndex i = 0; i < order + 1; ++i)
      {
         value += dofs(i, j, element_index) *
                  ShapeValue<order>(i, xi) *
                  ShapeValue<order>(j, eta);
      }
   }

   return value;
}

template <Integer order, typename FiniteElementSpace>
void AccumulateCellOracle(
   const FiniteElementSpace& fe_space,
   const Vector& u,
   Vector& y)
{
   constexpr Integer num_quad_1d = order + 3;
   using QuadPoints = GaussLegendrePoints<num_quad_1d>;

   auto u_dofs =
      MakeReadOnlyElementTensorView<SerialKernelConfiguration>(
         fe_space,
         u);
   auto y_dofs =
      MakeReadWriteElementTensorView<SerialKernelConfiguration>(
         fe_space,
         y);

   const GlobalIndex nx = fe_space.sizes[0];
   const Real hx = fe_space.h[0];
   const Real hy = fe_space.h[1];
   const Real det_J = hx * hy;

   for (GlobalIndex element_index = 0;
        element_index < fe_space.GetNumberOfCells();
        ++element_index)
   {
      const GlobalIndex ix = element_index % nx;
      const GlobalIndex iy = element_index / nx;

      for (GlobalIndex qy = 0; qy < num_quad_1d; ++qy)
      {
         for (GlobalIndex qx = 0; qx < num_quad_1d; ++qx)
         {
            const Real xi = QuadPoints::GetCoord(qx);
            const Real eta = QuadPoints::GetCoord(qy);
            const Real q_weight =
               QuadPoints::GetWeight(qx) * QuadPoints::GetWeight(qy);

            const Real x_q =
               fe_space.mesh_origin[0] + hx * (ix + xi);
            const Real y_q =
               fe_space.mesh_origin[1] + hy * (iy + eta);

            const auto beta = ManufacturedBeta(ix, iy, x_q, y_q);
            const Real u_q =
               EvaluateScalarDofsAt<order>(
                  u_dofs,
                  element_index,
                  xi,
                  eta);

            for (GlobalIndex j = 0; j < order + 1; ++j)
            {
               for (GlobalIndex i = 0; i < order + 1; ++i)
               {
                  const Real dphi_dx =
                     ShapeGradient<order>(i, xi) *
                     ShapeValue<order>(j, eta) / hx;
                  const Real dphi_dy =
                     ShapeValue<order>(i, xi) *
                     ShapeGradient<order>(j, eta) / hy;

                  const Real beta_dot_grad_phi =
                     beta[0] * dphi_dx + beta[1] * dphi_dy;

                  y_dofs(i, j, element_index) +=
                     -u_q * beta_dot_grad_phi * q_weight * det_J;
               }
            }
         }
      }
   }
}

template <Integer order, typename FiniteElementSpace>
void AccumulateInteriorUpwindOracle(
   const FiniteElementSpace& fe_space,
   const Vector& u,
   Vector& y,
   const bool use_averaged_beta)
{
   constexpr Integer num_quad_1d = order + 3;
   using QuadPoints = GaussLegendrePoints<num_quad_1d>;

   auto u_dofs =
      MakeReadOnlyElementTensorView<SerialKernelConfiguration>(
         fe_space,
         u);
   auto y_dofs =
      MakeReadWriteElementTensorView<SerialKernelConfiguration>(
         fe_space,
         y);

   const GlobalIndex nx = fe_space.sizes[0];
   const GlobalIndex ny = fe_space.sizes[1];

   for (GlobalIndex iy = 0; iy < ny; ++iy)
   {
      for (GlobalIndex ix = 0; ix < nx; ++ix)
      {
         const GlobalIndex element_index = ix + nx * iy;

         for (GlobalIndex face_index = 0; face_index < 4; ++face_index)
         {
            const GlobalIndex axis = face_index % 2;
            const bool plus_side_face = face_index >= 2;

            if ((axis == 0 && !plus_side_face && ix == 0) ||
                (axis == 0 &&  plus_side_face && ix + 1 == nx) ||
                (axis == 1 && !plus_side_face && iy == 0) ||
                (axis == 1 &&  plus_side_face && iy + 1 == ny))
            {
               continue;
            }

            const GlobalIndex neighbor_ix =
               axis == 0 ? (plus_side_face ? ix + 1 : ix - 1) : ix;
            const GlobalIndex neighbor_iy =
               axis == 1 ? (plus_side_face ? iy + 1 : iy - 1) : iy;
            const GlobalIndex neighbor_index = neighbor_ix + nx * neighbor_iy;

            const Real face_ref_coord = plus_side_face ? 1.0 : 0.0;
            const Real neighbor_ref_coord = plus_side_face ? 0.0 : 1.0;
            const Real ds = fe_space.h[1 - axis];

            const std::array<Real, 2> normal{
               axis == 0 ? (plus_side_face ? 1.0 : -1.0) : 0.0,
               axis == 1 ? (plus_side_face ? 1.0 : -1.0) : 0.0
            };

            for (GlobalIndex q = 0; q < num_quad_1d; ++q)
            {
               const Real tangent_ref_coord = QuadPoints::GetCoord(q);
               const Real q_weight = QuadPoints::GetWeight(q);

               const Real xi_minus =
                  axis == 0 ? face_ref_coord : tangent_ref_coord;
               const Real eta_minus =
                  axis == 1 ? face_ref_coord : tangent_ref_coord;
               const Real xi_plus =
                  axis == 0 ? neighbor_ref_coord : tangent_ref_coord;
               const Real eta_plus =
                  axis == 1 ? neighbor_ref_coord : tangent_ref_coord;

               const Real x_q =
                  fe_space.mesh_origin[0] + fe_space.h[0] * (ix + xi_minus);
               const Real y_q =
                  fe_space.mesh_origin[1] + fe_space.h[1] * (iy + eta_minus);

               const auto beta_minus =
                  ManufacturedBeta(ix, iy, x_q, y_q);
               const auto beta_plus =
                  ManufacturedBeta(neighbor_ix, neighbor_iy, x_q, y_q);

               const std::array<Real, 2> beta_facet{
                  use_averaged_beta
                     ? 0.5 * (beta_minus[0] + beta_plus[0])
                     : beta_minus[0],
                  use_averaged_beta
                     ? 0.5 * (beta_minus[1] + beta_plus[1])
                     : beta_minus[1]
               };

               const Real beta_n = Dot2(beta_facet, normal);

               const Real u_minus =
                  EvaluateScalarDofsAt<order>(
                     u_dofs,
                     element_index,
                     xi_minus,
                     eta_minus);
               const Real u_plus =
                  EvaluateScalarDofsAt<order>(
                     u_dofs,
                     neighbor_index,
                     xi_plus,
                     eta_plus);

               const Real u_upwind = beta_n >= 0.0 ? u_minus : u_plus;
               const Real weighted_value = beta_n * u_upwind * q_weight * ds;

               for (GlobalIndex j = 0; j < order + 1; ++j)
               {
                  for (GlobalIndex i = 0; i < order + 1; ++i)
                  {
                     const Real phi =
                        ShapeValue<order>(i, xi_minus) *
                        ShapeValue<order>(j, eta_minus);
                     y_dofs(i, j, element_index) += weighted_value * phi;
                  }
               }
            }
         }
      }
   }
}

template <Integer order, typename FiniteElementSpace>
Vector ApplyUpwindAdvectionOracle(
   const FiniteElementSpace& fe_space,
   const Vector& u,
   const bool use_averaged_beta)
{
   Vector y(fe_space.GetNumberOfFiniteElementDofs());
   y = 0.0;

   AccumulateCellOracle<order>(fe_space, u, y);
   AccumulateInteriorUpwindOracle<order>(
      fe_space,
      u,
      y,
      use_averaged_beta);

   return y;
}

template <typename OperatorType>
Vector ApplyOperator(const OperatorType& op, const Vector& x)
{
   Vector y(x.Size());
   y = 0.0;
   op(x, y);
   return y;
}

template <typename VectorType>
Real AbsoluteL2Error(const VectorType& a, const VectorType& b)
{
   GENDIL_VERIFY(a.Size() == b.Size(), "Vector sizes do not match.");

   Real err_sq = 0.0;
   for (Integer i = 0; i < a.Size(); ++i)
   {
      const Real d = a[i] - b[i];
      err_sq += d * d;
   }

   return std::sqrt(err_sq);
}

template <typename VectorType>
Real L2Norm(const VectorType& a)
{
   Real norm_sq = 0.0;
   for (Integer i = 0; i < a.Size(); ++i)
   {
      norm_sq += a[i] * a[i];
   }

   return std::sqrt(norm_sq);
}

template <typename VectorType>
Real RelativeL2Error(const VectorType& a, const VectorType& b)
{
   const Real err = AbsoluteL2Error(a, b);
   const Real ref = L2Norm(b);
   return ref == 0.0 ? err : err / ref;
}

template <Integer order>
int TestUpwindVectorCoefficientFieldInputs()
{
   std::cout
      << "\n=== Upwind vector coefficient field-input test, order = "
      << order << " ===\n";

   constexpr GlobalIndex nx = 3;
   constexpr GlobalIndex ny = 2;

   CartesianMesh<2> mesh({nx, ny}, {0.45, 0.80}, {0.10, -0.20}, false);

   FiniteElementOrders<order, order> orders;
   auto scalar_finite_element = MakeLobattoFiniteElement(orders);
   auto vector_finite_element =
      MakeVectorFiniteElement(
         scalar_finite_element,
         scalar_finite_element);

   auto scalar_fe_space =
      MakeFiniteElementSpace(mesh, scalar_finite_element);
   auto vector_fe_space =
      MakeFiniteElementSpace(mesh, vector_finite_element);

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

   using KernelPolicy = SerialKernelConfiguration;

   Vector u_h(scalar_fe_space.GetNumberOfFiniteElementDofs());
   FillDiscontinuousTrialField<order>(scalar_fe_space, u_h);

   Vector beta_h(vector_fe_space.GetNumberOfFiniteElementDofs());
   FillDiscontinuousBetaField<order>(vector_fe_space, beta_h);

   auto beta_view =
      MakeReadOnlyElementTensorView<KernelPolicy>(
         vector_fe_space,
         beta_h);

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   Cells<"mesh"> cells;
   InteriorFacets<"mesh"> interior_facets;

   auto beta =
      MakeVectorCoefficient<"beta", FieldValue<"beta_field">>(
         [] (const auto& b) -> std::array<Real, 2>
         {
            return { b(0), b(1) };
         });

   auto cell_form =
      integrate(cells, -u * dot(beta, grad(v)));
   auto facet_form =
      integrate(interior_facets, upwind(average(beta), u) * jump(v));
   auto advection_form = cell_form + facet_form;

   static_assert(!requires_plus_side_jacobian_v<decltype(facet_form)>);

   auto wf_ctx =
      MakeWeakFormContext(
         MakeTrialField<"u">(scalar_fe_space),
         MakeFiniteElementField<"beta_field">(vector_fe_space, beta_view),
         MakeIntegrationDomain<"mesh">(scalar_fe_space));

   auto op =
      MakeGenericOperator<KernelPolicy>(
         advection_form,
         wf_ctx,
         integration_rule);

   const Vector y_generic = ApplyOperator(op, u_h);
   const Vector y_ref =
      ApplyUpwindAdvectionOracle<order>(
         scalar_fe_space,
         u_h,
         true);
   const Vector y_minus_only =
      ApplyUpwindAdvectionOracle<order>(
         scalar_fe_space,
         u_h,
         false);

   const Real rel_err = RelativeL2Error(y_generic, y_ref);
   const Real minus_only_separation = RelativeL2Error(y_minus_only, y_ref);

   std::cout << "  Generic vs averaged-beta oracle error = "
             << rel_err << "\n";
   std::cout << "  Minus-only beta diagnostic separation = "
             << minus_only_separation << "\n";

   const Real tol = 1e-11;
   const Real separation_tol = 1e-3;
   if (rel_err > tol)
   {
      std::cerr
         << "FAILED: generic upwind advection does not match oracle.\n";
      return 1;
   }
   if (minus_only_separation < separation_tol)
   {
      std::cerr
         << "FAILED: manufactured data does not distinguish averaged beta "
         << "from minus-only beta.\n";
      return 1;
   }

   return 0;
}

} // namespace

int main()
{
   if (TestUpwindVectorCoefficientFieldInputs<1>() != 0) { return 1; }
   if (TestUpwindVectorCoefficientFieldInputs<2>() != 0) { return 1; }

   std::cout
      << "\nAll upwind vector coefficient field-input tests passed.\n";
   return 0;
}

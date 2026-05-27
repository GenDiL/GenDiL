// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{

struct CellAffineField
{
   Real offset;
   std::array<Real, 2> gradient;
};

CellAffineField CellMuParameters(
   const GlobalIndex ix,
   const GlobalIndex iy)
{
   const Real x_cell = static_cast<Real>(ix);
   const Real y_cell = static_cast<Real>(iy);

   return CellAffineField{
      Real(2.15) + Real(0.37) * x_cell - Real(0.23) * y_cell
                 + Real(0.08) * static_cast<Real>(ix + 2 * iy),
      std::array<Real, 2>{
         Real(0.42) + Real(0.05) * x_cell,
        -Real(0.18) + Real(0.04) * y_cell}
   };
}

Real ManufacturedMu(
   const GlobalIndex ix,
   const GlobalIndex iy,
   const Real x,
   const Real y)
{
   const auto p = CellMuParameters(ix, iy);
   return p.offset + p.gradient[0] * x + p.gradient[1] * y;
}

Real ManufacturedU(
   const GlobalIndex ix,
   const GlobalIndex iy,
   const Real x,
   const Real y)
{
   return Real(0.80)
        + Real(0.21) * x
        - Real(0.34) * y
        + Real(0.09) * x * y
        + Real(0.04) * x * x
        - Real(0.03) * y * y
        + Real(0.29) * static_cast<Real>(ix)
        - Real(0.17) * static_cast<Real>(iy)
        + Real(0.05) * static_cast<Real>(ix + 2 * iy);
}

template <Integer order, typename FiniteElementSpace>
void FillDiscontinuousMuField(
   const FiniteElementSpace& fe_space,
   Vector& field)
{
   using DofPoints = GaussLobattoLegendrePoints<order + 1>;

   auto dofs =
      MakeWriteOnlyElementVectorView<SerialKernelConfiguration>(
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

            dofs(i, j, element_index) = ManufacturedMu(ix, iy, x, y);
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
      MakeWriteOnlyElementVectorView<SerialKernelConfiguration>(
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

            dofs(i, j, element_index) = ManufacturedU(ix, iy, x, y);
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

template <Integer order, typename ElementDofs>
std::array<Real, 2> EvaluateScalarGradientAt(
   const ElementDofs& dofs,
   const GlobalIndex element_index,
   const Real xi,
   const Real eta,
   const Real hx,
   const Real hy)
{
   std::array<Real, 2> gradient{0.0, 0.0};

   for (GlobalIndex j = 0; j < order + 1; ++j)
   {
      for (GlobalIndex i = 0; i < order + 1; ++i)
      {
         const Real value = dofs(i, j, element_index);

         gradient[0] += value *
                        ShapeGradient<order>(i, xi) *
                        ShapeValue<order>(j, eta) / hx;
         gradient[1] += value *
                        ShapeValue<order>(i, xi) *
                        ShapeGradient<order>(j, eta) / hy;
      }
   }

   return gradient;
}

Real Dot2(const std::array<Real, 2>& a, const std::array<Real, 2>& b)
{
   return a[0] * b[0] + a[1] * b[1];
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
      MakeReadOnlyElementVectorView<SerialKernelConfiguration>(
         fe_space,
         u);
   auto y_dofs =
      MakeReadWriteElementVectorView<SerialKernelConfiguration>(
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

            const Real x_q = fe_space.mesh_origin[0] + hx * (ix + xi);
            const Real y_q = fe_space.mesh_origin[1] + hy * (iy + eta);

            const Real mu_q = ManufacturedMu(ix, iy, x_q, y_q);
            const auto grad_u =
               EvaluateScalarGradientAt<order>(
                  u_dofs,
                  element_index,
                  xi,
                  eta,
                  hx,
                  hy);

            for (GlobalIndex j = 0; j < order + 1; ++j)
            {
               for (GlobalIndex i = 0; i < order + 1; ++i)
               {
                  const std::array<Real, 2> grad_phi{
                     ShapeGradient<order>(i, xi) *
                     ShapeValue<order>(j, eta) / hx,
                     ShapeValue<order>(i, xi) *
                     ShapeGradient<order>(j, eta) / hy
                  };

                  y_dofs(i, j, element_index) +=
                     mu_q * Dot2(grad_u, grad_phi) * q_weight * det_J;
               }
            }
         }
      }
   }
}

template <Integer order, typename FiniteElementSpace>
void AccumulateInteriorSIPDGOracle(
   const FiniteElementSpace& fe_space,
   const Vector& u,
   Vector& y,
   const Real sigma,
   const Real kappa,
   const bool use_averaged_mu)
{
   constexpr Integer num_quad_1d = order + 3;
   using QuadPoints = GaussLegendrePoints<num_quad_1d>;

   auto u_dofs =
      MakeReadOnlyElementVectorView<SerialKernelConfiguration>(
         fe_space,
         u);
   auto y_dofs =
      MakeReadWriteElementVectorView<SerialKernelConfiguration>(
         fe_space,
         y);

   const GlobalIndex nx = fe_space.sizes[0];
   const GlobalIndex ny = fe_space.sizes[1];
   const Real hx = fe_space.h[0];
   const Real hy = fe_space.h[1];

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
            const Real h_inv = axis == 0 ? 1.0 / hx : 1.0 / hy;
            const Real tau = kappa * h_inv;

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

               const Real x_q = fe_space.mesh_origin[0] + hx * (ix + xi_minus);
               const Real y_q = fe_space.mesh_origin[1] + hy * (iy + eta_minus);

               const Real mu_minus = ManufacturedMu(ix, iy, x_q, y_q);
               const Real mu_plus =
                  ManufacturedMu(neighbor_ix, neighbor_iy, x_q, y_q);
               const Real avg_mu =
                  use_averaged_mu ? 0.5 * (mu_minus + mu_plus) : mu_minus;

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
               const Real jump_u = u_minus - u_plus;

               const auto grad_u_minus =
                  EvaluateScalarGradientAt<order>(
                     u_dofs,
                     element_index,
                     xi_minus,
                     eta_minus,
                     hx,
                     hy);
               const auto grad_u_plus =
                  EvaluateScalarGradientAt<order>(
                     u_dofs,
                     neighbor_index,
                     xi_plus,
                     eta_plus,
                     hx,
                     hy);

               const Real avg_mu_flux =
                  use_averaged_mu
                     ? 0.5 * (
                          mu_minus * Dot2(grad_u_minus, normal) +
                          mu_plus  * Dot2(grad_u_plus, normal))
                     : 0.5 * mu_minus * (
                          Dot2(grad_u_minus, normal) +
                          Dot2(grad_u_plus, normal));

               for (GlobalIndex j = 0; j < order + 1; ++j)
               {
                  for (GlobalIndex i = 0; i < order + 1; ++i)
                  {
                     const Real phi =
                        ShapeValue<order>(i, xi_minus) *
                        ShapeValue<order>(j, eta_minus);
                     const std::array<Real, 2> grad_phi{
                        ShapeGradient<order>(i, xi_minus) *
                        ShapeValue<order>(j, eta_minus) / hx,
                        ShapeValue<order>(i, xi_minus) *
                        ShapeGradient<order>(j, eta_minus) / hy
                     };
                     const Real grad_phi_n = Dot2(grad_phi, normal);

                     const Real consistency = -avg_mu_flux * phi;
                     const Real symmetry =
                        sigma * jump_u * 0.5 * mu_minus * grad_phi_n;
                     const Real penalty = tau * avg_mu * jump_u * phi;

                     y_dofs(i, j, element_index) +=
                        (consistency + symmetry + penalty) * q_weight * ds;
                  }
               }
            }
         }
      }
   }
}

template <Integer order, typename FiniteElementSpace>
Vector ApplySIPDGOracle(
   const FiniteElementSpace& fe_space,
   const Vector& u,
   const Real sigma,
   const Real kappa,
   const bool use_averaged_mu)
{
   Vector y(fe_space.GetNumberOfFiniteElementDofs());
   y = 0.0;

   AccumulateCellOracle<order>(fe_space, u, y);
   AccumulateInteriorSIPDGOracle<order>(
      fe_space,
      u,
      y,
      sigma,
      kappa,
      use_averaged_mu);

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
int TestSIPDGCoefficientFieldInputs()
{
   std::cout
      << "\n=== SIPDG coefficient field-input test, order = "
      << order << " ===\n";

   constexpr GlobalIndex nx = 3;
   constexpr GlobalIndex ny = 2;

   CartesianMesh<2> mesh({nx, ny}, {0.45, 0.80}, {0.10, -0.20}, false);

   FiniteElementOrders<order, order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

   using KernelPolicy = SerialKernelConfiguration;

   Vector u_h(fe_space.GetNumberOfFiniteElementDofs());
   FillDiscontinuousTrialField<order>(fe_space, u_h);

   Vector mu_h(fe_space.GetNumberOfFiniteElementDofs());
   FillDiscontinuousMuField<order>(fe_space, mu_h);

   auto mu_view =
      MakeReadOnlyElementVectorView<KernelPolicy>(
         fe_space,
         mu_h);

   const Real sigma = 1.0;
   const Real kappa = static_cast<Real>((order + 1) * (order + 1));

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   Cells<"mesh"> cells;
   InteriorFacets<"mesh"> interior_facets;

   auto mu =
      MakeCoefficient<"diffusivity", FieldValue<"mu">>(
         [] GENDIL_HOST_DEVICE (const Real mu_q) -> Real
         {
            return mu_q;
         });

   auto tau =
      MakeCoefficient<"penalty", InverseFacetSize>(
         [=] GENDIL_HOST_DEVICE (const Real h_inv) -> Real
         {
            return kappa * h_inv;
         });

   auto average_mu_form =
      integrate(interior_facets, average(mu) * jump(u) * jump(v));
   auto facet_form =
      integrate(
         interior_facets,
         - average(mu * dot(grad(u), Normal{})) * jump(v)
         + sigma * jump(u) * average(mu * dot(grad(v), Normal{}))
         + tau * average(mu) * jump(u) * jump(v));
   auto diffusion_form =
      integrate(cells, mu * dot(grad(u), grad(v))) + facet_form;

   static_assert(!requires_plus_side_jacobian_v<decltype(average_mu_form)>);
   static_assert(requires_plus_side_jacobian_v<decltype(facet_form)>);
   static_assert(!has_unqualified_side_dependent_inputs_v<decltype(facet_form)>);

   auto wf_ctx =
      MakeWeakFormContext(
         MakeTrialField<"u">(fe_space),
         MakeFiniteElementField<"mu">(fe_space, mu_view),
         MakeDomain<"mesh">(mesh));

   auto op =
      MakeGenericOperator<KernelPolicy>(
         diffusion_form,
         wf_ctx,
         integration_rule);

   const Vector y_generic = ApplyOperator(op, u_h);
   const Vector y_ref =
      ApplySIPDGOracle<order>(
         fe_space,
         u_h,
         sigma,
         kappa,
         true);
   const Vector y_minus_only =
      ApplySIPDGOracle<order>(
         fe_space,
         u_h,
         sigma,
         kappa,
         false);

   const Real rel_err = RelativeL2Error(y_generic, y_ref);
   const Real minus_only_separation = RelativeL2Error(y_minus_only, y_ref);

   std::cout << "  Generic vs averaged-mu SIPDG oracle error = "
             << rel_err << "\n";
   std::cout << "  Minus-only mu diagnostic separation = "
             << minus_only_separation << "\n";

   const Real tol = 1e-11;
   const Real separation_tol = 1e-3;
   if (rel_err > tol)
   {
      std::cerr
         << "FAILED: generic SIPDG coefficient field-input form does not "
         << "match the hand oracle.\n";
      return 1;
   }
   if (minus_only_separation < separation_tol)
   {
      std::cerr
         << "FAILED: manufactured data does not distinguish averaged mu "
         << "from minus-only mu.\n";
      return 1;
   }

   return 0;
}

} // namespace

int main()
{
   if (TestSIPDGCoefficientFieldInputs<1>() != 0) { return 1; }
   if (TestSIPDGCoefficientFieldInputs<2>() != 0) { return 1; }

   std::cout
      << "\nAll SIPDG coefficient field-input tests passed.\n";
   return 0;
}


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

CellAffineField CellwiseWParameters(
   const GlobalIndex ix,
   const GlobalIndex iy)
{
   return CellAffineField{
      0.35 + 0.17 * static_cast<Real>(ix)
           - 0.09 * static_cast<Real>(iy)
           + 0.04 * static_cast<Real>(ix + 3 * iy),
      std::array<Real, 2>{
         0.80 + 0.11 * static_cast<Real>(ix + 1)
              - 0.04 * static_cast<Real>(iy),
        -0.55 + 0.07 * static_cast<Real>(iy + 1)
              + 0.03 * static_cast<Real>(ix)}
   };
}

Real ManufacturedW(
   const GlobalIndex ix,
   const GlobalIndex iy,
   const Real x,
   const Real y)
{
   const auto p = CellwiseWParameters(ix, iy);
   return p.offset + p.gradient[0] * x + p.gradient[1] * y;
}

template <Integer order, typename FiniteElementSpace>
void FillDiscontinuousAffineW(
   const FiniteElementSpace& fe_space,
   Vector& field)
{
   using DofPoints = GaussLobattoLegendrePoints<order + 1>;

   auto dofs = MakeWriteOnlyElementTensorView<SerialKernelConfiguration>(
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

            dofs(i, j, element_index) = ManufacturedW(ix, iy, x, y);
         }
      }
   }
}

template <Integer order, typename FiniteElementSpace>
void FillTrialField(
   const FiniteElementSpace& fe_space,
   Vector& field)
{
   using DofPoints = GaussLobattoLegendrePoints<order + 1>;

   auto dofs = MakeWriteOnlyElementTensorView<SerialKernelConfiguration>(
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
               1.2 + 0.23 * x - 0.31 * y + 0.06 * x * y
                   + 0.025 * static_cast<Real>(ix + 2 * iy);
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

template <Integer order, typename FiniteElementSpace, typename Coefficient>
Vector ApplyInteriorJumpOracle(
   const FiniteElementSpace& fe_space,
   const Vector& u,
   const Coefficient& coefficient)
{
   constexpr Integer num_quad_1d = order + 3;
   using QuadPoints = GaussLegendrePoints<num_quad_1d>;

   Vector y(fe_space.GetNumberOfFiniteElementDofs());
   y = 0.0;

   auto u_dofs = MakeReadOnlyElementTensorView<SerialKernelConfiguration>(
      fe_space,
      u);
   auto y_dofs = MakeReadWriteElementTensorView<SerialKernelConfiguration>(
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
               const Real coeff_q =
                  coefficient(ix, iy, neighbor_ix, neighbor_iy, x_q, y_q);
               const Real weighted_value = coeff_q * jump_u * q_weight * ds;

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
Real RelativeL2Error(const VectorType& a, const VectorType& b)
{
   GENDIL_VERIFY(a.Size() == b.Size(), "Vector sizes do not match.");

   Real err_sq = 0.0;
   Real ref_sq = 0.0;
   for (Integer i = 0; i < a.Size(); ++i)
   {
      const Real d = a[i] - b[i];
      err_sq += d * d;
      ref_sq += b[i] * b[i];
   }

   const Real err = std::sqrt(err_sq);
   const Real ref = std::sqrt(ref_sq);
   return ref == 0.0 ? err : err / ref;
}

template <Integer order>
int TestInteriorCoefficientFieldInputsDG()
{
   std::cout << "\n=== Interior coefficient field-input DG test, order = "
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

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();

   Vector u_h(num_dofs);
   FillTrialField<order>(fe_space, u_h);

   Vector w_h(num_dofs);
   FillDiscontinuousAffineW<order>(fe_space, w_h);

   auto w_view = MakeReadOnlyElementTensorView<KernelPolicy>(fe_space, w_h);

   InteriorFacets<"mesh"> interior_facets;
   TrialSpace<"u"> u;
   TestSpace<"u"> v;

   auto value_coeff = MakeCoefficient<"interior_value", FieldValue<"w_dg">>(
      [] (const Real w) { return 0.5 + 0.75 * w; });

   auto gradient_coeff =
      MakeCoefficient<"interior_gradient", FieldGradient<"w_dg">>(
         [] (const auto& grad_w)
         {
            return 1.0 + grad_w[0] - 0.5 * grad_w[1];
         });

   auto wf_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeFiniteElementField<"w_dg">(fe_space, w_view),
      MakeIntegrationDomain<"mesh">(fe_space));

   auto avg_value_form =
      integrate(interior_facets, average(value_coeff) * jump(u) * jump(v));
   auto jump_value_form =
      integrate(interior_facets, jump(value_coeff) * jump(u) * jump(v));
   auto avg_gradient_form =
      integrate(interior_facets, average(gradient_coeff) * jump(u) * jump(v));

   static_assert(!has_unqualified_side_dependent_inputs_v<decltype(avg_value_form)>);
   static_assert(!has_unqualified_side_dependent_inputs_v<decltype(jump_value_form)>);
   static_assert(!has_unqualified_side_dependent_inputs_v<decltype(avg_gradient_form)>);

   auto avg_value_op = MakeGenericOperator<KernelPolicy>(
      avg_value_form,
      wf_ctx,
      integration_rule);
   auto jump_value_op = MakeGenericOperator<KernelPolicy>(
      jump_value_form,
      wf_ctx,
      integration_rule);
   auto avg_gradient_op = MakeGenericOperator<KernelPolicy>(
      avg_gradient_form,
      wf_ctx,
      integration_rule);

   const Vector y_avg_value = ApplyOperator(avg_value_op, u_h);
   const Vector y_jump_value = ApplyOperator(jump_value_op, u_h);
   const Vector y_avg_gradient = ApplyOperator(avg_gradient_op, u_h);

   const Vector y_avg_value_ref =
      ApplyInteriorJumpOracle<order>(
         fe_space,
         u_h,
         [] (
            const GlobalIndex ix,
            const GlobalIndex iy,
            const GlobalIndex neighbor_ix,
            const GlobalIndex neighbor_iy,
            const Real x,
            const Real y)
         {
            const Real k_minus =
               0.5 + 0.75 * ManufacturedW(ix, iy, x, y);
            const Real k_plus =
               0.5 + 0.75 * ManufacturedW(neighbor_ix, neighbor_iy, x, y);
            return 0.5 * (k_minus + k_plus);
         });

   const Vector y_jump_value_ref =
      ApplyInteriorJumpOracle<order>(
         fe_space,
         u_h,
         [] (
            const GlobalIndex ix,
            const GlobalIndex iy,
            const GlobalIndex neighbor_ix,
            const GlobalIndex neighbor_iy,
            const Real x,
            const Real y)
         {
            const Real k_minus =
               0.5 + 0.75 * ManufacturedW(ix, iy, x, y);
            const Real k_plus =
               0.5 + 0.75 * ManufacturedW(neighbor_ix, neighbor_iy, x, y);
            return k_minus - k_plus;
         });

   const Vector y_avg_gradient_ref =
      ApplyInteriorJumpOracle<order>(
         fe_space,
         u_h,
         [] (
            const GlobalIndex ix,
            const GlobalIndex iy,
            const GlobalIndex neighbor_ix,
            const GlobalIndex neighbor_iy,
            const Real,
            const Real)
         {
            const auto grad_minus = CellwiseWParameters(ix, iy).gradient;
            const auto grad_plus =
               CellwiseWParameters(neighbor_ix, neighbor_iy).gradient;
            const Real k_minus =
               1.0 + grad_minus[0] - 0.5 * grad_minus[1];
            const Real k_plus =
               1.0 + grad_plus[0] - 0.5 * grad_plus[1];
            return 0.5 * (k_minus + k_plus);
         });

   const Real avg_value_err = RelativeL2Error(y_avg_value, y_avg_value_ref);
   const Real jump_value_err = RelativeL2Error(y_jump_value, y_jump_value_ref);
   const Real avg_gradient_err =
      RelativeL2Error(y_avg_gradient, y_avg_gradient_ref);

   std::cout << "  average(FieldValue) error   = " << avg_value_err << "\n";
   std::cout << "  jump(FieldValue) error      = " << jump_value_err << "\n";
   std::cout << "  average(FieldGradient) err  = " << avg_gradient_err << "\n";

   const Real tol = 1e-12;
   if (avg_value_err > tol ||
       jump_value_err > tol ||
       avg_gradient_err > tol)
   {
      std::cerr << "FAILED: interior coefficient field-input oracle mismatch.\n";
      return 1;
   }

   return 0;
}

} // namespace

int main()
{
   if (TestInteriorCoefficientFieldInputsDG<1>() != 0) { return 1; }
   if (TestInteriorCoefficientFieldInputsDG<2>() != 0) { return 1; }

   std::cout << "\nAll interior coefficient field-input DG tests passed.\n";
   return 0;
}

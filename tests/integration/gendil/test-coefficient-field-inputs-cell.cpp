// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{

template <Integer order, typename FiniteElementSpace>
void FillAffineScalarField(
   const FiniteElementSpace& fe_space,
   Vector& field,
   const std::array<Real, 3>& gradient,
   const Real offset)
{
   using DofPoints = GaussLobattoLegendrePoints<order + 1>;

   auto dofs = MakeWriteOnlyElementVectorView<SerialKernelConfiguration>(fe_space, field);

   for (GlobalIndex element_index = 0; element_index < fe_space.GetNumberOfCells(); ++element_index)
   {
      const auto cell_index = GetStructuredSubIndices(element_index, fe_space.sizes);

      for (GlobalIndex k = 0; k < order + 1; ++k)
      {
         for (GlobalIndex j = 0; j < order + 1; ++j)
         {
            for (GlobalIndex i = 0; i < order + 1; ++i)
            {
               const Real x =
                  fe_space.mesh_origin[0] + fe_space.h[0] * (cell_index[0] + DofPoints::GetCoord(i));
               const Real y =
                  fe_space.mesh_origin[1] + fe_space.h[1] * (cell_index[1] + DofPoints::GetCoord(j));
               const Real z =
                  fe_space.mesh_origin[2] + fe_space.h[2] * (cell_index[2] + DofPoints::GetCoord(k));

               dofs(i, j, k, element_index) =
                  offset + gradient[0] * x + gradient[1] * y + gradient[2] * z;
            }
         }
      }
   }
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

template <typename OperatorType>
Vector ApplyOperator(const OperatorType& op, const Vector& x)
{
   Vector y(x.Size());
   y = 0.0;
   op(x, y);
   return y;
}

template <Integer order>
int TestCoefficientFieldInputsCell()
{
   std::cout << "\n=== Cell coefficient field-input test, order = " << order << " ===\n";

   const Integer n = 3;
   const Real h = 1.0 / n;
   CartesianMesh<3> mesh({n, n, n}, {h, 2.0 * h, 0.5 * h}, {0.0, 0.0, 0.0});

   FiniteElementOrders<order, order, order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

   using KernelPolicy = SerialKernelConfiguration;

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();

   Vector x(num_dofs);
   FillRandom(x);

   Vector w_h(num_dofs);
   w_h = 2.5;

   Vector z_h(num_dofs);
   z_h = 3.0;

   Vector g_h(num_dofs);
   constexpr std::array<Real, 3> affine_gradient{1.0, 2.0, -0.4};
   FillAffineScalarField<order>(fe_space, g_h, affine_gradient, 0.75);

   auto w_view = MakeReadOnlyElementVectorView<KernelPolicy>(fe_space, w_h);
   auto z_view = MakeReadOnlyElementVectorView<KernelPolicy>(fe_space, z_h);
   auto g_view = MakeReadOnlyElementVectorView<KernelPolicy>(fe_space, g_h);

   Cells<"mesh"> cells;
   TrialSpace<"u"> u;
   TestSpace<"u"> v;

   auto constant_25 = MakeCoefficient<"constant_25">(
      [] () { return 2.5; });

   auto field_value_coeff = MakeCoefficient<"field_value", FieldValue<"w">>(
      [] (const Real w) { return w; });

   auto field_gradient_coeff = MakeCoefficient<"field_gradient", FieldGradient<"g">>(
      [] (const auto& grad_w)
      {
         return 1.0 + grad_w[0] + grad_w[1] + grad_w[2];
      });

   auto value_and_gradient_coeff =
      MakeCoefficient<"field_value_and_gradient", FieldValue<"w">, FieldGradient<"w">>(
         [] (const Real w, const auto& grad_w)
         {
            return w + grad_w[0] + grad_w[1] + grad_w[2];
         });

   auto multiple_fields_coeff =
      MakeCoefficient<"multiple_fields", FieldValue<"w">, FieldValue<"z">>(
         [] (const Real w, const Real z) { return w + z; });

   auto constant_1 = MakeCoefficient<"constant_1">(
      [] () { return 1.0; });

   auto constant_55 = MakeCoefficient<"constant_55">(
      [] () { return 5.5; });

   auto constant_affine_gradient = MakeCoefficient<"constant_affine_gradient">(
      [] () { return 3.6; });

   auto wf_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeFiniteElementField<"w">(fe_space, w_view),
      MakeFiniteElementField<"z">(fe_space, z_view),
      MakeFiniteElementField<"g">(fe_space, g_view),
      MakeDomain<"mesh">(mesh));

   auto ref_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeDomain<"mesh">(mesh));

   auto ref_25_op = MakeGenericOperator<KernelPolicy>(
      integrate(cells, constant_25 * u * v),
      ref_ctx,
      integration_rule);

   auto ref_1_op = MakeGenericOperator<KernelPolicy>(
      integrate(cells, constant_1 * u * v),
      ref_ctx,
      integration_rule);

   auto ref_55_op = MakeGenericOperator<KernelPolicy>(
      integrate(cells, constant_55 * u * v),
      ref_ctx,
      integration_rule);

   auto ref_affine_gradient_op = MakeGenericOperator<KernelPolicy>(
      integrate(cells, constant_affine_gradient * u * v),
      ref_ctx,
      integration_rule);

   auto value_op = MakeGenericOperator<KernelPolicy>(
      integrate(cells, field_value_coeff * u * v),
      wf_ctx,
      integration_rule);

   auto gradient_op = MakeGenericOperator<KernelPolicy>(
      integrate(cells, field_gradient_coeff * u * v),
      wf_ctx,
      integration_rule);

   auto value_and_gradient_op = MakeGenericOperator<KernelPolicy>(
      integrate(cells, value_and_gradient_coeff * u * v),
      wf_ctx,
      integration_rule);

   auto multiple_fields_op = MakeGenericOperator<KernelPolicy>(
      integrate(cells, multiple_fields_coeff * u * v),
      wf_ctx,
      integration_rule);

   const Vector y_ref_25 = ApplyOperator(ref_25_op, x);
   const Vector y_ref_1 = ApplyOperator(ref_1_op, x);
   const Vector y_ref_55 = ApplyOperator(ref_55_op, x);
   const Vector y_ref_affine_gradient = ApplyOperator(ref_affine_gradient_op, x);

   const Vector y_value = ApplyOperator(value_op, x);
   const Vector y_gradient = ApplyOperator(gradient_op, x);
   const Vector y_value_and_gradient = ApplyOperator(value_and_gradient_op, x);
   const Vector y_multiple_fields = ApplyOperator(multiple_fields_op, x);

   const Real tol = 1e-12;

   const Real err_value = RelativeL2Error(y_value, y_ref_25);
   const Real err_gradient = RelativeL2Error(y_gradient, y_ref_affine_gradient);
   const Real err_value_and_gradient = RelativeL2Error(y_value_and_gradient, y_ref_25);
   const Real err_multiple_fields = RelativeL2Error(y_multiple_fields, y_ref_55);

   std::cout << "  FieldValue error              = " << err_value << "\n";
   std::cout << "  FieldGradient affine error    = " << err_gradient << "\n";
   std::cout << "  FieldValue+FieldGradient err  = " << err_value_and_gradient << "\n";
   std::cout << "  Multiple FieldValue error     = " << err_multiple_fields << "\n";

   if (err_value > tol ||
       err_gradient > tol ||
       err_value_and_gradient > tol ||
       err_multiple_fields > tol)
   {
      std::cerr << "FAILED: cell coefficient field-input operator mismatch.\n";
      return 1;
   }

   return 0;
}

} // namespace

int main()
{
   if (TestCoefficientFieldInputsCell<1>() != 0) { return 1; }
   if (TestCoefficientFieldInputsCell<2>() != 0) { return 1; }

   std::cout << "\nAll cell coefficient field-input tests passed.\n";
   return 0;
}

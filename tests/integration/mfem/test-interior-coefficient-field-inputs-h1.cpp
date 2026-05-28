// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{

Real ContinuousW(const Real x, const Real y)
{
   return 0.45 + 0.70 * x - 0.35 * y + 0.20 * x * y;
}

template <Integer order, typename FiniteElementSpace>
void FillContinuousW(
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

            dofs(i, j, element_index) = ContinuousW(x, y);
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
               1.0 + 0.19 * x - 0.27 * y + 0.04 * x * y
                   + 0.02 * static_cast<Real>(ix + iy);
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

template <Integer order, typename DGSpace, typename H1Space, typename Coefficient>
Vector ApplyInteriorJumpOracle(
   const DGSpace& dg_space,
   const H1Space& h1_space,
   const Vector& u,
   const Vector& w,
   const Coefficient& coefficient,
   Real& max_trace_difference)
{
   constexpr Integer num_quad_1d = order + 3;
   using QuadPoints = GaussLegendrePoints<num_quad_1d>;

   Vector y(dg_space.GetNumberOfFiniteElementDofs());
   y = 0.0;
   max_trace_difference = 0.0;

   auto u_dofs = MakeReadOnlyElementTensorView<SerialKernelConfiguration>(
      dg_space,
      u);
   auto y_dofs = MakeReadWriteElementTensorView<SerialKernelConfiguration>(
      dg_space,
      y);
   auto w_dofs = MakeReadOnlyElementTensorView<SerialKernelConfiguration>(
      h1_space,
      w);

   const GlobalIndex nx = dg_space.sizes[0];
   const GlobalIndex ny = dg_space.sizes[1];

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
            const Real ds = dg_space.h[1 - axis];

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
                  dg_space.mesh_origin[0] + dg_space.h[0] * (ix + xi_minus);
               const Real y_q =
                  dg_space.mesh_origin[1] + dg_space.h[1] * (iy + eta_minus);

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

               const Real w_minus =
                  EvaluateScalarDofsAt<order>(
                     w_dofs,
                     element_index,
                     xi_minus,
                     eta_minus);
               const Real w_plus =
                  EvaluateScalarDofsAt<order>(
                     w_dofs,
                     neighbor_index,
                     xi_plus,
                     eta_plus);

               max_trace_difference =
                  std::max(max_trace_difference, std::abs(w_minus - w_plus));

               const Real jump_u = u_minus - u_plus;
               const Real coeff_q = coefficient(x_q, y_q);
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
int TestInteriorCoefficientFieldInputsH1()
{
   std::cout << "\n=== Interior coefficient field-input H1 exception test, order = "
             << order << " ===\n";

   constexpr GlobalIndex nx = 3;
   constexpr GlobalIndex ny = 2;
   constexpr Real hx = 0.50;
   constexpr Real hy = 0.75;

   CartesianMesh<2> mesh({nx, ny}, {hx, hy}, {0.0, 0.0}, false);

   FiniteElementOrders<order, order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto dg_space = MakeFiniteElementSpace(mesh, finite_element);

   auto mfem_mesh = mfem::Mesh::MakeCartesian2D(
      nx,
      ny,
      mfem::Element::Type::QUADRILATERAL,
      false,
      nx * hx,
      ny * hy,
      false);
   mfem::H1_FECollection fec(order, 2);
   mfem::FiniteElementSpace mfem_fes(&mfem_mesh, &fec);
   auto h1_restriction = GetH1Restriction(mfem_fes);
   auto h1_space = MakeFiniteElementSpace(mesh, finite_element, h1_restriction);

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

   using KernelPolicy = SerialKernelConfiguration;

   Vector u_h(dg_space.GetNumberOfFiniteElementDofs());
   FillTrialField<order>(dg_space, u_h);

   Vector w_h(h1_space.GetNumberOfFiniteElementDofs());
   FillContinuousW<order>(h1_space, w_h);

   auto w_view = MakeReadOnlyElementTensorView<KernelPolicy>(h1_space, w_h);

   InteriorFacets<"mesh"> interior_facets;
   TrialSpace<"u"> u;
   TestSpace<"u"> v;

   auto value_coeff = MakeCoefficient<"interior_h1_value", FieldValue<"w_cg">>(
      [] (const Real w) { return 1.10 + 0.35 * w; });

   auto gradient_coeff =
      MakeCoefficient<"interior_h1_gradient", FieldGradient<"w_cg">>(
         [] (const auto& grad_w)
         {
            return 1.0 + grad_w[0] - 0.5 * grad_w[1];
         });

   auto unqualified_form =
      integrate(interior_facets, value_coeff * jump(u) * jump(v));
   auto average_form =
      integrate(interior_facets, average(value_coeff) * jump(u) * jump(v));
   auto unqualified_gradient_form =
      integrate(interior_facets, gradient_coeff * jump(u) * jump(v));

   static_assert(has_unqualified_side_dependent_inputs_v<decltype(unqualified_form)>);
   static_assert(!has_unqualified_side_dependent_inputs_v<decltype(average_form)>);
   static_assert(has_unqualified_side_dependent_inputs_v<decltype(unqualified_gradient_form)>);

   auto wf_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(dg_space),
      MakeFiniteElementField<"w_cg">(h1_space, w_view),
      MakeDomain<"mesh">(mesh));

   auto unqualified_op = MakeGenericOperator<KernelPolicy>(
      unqualified_form,
      wf_ctx,
      integration_rule);
   auto average_op = MakeGenericOperator<KernelPolicy>(
      average_form,
      wf_ctx,
      integration_rule);

   const Vector y_unqualified = ApplyOperator(unqualified_op, u_h);
   const Vector y_average = ApplyOperator(average_op, u_h);

   Real max_trace_difference = 0.0;
   const Vector y_ref =
      ApplyInteriorJumpOracle<order>(
         dg_space,
         h1_space,
         u_h,
         w_h,
         [] (const Real x, const Real y)
         {
            return 1.10 + 0.35 * ContinuousW(x, y);
         },
         max_trace_difference);

   const Real unqualified_err = RelativeL2Error(y_unqualified, y_ref);
   const Real average_err = RelativeL2Error(y_average, y_ref);

   std::cout << "  unqualified H1 FieldValue error = " << unqualified_err << "\n";
   std::cout << "  average H1 FieldValue error     = " << average_err << "\n";
   std::cout << "  max H1 minus/plus trace diff    = "
             << max_trace_difference << "\n";

   const Real tol = 1e-12;
   if (unqualified_err > tol ||
       average_err > tol ||
       max_trace_difference > 50.0 * tol)
   {
      std::cerr << "FAILED: H1 interior FieldValue exception mismatch.\n";
      return 1;
   }

   return 0;
}

} // namespace

int main()
{
   if (TestInteriorCoefficientFieldInputsH1<1>() != 0) { return 1; }
   if (TestInteriorCoefficientFieldInputsH1<2>() != 0) { return 1; }

   std::cout << "\nAll interior coefficient field-input H1 tests passed.\n";
   return 0;
}

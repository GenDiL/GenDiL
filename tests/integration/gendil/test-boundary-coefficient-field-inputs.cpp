// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>

using namespace gendil;

namespace
{

constexpr std::array<Real, 2> w_gradient{1.1, -0.7};
constexpr Real w_offset = 0.35;

Real ManufacturedW(const Real x, const Real y)
{
   return w_offset + w_gradient[0] * x + w_gradient[1] * y;
}

template <Integer order, typename FiniteElementSpace>
void FillAffineScalarField(
   const FiniteElementSpace& fe_space,
   Vector& field)
{
   using DofPoints = GaussLobattoLegendrePoints<order + 1>;

   auto dofs = MakeWriteOnlyElementTensorView<SerialKernelConfiguration>(fe_space, field);

   for (GlobalIndex element_index = 0; element_index < fe_space.GetNumberOfCells(); ++element_index)
   {
      const auto cell_index = GetStructuredSubIndices(element_index, fe_space.sizes);

      for (GlobalIndex j = 0; j < order + 1; ++j)
      {
         for (GlobalIndex i = 0; i < order + 1; ++i)
         {
            const Real x =
               fe_space.mesh_origin[0] + fe_space.h[0] * (cell_index[0] + DofPoints::GetCoord(i));
            const Real y =
               fe_space.mesh_origin[1] + fe_space.h[1] * (cell_index[1] + DofPoints::GetCoord(j));

            dofs(i, j, element_index) = ManufacturedW(x, y);
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

   auto dofs = MakeWriteOnlyElementTensorView<SerialKernelConfiguration>(fe_space, field);

   for (GlobalIndex element_index = 0; element_index < fe_space.GetNumberOfCells(); ++element_index)
   {
      const auto cell_index = GetStructuredSubIndices(element_index, fe_space.sizes);

      for (GlobalIndex j = 0; j < order + 1; ++j)
      {
         for (GlobalIndex i = 0; i < order + 1; ++i)
         {
            const Real x =
               fe_space.mesh_origin[0] + fe_space.h[0] * (cell_index[0] + DofPoints::GetCoord(i));
            const Real y =
               fe_space.mesh_origin[1] + fe_space.h[1] * (cell_index[1] + DofPoints::GetCoord(j));

            dofs(i, j, element_index) =
               1.0 + 0.25 * x - 0.15 * y + 0.08 * x * y + 0.01 * element_index;
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

template <Integer order, typename FiniteElementSpace, typename Coefficient>
void AccumulateBoundaryFaceOracle(
   const FiniteElementSpace& fe_space,
   const Vector& x,
   Vector& y,
   const GlobalIndex face_index,
   const Coefficient& coefficient)
{
   constexpr Integer num_quad_1d = order + 3;
   using QuadPoints = GaussLegendrePoints<num_quad_1d>;

   auto x_dofs = MakeReadOnlyElementTensorView<SerialKernelConfiguration>(fe_space, x);
   auto y_dofs = MakeReadWriteElementTensorView<SerialKernelConfiguration>(fe_space, y);

   const GlobalIndex nx = fe_space.sizes[0];
   const GlobalIndex ny = fe_space.sizes[1];

   const GlobalIndex axis = face_index % 2;
   const bool plus_side = face_index >= 2;
   const Real face_ref_coord = plus_side ? 1.0 : 0.0;
   const Real ds = fe_space.h[1 - axis];

   const GlobalIndex ix_begin = axis == 0 ? (plus_side ? nx - 1 : 0) : 0;
   const GlobalIndex ix_end   = axis == 0 ? ix_begin + 1 : nx;
   const GlobalIndex iy_begin = axis == 1 ? (plus_side ? ny - 1 : 0) : 0;
   const GlobalIndex iy_end   = axis == 1 ? iy_begin + 1 : ny;

   for (GlobalIndex iy = iy_begin; iy < iy_end; ++iy)
   {
      for (GlobalIndex ix = ix_begin; ix < ix_end; ++ix)
      {
         const GlobalIndex element_index = ix + nx * iy;

         for (GlobalIndex q = 0; q < num_quad_1d; ++q)
         {
            const Real tangent_ref_coord = QuadPoints::GetCoord(q);
            const Real q_weight = QuadPoints::GetWeight(q);

            const Real xi  = axis == 0 ? face_ref_coord : tangent_ref_coord;
            const Real eta = axis == 1 ? face_ref_coord : tangent_ref_coord;

            const Real x_q = fe_space.mesh_origin[0] + fe_space.h[0] * (ix + xi);
            const Real y_q = fe_space.mesh_origin[1] + fe_space.h[1] * (iy + eta);

            Real u_q = 0.0;

            for (GlobalIndex j = 0; j < order + 1; ++j)
            {
               for (GlobalIndex i = 0; i < order + 1; ++i)
               {
                  const Real phi =
                     ShapeValue<order>(i, xi) * ShapeValue<order>(j, eta);
                  u_q += x_dofs(i, j, element_index) * phi;
               }
            }

            const Real coeff_q = coefficient(x_q, y_q);
            const Real weighted_value = coeff_q * u_q * q_weight * ds;

            for (GlobalIndex j = 0; j < order + 1; ++j)
            {
               for (GlobalIndex i = 0; i < order + 1; ++i)
               {
                  const Real phi =
                     ShapeValue<order>(i, xi) * ShapeValue<order>(j, eta);
                  y_dofs(i, j, element_index) += weighted_value * phi;
               }
            }
         }
      }
   }
}

template <Integer order, typename FiniteElementSpace, typename Coefficient>
Vector ApplyBoundaryMassOracle(
   const FiniteElementSpace& fe_space,
   const Vector& x,
   const Coefficient& coefficient)
{
   Vector y(fe_space.GetNumberOfFiniteElementDofs());
   y = 0.0;

   for (GlobalIndex face_index = 0; face_index < 4; ++face_index)
   {
      AccumulateBoundaryFaceOracle<order>(fe_space, x, y, face_index, coefficient);
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

template <class List, StaticString Name>
struct CountNamedFieldRequirements;

template <StaticString Name>
struct CountNamedFieldRequirements<type_list<>, Name> : std::integral_constant<Integer, 0> {};

template <class Req, class... Rest, StaticString Name>
struct CountNamedFieldRequirements<type_list<Req, Rest...>, Name>
   : std::integral_constant<
        Integer,
        (Req::name == Name ? 1 : 0) +
           CountNamedFieldRequirements<type_list<Rest...>, Name>::value> {};

template <class List, StaticString Name>
inline constexpr Integer count_named_field_requirements_v =
   CountNamedFieldRequirements<List, Name>::value;

template <Integer order>
int TestBoundaryCoefficientFieldInputs()
{
   std::cout << "\n=== Boundary coefficient field-input test, order = " << order << " ===\n";

   constexpr GlobalIndex nx = 3;
   constexpr GlobalIndex ny = 2;

   CartesianMesh<2> mesh({nx, ny}, {0.4, 0.7}, {0.2, -0.3}, false);

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
   FillAffineScalarField<order>(fe_space, w_h);

   auto w_view = MakeReadOnlyElementTensorView<KernelPolicy>(fe_space, w_h);

   BoundaryFacets<"mesh"> boundary_facets;
   TrialSpace<"u"> u;
   TestSpace<"u"> v;

   auto value_coeff = MakeCoefficient<"boundary_value", FieldValue<"w">>(
      [] (const Real w) { return 0.5 + w; });

   auto gradient_coeff = MakeCoefficient<"boundary_gradient", FieldGradient<"w">>(
      [] (const auto& grad_w)
      {
         return 1.0 + grad_w[0] - 0.5 * grad_w[1];
      });

   auto value_and_gradient_coeff =
      MakeCoefficient<"boundary_value_and_gradient", FieldValue<"w">, FieldGradient<"w">>(
         [] (const Real w, const auto& grad_w)
         {
            return 0.25 + 0.75 * w + grad_w[0] - 0.5 * grad_w[1];
         });

   auto wf_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeFiniteElementField<"w">(fe_space, w_view),
      MakeIntegrationDomain<"mesh">(fe_space));

   auto value_form = integrate(boundary_facets, value_coeff * u * v);
   auto gradient_form = integrate(boundary_facets, gradient_coeff * u * v);
   auto value_and_gradient_form =
      integrate(boundary_facets, value_and_gradient_coeff * u * v);

   using ValueAndGradientReqs =
      interpolation_named_field_requirements_t<decltype(value_and_gradient_form)>;
   static_assert(contains_named_field_requirement_v<ValueAndGradientReqs, "w">);
   using WReq = find_named_field_requirement_t<ValueAndGradientReqs, "w">;
   static_assert(need_values(WReq::mask));
   static_assert(need_gradients(WReq::mask));
   static_assert(has_provenance(WReq::provenance, NamedFieldProvenance::CoefficientInput));
   static_assert(count_named_field_requirements_v<ValueAndGradientReqs, "w"> == 1);

   auto value_op = MakeGenericOperator<KernelPolicy>(
      value_form,
      wf_ctx,
      integration_rule);

   auto gradient_op = MakeGenericOperator<KernelPolicy>(
      gradient_form,
      wf_ctx,
      integration_rule);

   auto value_and_gradient_op = MakeGenericOperator<KernelPolicy>(
      value_and_gradient_form,
      wf_ctx,
      integration_rule);

   const Vector y_value = ApplyOperator(value_op, u_h);
   const Vector y_gradient = ApplyOperator(gradient_op, u_h);
   const Vector y_value_and_gradient = ApplyOperator(value_and_gradient_op, u_h);

   const Vector y_value_ref =
      ApplyBoundaryMassOracle<order>(
         fe_space,
         u_h,
         [] (const Real x, const Real y)
         {
            return 0.5 + ManufacturedW(x, y);
         });

   const Vector y_gradient_ref =
      ApplyBoundaryMassOracle<order>(
         fe_space,
         u_h,
         [] (const Real, const Real)
         {
            return 1.0 + w_gradient[0] - 0.5 * w_gradient[1];
         });

   const Vector y_value_and_gradient_ref =
      ApplyBoundaryMassOracle<order>(
         fe_space,
         u_h,
         [] (const Real x, const Real y)
         {
            return 0.25 + 0.75 * ManufacturedW(x, y)
                         + w_gradient[0] - 0.5 * w_gradient[1];
         });

   const Real value_err = RelativeL2Error(y_value, y_value_ref);
   const Real gradient_err = RelativeL2Error(y_gradient, y_gradient_ref);
   const Real value_and_gradient_err =
      RelativeL2Error(y_value_and_gradient, y_value_and_gradient_ref);

   std::cout << "  Boundary FieldValue error              = " << value_err << "\n";
   std::cout << "  Boundary FieldGradient error           = " << gradient_err << "\n";
   std::cout << "  Boundary FieldValue+FieldGradient err  = "
             << value_and_gradient_err << "\n";

   const Real tol = 1e-12;
   if (value_err > tol ||
       gradient_err > tol ||
       value_and_gradient_err > tol)
   {
      std::cerr << "FAILED: boundary coefficient field-input oracle mismatch.\n";
      return 1;
   }

   return 0;
}

} // namespace

int main()
{
   if (TestBoundaryCoefficientFieldInputs<1>() != 0) { return 1; }
   if (TestBoundaryCoefficientFieldInputs<2>() != 0) { return 1; }

   std::cout << "\nAll boundary coefficient field-input tests passed.\n";
   return 0;
}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <tuple>
#include <type_traits>

using namespace gendil;

namespace
{

constexpr std::array<Real, 2> w0_gradient{1.5, -0.75};
constexpr std::array<Real, 2> w1_gradient{0.25, 2.0};
constexpr Real w0_offset = 0.25;
constexpr Real w1_offset = -0.5;

std::array<Real, 2> ManufacturedVectorW(const Real x, const Real y)
{
   return {
      w0_offset + w0_gradient[0] * x + w0_gradient[1] * y,
      w1_offset + w1_gradient[0] * x + w1_gradient[1] * y
   };
}

Real ValueCoefficient(const Real x, const Real y)
{
   const auto w = ManufacturedVectorW(x, y);
   return 1.0 + 2.0 * w[0] - 0.5 * w[1];
}

Real GradientCoefficient()
{
   return 1.0
        + 0.7 * w0_gradient[0]
        - 1.1 * w0_gradient[1]
        + 1.3 * w1_gradient[0]
        - 0.9 * w1_gradient[1];
}

Real ValueAndGradientCoefficient(const Real x, const Real y)
{
   const auto w = ManufacturedVectorW(x, y);
   return 0.35
        + 0.6 * w[0]
        - 0.4 * w[1]
        + 0.7 * w0_gradient[0]
        - 1.1 * w0_gradient[1]
        + 1.3 * w1_gradient[0]
        - 0.9 * w1_gradient[1];
}

template <Integer order, typename FiniteElementSpace>
void FillAffineVectorField(
   const FiniteElementSpace& fe_space,
   Vector& field)
{
   using DofPoints = GaussLobattoLegendrePoints<order + 1>;

   auto dofs =
      MakeWriteOnlyElementTensorView<SerialKernelConfiguration>(
         fe_space,
         field);

   for (GlobalIndex element_index = 0;
        element_index < fe_space.GetNumberOfCells();
        ++element_index)
   {
      const auto cell_index =
         GetStructuredSubIndices(element_index, fe_space.sizes);

      for (GlobalIndex j = 0; j < order + 1; ++j)
      {
         for (GlobalIndex i = 0; i < order + 1; ++i)
         {
            const Real x =
               fe_space.mesh_origin[0] +
               fe_space.h[0] * (cell_index[0] + DofPoints::GetCoord(i));
            const Real y =
               fe_space.mesh_origin[1] +
               fe_space.h[1] * (cell_index[1] + DofPoints::GetCoord(j));

            const auto w = ManufacturedVectorW(x, y);
            std::get<0>(dofs)(i, j, element_index) = w[0];
            std::get<1>(dofs)(i, j, element_index) = w[1];
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

   auto dofs =
      MakeWriteOnlyElementTensorView<SerialKernelConfiguration>(
         fe_space,
         field);

   for (GlobalIndex element_index = 0;
        element_index < fe_space.GetNumberOfCells();
        ++element_index)
   {
      const auto cell_index =
         GetStructuredSubIndices(element_index, fe_space.sizes);

      for (GlobalIndex j = 0; j < order + 1; ++j)
      {
         for (GlobalIndex i = 0; i < order + 1; ++i)
         {
            const Real x =
               fe_space.mesh_origin[0] +
               fe_space.h[0] * (cell_index[0] + DofPoints::GetCoord(i));
            const Real y =
               fe_space.mesh_origin[1] +
               fe_space.h[1] * (cell_index[1] + DofPoints::GetCoord(j));

            dofs(i, j, element_index) =
               1.0 + 0.20 * x - 0.35 * y + 0.07 * x * y
                   + 0.015 * static_cast<Real>(element_index);
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
Vector ApplyCellMassOracle(
   const FiniteElementSpace& fe_space,
   const Vector& x,
   const Coefficient& coefficient)
{
   constexpr Integer num_quad_1d = order + 3;
   using QuadPoints = GaussLegendrePoints<num_quad_1d>;

   Vector y(fe_space.GetNumberOfFiniteElementDofs());
   y = 0.0;

   auto x_dofs =
      MakeReadOnlyElementTensorView<SerialKernelConfiguration>(
         fe_space,
         x);
   auto y_dofs =
      MakeReadWriteElementTensorView<SerialKernelConfiguration>(
         fe_space,
         y);

   const Real dx = fe_space.h[0];
   const Real dy = fe_space.h[1];
   const Real det_J = dx * dy;

   for (GlobalIndex element_index = 0;
        element_index < fe_space.GetNumberOfCells();
        ++element_index)
   {
      const auto cell_index =
         GetStructuredSubIndices(element_index, fe_space.sizes);

      for (GlobalIndex qy = 0; qy < num_quad_1d; ++qy)
      {
         for (GlobalIndex qx = 0; qx < num_quad_1d; ++qx)
         {
            const Real xi = QuadPoints::GetCoord(qx);
            const Real eta = QuadPoints::GetCoord(qy);
            const Real q_weight =
               QuadPoints::GetWeight(qx) * QuadPoints::GetWeight(qy);

            const Real x_q =
               fe_space.mesh_origin[0] + dx * (cell_index[0] + xi);
            const Real y_q =
               fe_space.mesh_origin[1] + dy * (cell_index[1] + eta);

            Real u_q = 0.0;
            for (GlobalIndex j = 0; j < order + 1; ++j)
            {
               for (GlobalIndex i = 0; i < order + 1; ++i)
               {
                  const Real phi =
                     ShapeValue<order>(i, xi) *
                     ShapeValue<order>(j, eta);
                  u_q += x_dofs(i, j, element_index) * phi;
               }
            }

            const Real weighted_value =
               coefficient(x_q, y_q) * u_q * q_weight * det_J;

            for (GlobalIndex j = 0; j < order + 1; ++j)
            {
               for (GlobalIndex i = 0; i < order + 1; ++i)
               {
                  const Real phi =
                     ShapeValue<order>(i, xi) *
                     ShapeValue<order>(j, eta);
                  y_dofs(i, j, element_index) += weighted_value * phi;
               }
            }
         }
      }
   }

   return y;
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

   auto x_dofs =
      MakeReadOnlyElementTensorView<SerialKernelConfiguration>(
         fe_space,
         x);
   auto y_dofs =
      MakeReadWriteElementTensorView<SerialKernelConfiguration>(
         fe_space,
         y);

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

            const Real xi =
               axis == 0 ? face_ref_coord : tangent_ref_coord;
            const Real eta =
               axis == 1 ? face_ref_coord : tangent_ref_coord;

            const Real x_q =
               fe_space.mesh_origin[0] + fe_space.h[0] * (ix + xi);
            const Real y_q =
               fe_space.mesh_origin[1] + fe_space.h[1] * (iy + eta);

            Real u_q = 0.0;
            for (GlobalIndex j = 0; j < order + 1; ++j)
            {
               for (GlobalIndex i = 0; i < order + 1; ++i)
               {
                  const Real phi =
                     ShapeValue<order>(i, xi) *
                     ShapeValue<order>(j, eta);
                  u_q += x_dofs(i, j, element_index) * phi;
               }
            }

            const Real weighted_value =
               coefficient(x_q, y_q) * u_q * q_weight * ds;

            for (GlobalIndex j = 0; j < order + 1; ++j)
            {
               for (GlobalIndex i = 0; i < order + 1; ++i)
               {
                  const Real phi =
                     ShapeValue<order>(i, xi) *
                     ShapeValue<order>(j, eta);
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
      AccumulateBoundaryFaceOracle<order>(
         fe_space,
         x,
         y,
         face_index,
         coefficient);
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
struct CountNamedFieldRequirements<type_list<>, Name>
   : std::integral_constant<Integer, 0> {};

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
int TestVectorCoefficientFieldInputs()
{
   std::cout << "\n=== Vector coefficient field-input test, order = "
             << order << " ===\n";

   constexpr GlobalIndex nx = 3;
   constexpr GlobalIndex ny = 2;

   CartesianMesh<2> mesh({nx, ny}, {0.40, 0.70}, {0.20, -0.30}, false);

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
   FillTrialField<order>(scalar_fe_space, u_h);

   Vector w_vec_h(vector_fe_space.GetNumberOfFiniteElementDofs());
   FillAffineVectorField<order>(vector_fe_space, w_vec_h);

   auto w_vec_view =
      MakeReadOnlyElementTensorView<KernelPolicy>(
         vector_fe_space,
         w_vec_h);

   Cells<"mesh"> cells;
   BoundaryFacets<"mesh"> boundary_facets;
   TrialSpace<"u"> u;
   TestSpace<"u"> v;

   auto value_coeff =
      MakeCoefficient<"vector_value", FieldValue<"w_vec">>(
         [] (const auto& w)
         {
            return 1.0 + 2.0 * w(0) - 0.5 * w(1);
         });

   auto gradient_coeff =
      MakeCoefficient<"vector_gradient", FieldGradient<"w_vec">>(
         [] (const auto& grad_w)
         {
            return 1.0
                 + 0.7 * grad_w(0, 0)
                 - 1.1 * grad_w(0, 1)
                 + 1.3 * grad_w(1, 0)
                 - 0.9 * grad_w(1, 1);
         });

   auto value_and_gradient_coeff =
      MakeCoefficient<
         "vector_value_and_gradient",
         FieldValue<"w_vec">,
         FieldGradient<"w_vec">>(
         [] (const auto& w, const auto& grad_w)
         {
            return 0.35
                 + 0.6 * w(0)
                 - 0.4 * w(1)
                 + 0.7 * grad_w(0, 0)
                 - 1.1 * grad_w(0, 1)
                 + 1.3 * grad_w(1, 0)
                 - 0.9 * grad_w(1, 1);
         });

   auto wf_ctx =
      MakeWeakFormContext(
         MakeTrialField<"u">(scalar_fe_space),
         MakeFiniteElementField<"w_vec">(vector_fe_space, w_vec_view),
         MakeDomain<"mesh">(mesh));

   auto cell_value_form =
      integrate(cells, value_coeff * u * v);
   auto cell_gradient_form =
      integrate(cells, gradient_coeff * u * v);
   auto cell_value_and_gradient_form =
      integrate(cells, value_and_gradient_coeff * u * v);
   auto boundary_value_form =
      integrate(boundary_facets, value_coeff * u * v);
   auto boundary_gradient_form =
      integrate(boundary_facets, gradient_coeff * u * v);

   using ValueAndGradientReqs =
      interpolation_named_field_requirements_t<
         decltype(cell_value_and_gradient_form)>;
   static_assert(
      contains_named_field_requirement_v<ValueAndGradientReqs, "w_vec">);
   using WVecReq =
      find_named_field_requirement_t<ValueAndGradientReqs, "w_vec">;
   static_assert(need_values(WVecReq::mask));
   static_assert(need_gradients(WVecReq::mask));
   static_assert(
      has_provenance(
         WVecReq::provenance,
         NamedFieldProvenance::CoefficientInput));
   static_assert(
      count_named_field_requirements_v<ValueAndGradientReqs, "w_vec"> == 1);

   auto cell_value_op =
      MakeGenericOperator<KernelPolicy>(
         cell_value_form,
         wf_ctx,
         integration_rule);
   auto cell_gradient_op =
      MakeGenericOperator<KernelPolicy>(
         cell_gradient_form,
         wf_ctx,
         integration_rule);
   auto cell_value_and_gradient_op =
      MakeGenericOperator<KernelPolicy>(
         cell_value_and_gradient_form,
         wf_ctx,
         integration_rule);
   auto boundary_value_op =
      MakeGenericOperator<KernelPolicy>(
         boundary_value_form,
         wf_ctx,
         integration_rule);
   auto boundary_gradient_op =
      MakeGenericOperator<KernelPolicy>(
         boundary_gradient_form,
         wf_ctx,
         integration_rule);

   const Vector y_cell_value =
      ApplyOperator(cell_value_op, u_h);
   const Vector y_cell_gradient =
      ApplyOperator(cell_gradient_op, u_h);
   const Vector y_cell_value_and_gradient =
      ApplyOperator(cell_value_and_gradient_op, u_h);
   const Vector y_boundary_value =
      ApplyOperator(boundary_value_op, u_h);
   const Vector y_boundary_gradient =
      ApplyOperator(boundary_gradient_op, u_h);

   const Vector y_cell_value_ref =
      ApplyCellMassOracle<order>(
         scalar_fe_space,
         u_h,
         [] (const Real x, const Real y)
         {
            return ValueCoefficient(x, y);
         });

   const Vector y_cell_gradient_ref =
      ApplyCellMassOracle<order>(
         scalar_fe_space,
         u_h,
         [] (const Real, const Real)
         {
            return GradientCoefficient();
         });

   const Vector y_cell_value_and_gradient_ref =
      ApplyCellMassOracle<order>(
         scalar_fe_space,
         u_h,
         [] (const Real x, const Real y)
         {
            return ValueAndGradientCoefficient(x, y);
         });

   const Vector y_boundary_value_ref =
      ApplyBoundaryMassOracle<order>(
         scalar_fe_space,
         u_h,
         [] (const Real x, const Real y)
         {
            return ValueCoefficient(x, y);
         });

   const Vector y_boundary_gradient_ref =
      ApplyBoundaryMassOracle<order>(
         scalar_fe_space,
         u_h,
         [] (const Real, const Real)
         {
            return GradientCoefficient();
         });

   const Real cell_value_err =
      RelativeL2Error(y_cell_value, y_cell_value_ref);
   const Real cell_gradient_err =
      RelativeL2Error(y_cell_gradient, y_cell_gradient_ref);
   const Real cell_value_and_gradient_err =
      RelativeL2Error(
         y_cell_value_and_gradient,
         y_cell_value_and_gradient_ref);
   const Real boundary_value_err =
      RelativeL2Error(y_boundary_value, y_boundary_value_ref);
   const Real boundary_gradient_err =
      RelativeL2Error(y_boundary_gradient, y_boundary_gradient_ref);

   std::cout << "  Cell vector FieldValue error              = "
             << cell_value_err << "\n";
   std::cout << "  Cell vector FieldGradient error           = "
             << cell_gradient_err << "\n";
   std::cout << "  Cell vector FieldValue+FieldGradient err  = "
             << cell_value_and_gradient_err << "\n";
   std::cout << "  Boundary vector FieldValue error          = "
             << boundary_value_err << "\n";
   std::cout << "  Boundary vector FieldGradient error       = "
             << boundary_gradient_err << "\n";

   const Real tol = 1e-12;
   if (cell_value_err > tol ||
       cell_gradient_err > tol ||
       cell_value_and_gradient_err > tol ||
       boundary_value_err > tol ||
       boundary_gradient_err > tol)
   {
      std::cerr
         << "FAILED: vector coefficient field-input oracle mismatch.\n";
      return 1;
   }

   return 0;
}

} // namespace

int main()
{
   if (TestVectorCoefficientFieldInputs<1>() != 0) { return 1; }
   if (TestVectorCoefficientFieldInputs<2>() != 0) { return 1; }

   std::cout
      << "\nAll vector coefficient field-input tests passed.\n";
   return 0;
}

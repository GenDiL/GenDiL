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

using Matrix2 = std::array<std::array<Real, 2>, 2>;
using Vector2 = std::array<Real, 2>;

Matrix2 ConstantMatrix()
{
   return Matrix2{{
      {{Real{2.0}, -Real{0.35}}},
      {{Real{0.65}, Real{1.4}}}
   }};
}

Matrix2 NamedFieldMatrix(const Real a)
{
   return Matrix2{{
      {{Real{1.20} + Real{0.40} * a,
        -Real{0.25} + Real{0.10} * a}},
      {{Real{0.35} - Real{0.05} * a,
        Real{1.80} + Real{0.20} * a}}
   }};
}

Vector2 MatVec(const Matrix2& A, const Vector2& x)
{
   return Vector2{
      A[0][0] * x[0] + A[0][1] * x[1],
      A[1][0] * x[0] + A[1][1] * x[1]};
}

Vector2 MatTransposeVec(const Matrix2& A, const Vector2& x)
{
   return Vector2{
      A[0][0] * x[0] + A[1][0] * x[1],
      A[0][1] * x[0] + A[1][1] * x[1]};
}

Real Dot2(const Vector2& a, const Vector2& b)
{
   return a[0] * b[0] + a[1] * b[1];
}

Real ManufacturedU(
   const GlobalIndex ix,
   const GlobalIndex iy,
   const Real x,
   const Real y)
{
   return Real{0.73}
        + Real{0.31} * x
        - Real{0.27} * y
        + Real{0.11} * x * y
        + Real{0.04} * x * x
        - Real{0.06} * y * y
        + Real{0.17} * static_cast<Real>(ix)
        - Real{0.09} * static_cast<Real>(iy);
}

Real ManufacturedAField(
   const GlobalIndex ix,
   const GlobalIndex iy,
   const Real x,
   const Real y)
{
   return Real{0.60}
        + Real{0.24} * x
        - Real{0.18} * y
        + Real{0.07} * static_cast<Real>(ix + 2 * iy);
}

template <Integer order, typename FiniteElementSpace>
void FillTrialField(const FiniteElementSpace& fe_space, Vector& field)
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

template <Integer order, typename FiniteElementSpace>
void FillScalarCoefficientField(const FiniteElementSpace& fe_space, Vector& field)
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

            dofs(i, j, element_index) =
               ManufacturedAField(ix, iy, x, y);
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
Vector2 EvaluateScalarGradientAt(
   const ElementDofs& dofs,
   const GlobalIndex element_index,
   const Real xi,
   const Real eta,
   const Real hx,
   const Real hy)
{
   Vector2 gradient{0.0, 0.0};

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

enum class MatrixPlacement
{
   TrialGradient,
   TestGradient
};

template <Integer order, typename FiniteElementSpace, typename MatrixAt>
Vector ApplyAnisotropicCellOracle(
   const FiniteElementSpace& fe_space,
   const Vector& u,
   const MatrixAt& matrix_at,
   const MatrixPlacement placement)
{
   constexpr Integer num_quad_1d = order + 3;
   using QuadPoints = GaussLegendrePoints<num_quad_1d>;

   Vector y(fe_space.GetNumberOfFiniteElementDofs());
   y = 0.0;

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

            const Real x =
               fe_space.mesh_origin[0] + hx * (ix + xi);
            const Real y_phys =
               fe_space.mesh_origin[1] + hy * (iy + eta);

            const Vector2 grad_u =
               EvaluateScalarGradientAt<order>(
                  u_dofs,
                  element_index,
                  xi,
                  eta,
                  hx,
                  hy);

            const Matrix2 A = matrix_at(ix, iy, x, y_phys);
            const Vector2 trial_seed =
               placement == MatrixPlacement::TrialGradient
                  ? MatVec(A, grad_u)
                  : MatTransposeVec(A, grad_u);

            const Real measure = q_weight * det_J;

            for (GlobalIndex j = 0; j < order + 1; ++j)
            {
               for (GlobalIndex i = 0; i < order + 1; ++i)
               {
                  const Vector2 grad_phi{
                     ShapeGradient<order>(i, xi) *
                        ShapeValue<order>(j, eta) / hx,
                     ShapeValue<order>(i, xi) *
                        ShapeGradient<order>(j, eta) / hy};

                  y_dofs(i, j, element_index) +=
                     Dot2(trial_seed, grad_phi) * measure;
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
int TestAnisotropicDiffusionCoefficient()
{
   std::cout << "\n=== Anisotropic matrix coefficient test, order = "
             << order << " ===\n";

   constexpr GlobalIndex nx = 3;
   constexpr GlobalIndex ny = 2;

   CartesianMesh<2> mesh({nx, ny}, {0.43, 0.71}, {0.20, -0.25}, false);

   FiniteElementOrders<order, order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

   using KernelPolicy = SerialKernelConfiguration;

   Vector u_h(fe_space.GetNumberOfFiniteElementDofs());
   FillTrialField<order>(fe_space, u_h);

   Vector a_h(fe_space.GetNumberOfFiniteElementDofs());
   FillScalarCoefficientField<order>(fe_space, a_h);
   auto a_view =
      MakeReadOnlyElementVectorView<KernelPolicy>(
         fe_space,
         a_h);

   Cells<"mesh"> cells;
   TrialSpace<"u"> u;
   TestSpace<"u"> v;

   auto A_const =
      MakeMatrixCoefficient<"A_const">(
         [] ()
         {
            return ConstantMatrix();
         });

   auto A_named =
      MakeMatrixCoefficient<"A_named", FieldValue<"a">>(
         [] (const Real a)
         {
            return NamedFieldMatrix(a);
         });

   auto primal_form =
      integrate(cells, dot(A_const * grad(u), grad(v)));
   auto transpose_pullback_form =
      integrate(cells, dot(A_const * grad(v), grad(u)));
   auto named_field_form =
      integrate(cells, dot(A_named * grad(u), grad(v)));

   static_assert(is_test_linear_v<decltype(A_const * grad(v))>);
   static_assert(is_test_linear_v<decltype(dot(A_const * grad(v), grad(u)))>);

   using NamedReqs =
      interpolation_named_field_requirements_t<decltype(named_field_form)>;
   static_assert(contains_named_field_requirement_v<NamedReqs, "a">);
   using AReq = find_named_field_requirement_t<NamedReqs, "a">;
   static_assert(need_values(AReq::mask));
   static_assert(
      has_provenance(
         AReq::provenance,
         NamedFieldProvenance::CoefficientInput));

   auto base_ctx =
      MakeWeakFormContext(
         MakeTrialField<"u">(fe_space),
         MakeDomain<"mesh">(mesh));

   auto named_ctx =
      MakeWeakFormContext(
         MakeTrialField<"u">(fe_space),
         MakeFiniteElementField<"a">(fe_space, a_view),
         MakeDomain<"mesh">(mesh));

   auto primal_op =
      MakeGenericOperator<KernelPolicy>(
         primal_form,
         base_ctx,
         integration_rule);
   auto transpose_pullback_op =
      MakeGenericOperator<KernelPolicy>(
         transpose_pullback_form,
         base_ctx,
         integration_rule);
   auto named_field_op =
      MakeGenericOperator<KernelPolicy>(
         named_field_form,
         named_ctx,
         integration_rule);

   const Vector y_primal = ApplyOperator(primal_op, u_h);
   const Vector y_transpose = ApplyOperator(transpose_pullback_op, u_h);
   const Vector y_named = ApplyOperator(named_field_op, u_h);

   const Vector y_primal_ref =
      ApplyAnisotropicCellOracle<order>(
         fe_space,
         u_h,
         [] (const GlobalIndex, const GlobalIndex, const Real, const Real)
         {
            return ConstantMatrix();
         },
         MatrixPlacement::TrialGradient);

   const Vector y_transpose_ref =
      ApplyAnisotropicCellOracle<order>(
         fe_space,
         u_h,
         [] (const GlobalIndex, const GlobalIndex, const Real, const Real)
         {
            return ConstantMatrix();
         },
         MatrixPlacement::TestGradient);

   const Vector y_named_ref =
      ApplyAnisotropicCellOracle<order>(
         fe_space,
         u_h,
         [] (const GlobalIndex ix,
             const GlobalIndex iy,
             const Real x,
             const Real y)
         {
            return NamedFieldMatrix(ManufacturedAField(ix, iy, x, y));
         },
         MatrixPlacement::TrialGradient);

   const Real primal_err = RelativeL2Error(y_primal, y_primal_ref);
   const Real transpose_err =
      RelativeL2Error(y_transpose, y_transpose_ref);
   const Real named_err = RelativeL2Error(y_named, y_named_ref);

   const Real wrong_transpose_sep =
      RelativeL2Error(y_transpose_ref, y_primal_ref);

   std::cout << "  Constant A normal matvec error       = "
             << primal_err << "\n";
   std::cout << "  Constant A transpose pullback error  = "
             << transpose_err << "\n";
   std::cout << "  Named-field matrix coefficient error = "
             << named_err << "\n";
   std::cout << "  A^T seed vs wrong A seed separation  = "
             << wrong_transpose_sep << "\n";

   const Real tol = 1e-11;
   if (primal_err > tol ||
       transpose_err > tol ||
       named_err > tol ||
       wrong_transpose_sep < Real{1e-4})
   {
      std::cerr
         << "FAILED: anisotropic matrix coefficient oracle mismatch.\n";
      return 1;
   }

   return 0;
}

} // namespace

int main()
{
   if (TestAnisotropicDiffusionCoefficient<1>() != 0) { return 1; }
   if (TestAnisotropicDiffusionCoefficient<2>() != 0) { return 1; }

   std::cout
      << "\nAll anisotropic matrix coefficient tests passed.\n";
   return 0;
}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace gendil;

namespace
{

#if defined(GENDIL_USE_DEVICE)
template <Integer NumQuad1D>
using GlobalFaceKernelPolicy =
   DeviceKernelConfiguration<ThreadBlockLayout<NumQuad1D>, 1, 2>;
#else
template <Integer>
using GlobalFaceKernelPolicy = SerialKernelConfiguration;
#endif

constexpr GlobalIndex nxL = 2;
constexpr GlobalIndex nyL = 2;
constexpr GlobalIndex nxR = 2;
constexpr GlobalIndex nyR = 4;
constexpr Real hx = Real{1.0} / static_cast<Real>(nxL);
constexpr Real hyL = Real{1.0} / static_cast<Real>(nyL);
constexpr Real hyR = Real{1.0} / static_cast<Real>(nyR);
constexpr Real interface_measure = static_cast<Real>(nyL) * hyL;

struct ConstantState
{
   static Real Minus(const Real, const Real)
   {
      return 1.0;
   }

   static Real Plus(const Real, const Real)
   {
      return 2.0;
   }
};

struct LinearState
{
   static Real Minus(const Real x, const Real y)
   {
      return x + 2.0 * y;
   }

   static Real Plus(const Real x, const Real y)
   {
      return 2.0 - x + y;
   }
};

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
         value +=
            dofs(i, j, element_index)
          * ShapeValue<order>(i, xi)
          * ShapeValue<order>(j, eta);
      }
   }
   return value;
}

template <Integer order, typename FiniteElementSpace, typename State>
void FillMinusDofs(
   const FiniteElementSpace& fe_space,
   Vector& x)
{
   using DofPoints = GaussLobattoLegendrePoints<order + 1>;
   auto dofs =
      MakeWriteOnlyElementTensorView<SerialKernelConfiguration>(
         fe_space,
         x);

   const GlobalIndex nx = fe_space.connectivity.sizes[0];
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
            const Real x_q =
               fe_space.mesh_origin[0]
             + fe_space.h_x * (static_cast<Real>(ix) + DofPoints::GetCoord(i));
            const Real y_q =
               fe_space.mesh_origin[1]
             + fe_space.h_y * (static_cast<Real>(iy) + DofPoints::GetCoord(j));
            dofs(i, j, element_index) = State::Minus(x_q, y_q);
         }
      }
   }
}

template <Integer order, typename FiniteElementSpace, typename State>
void FillPlusDofs(
   const FiniteElementSpace& fe_space,
   Vector& x)
{
   using DofPoints = GaussLobattoLegendrePoints<order + 1>;
   auto dofs =
      MakeWriteOnlyElementTensorView<SerialKernelConfiguration>(
         fe_space,
         x);

   const GlobalIndex nx = fe_space.connectivity.sizes[0];
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
            const Real x_q =
               fe_space.mesh_origin[0]
             + fe_space.h_x * (static_cast<Real>(ix) + DofPoints::GetCoord(i));
            const Real y_q =
               fe_space.mesh_origin[1]
             + fe_space.h_y * (static_cast<Real>(iy) + DofPoints::GetCoord(j));
            dofs(i, j, element_index) = State::Plus(x_q, y_q);
         }
      }
   }
}

template <Integer order, Integer q1d, typename FiniteElementSpace>
void AccumulateCellOracle(
   const FiniteElementSpace& fe_space,
   const Vector& x,
   Vector& y,
   const Real beta_x)
{
   using QuadPoints = GaussLegendrePoints<q1d>;

   const auto x_dofs =
      MakeReadOnlyElementTensorView<SerialKernelConfiguration>(
         fe_space,
         x);
   auto y_dofs =
      MakeReadWriteElementTensorView<SerialKernelConfiguration>(
         fe_space,
         y);

   const Real h_x = fe_space.h_x;
   const Real h_y = fe_space.h_y;
   const Real det_J = h_x * h_y;

   for (GlobalIndex element_index = 0;
        element_index < fe_space.GetNumberOfCells();
        ++element_index)
   {
      for (GlobalIndex qy = 0; qy < q1d; ++qy)
      {
         const Real eta = QuadPoints::GetCoord(qy);
         const Real wy = QuadPoints::GetWeight(qy);
         for (GlobalIndex qx = 0; qx < q1d; ++qx)
         {
            const Real xi = QuadPoints::GetCoord(qx);
            const Real wx = QuadPoints::GetWeight(qx);
            const Real u_q =
               EvaluateScalarDofsAt<order>(x_dofs, element_index, xi, eta);
            const Real weight = wx * wy * det_J;

            for (GlobalIndex j = 0; j < order + 1; ++j)
            {
               for (GlobalIndex i = 0; i < order + 1; ++i)
               {
                  const Real dphi_dx =
                     ShapeGradient<order>(i, xi)
                   * ShapeValue<order>(j, eta) / h_x;
                  y_dofs(i, j, element_index) +=
                     -u_q * beta_x * dphi_dx * weight;
               }
            }
         }
      }
   }
}

template <
   Integer p_minus,
   Integer p_plus,
   int beta_x_value,
   bool include_volume,
   typename MinusSpace,
   typename PlusSpace>
Vector ApplyManualOracle(
   const MinusSpace& minus_space,
   const PlusSpace& plus_space,
   const Vector& x)
{
   constexpr Integer q1d =
      (p_minus > p_plus ? p_minus : p_plus) + 3;
   using QuadPoints = GaussLegendrePoints<q1d>;

   Vector y(
      minus_space.GetNumberOfFiniteElementDofs()
    + plus_space.GetNumberOfFiniteElementDofs());
   y = 0.0;

   const Real beta_x = static_cast<Real>(beta_x_value);

   if constexpr (include_volume)
   {
      AccumulateCellOracle<p_minus, q1d>(minus_space, x, y, beta_x);
      AccumulateCellOracle<p_plus, q1d>(plus_space, x, y, beta_x);
   }

   const auto x_minus =
      MakeReadOnlyElementTensorView<SerialKernelConfiguration>(
         minus_space,
         x);
   const auto x_plus =
      MakeReadOnlyElementTensorView<SerialKernelConfiguration>(
         plus_space,
         x);
   auto y_minus =
      MakeReadWriteElementTensorView<SerialKernelConfiguration>(
         minus_space,
         y);
   auto y_plus =
      MakeReadWriteElementTensorView<SerialKernelConfiguration>(
         plus_space,
         y);

   const Real beta_n = beta_x; // minus normal is +e_x on the coarse interface.
   constexpr Real subface_measure = 0.5;

   for (GlobalIndex jL = 0; jL < nyL; ++jL)
   {
      const GlobalIndex minus_element = (nxL - 1) + nxL * jL;
      for (GlobalIndex subface = 0; subface < 2; ++subface)
      {
         const GlobalIndex jR = 2 * jL + subface;
         const GlobalIndex plus_element = nxR * jR;
         const Real coarse_eta_origin =
            subface_measure * static_cast<Real>(subface);

         for (GlobalIndex q = 0; q < q1d; ++q)
         {
            const Real eta_leaf = QuadPoints::GetCoord(q);
            const Real w = QuadPoints::GetWeight(q);
            const Real eta_minus =
               coarse_eta_origin + subface_measure * eta_leaf;
            const Real eta_plus = eta_leaf;
            const Real u_minus =
               EvaluateScalarDofsAt<p_minus>(
                  x_minus,
                  minus_element,
                  1.0,
                  eta_minus);
            const Real u_plus =
               EvaluateScalarDofsAt<p_plus>(
                  x_plus,
                  plus_element,
                  0.0,
                  eta_plus);
            const Real u_upwind = beta_n >= 0.0 ? u_minus : u_plus;
            const Real flux =
               beta_n * u_upwind * w * subface_measure * hyL;

            for (GlobalIndex j = 0; j < p_minus + 1; ++j)
            {
               for (GlobalIndex i = 0; i < p_minus + 1; ++i)
               {
                  const Real phi =
                     ShapeValue<p_minus>(i, 1.0)
                   * ShapeValue<p_minus>(j, eta_minus);
                  y_minus(i, j, minus_element) += flux * phi;
               }
            }

            for (GlobalIndex j = 0; j < p_plus + 1; ++j)
            {
               for (GlobalIndex i = 0; i < p_plus + 1; ++i)
               {
                  const Real phi =
                     ShapeValue<p_plus>(i, 0.0)
                   * ShapeValue<p_plus>(j, eta_plus);
                  y_plus(i, j, plus_element) -= flux * phi;
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

template <typename Apply>
std::vector<Real> BuildDenseMatrix(const Integer size, const Apply& apply)
{
   std::vector<Real> dense(static_cast<size_t>(size * size), 0.0);
   for (Integer col = 0; col < size; ++col)
   {
      Vector basis(size);
      // Initialize basis vector entirely on host
      Real* basis_data = basis.WriteHostData();
      std::fill(basis_data, basis_data + size, 0.0);
      basis_data[col] = 1.0;

      const Vector y = apply(basis);
      const Real* y_data = y.ReadHostData();
      for (Integer row = 0; row < size; ++row)
      {
         dense[static_cast<size_t>(row * size + col)] = y_data[row];
      }
   }
   return dense;
}

bool CheckVectorClose(
   const char* label,
   const Vector& got,
   const Vector& expected,
   const Real tol = 1.0e-11)
{
   GENDIL_VERIFY(got.Size() == expected.Size(), "Vector sizes do not match.");
   const Real* got_data = got.ReadHostData();
   const Real* expected_data = expected.ReadHostData();
   Real max_err = 0.0;
   for (Integer i = 0; i < got.Size(); ++i)
   {
      max_err = std::max(max_err, std::abs(got_data[i] - expected_data[i]));
   }
   std::cout << label << " | vector max error = " << max_err << "\n";
   if (max_err > tol)
   {
      std::cerr << "FAILED: " << label << "\n";
      return false;
   }
   return true;
}

bool CheckDenseClose(
   const char* label,
   const std::vector<Real>& got,
   const std::vector<Real>& expected,
   const Real tol = 1.0e-11)
{
   GENDIL_VERIFY(got.size() == expected.size(), "Dense matrix sizes do not match.");
   Real max_err = 0.0;
   size_t max_err_index = 0;
   for (size_t i = 0; i < got.size(); ++i)
   {
      const Real err = std::abs(got[i] - expected[i]);
      if (err > max_err) {
         max_err = err;
         max_err_index = i;
      }
   }
   std::cout << label << " | dense max error = " << max_err << "\n";
   if (max_err > tol)
   {
      std::cerr << "FAILED: " << label << "\n";
      std::cerr << "  Max error: " << max_err << " at flat index " << max_err_index << "\n";
      std::cerr << "  Expected: " << expected[max_err_index]
                << ", Got: " << got[max_err_index] << "\n";
      return false;
   }
   return true;
}

bool CheckNear(
   const char* label,
   const Real got,
   const Real expected,
   const Real tol = 1.0e-11)
{
   const Real err = std::abs(got - expected);
   std::cout << label << " | got = " << got
             << ", expected = " << expected
             << ", error = " << err << "\n";
   if (err > tol)
   {
      std::cerr << "FAILED: " << label << "\n";
      return false;
   }
   return true;
}

Real SumRange(const Vector& y, const Integer begin, const Integer end)
{
   const Real* data = y.ReadHostData();
   Real sum = 0.0;
   for (Integer i = begin; i < end; ++i)
   {
      sum += data[i];
   }
   return sum;
}

template <Integer p_minus, Integer p_plus, int beta_x_value>
bool TestFacetOnlyCase(
   const char* label,
   const bool check_p0_sums,
   const bool check_dense)
{
   Cartesian2DMesh meshL(hx, hyL, nxL, nyL, Point<2>{0.0, 0.0});
   Cartesian2DMesh meshR(hx, hyR, nxR, nyR, Point<2>{1.0, 0.0});

   auto fe_minus =
      MakeLobattoFiniteElement(FiniteElementOrders<p_minus, p_minus>{});
   auto fe_plus =
      MakeLobattoFiniteElement(FiniteElementOrders<p_plus, p_plus>{});

   auto minus_space = MakeFiniteElementSpace(meshL, fe_minus, L2Restriction{0});
   const Integer ndofs_minus = minus_space.GetNumberOfFiniteElementDofs();
   auto plus_space =
      MakeFiniteElementSpace(meshR, fe_plus, L2Restriction{ndofs_minus});
   const Integer ndofs_plus = plus_space.GetNumberOfFiniteElementDofs();

   NonconformingCartesianIntermeshFaceConnectivity<2, 2>
      iface({nxL, nyL}, {nxR, nyR});
   auto partition =
      MakePartition(
         MakeCellPart(meshL),
         MakeCellPart(meshR),
         MakeInteriorFacePart<0, 1>(iface));
   auto mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{fe_minus, fe_plus},
         std::tuple{L2Restriction{0}, L2Restriction{ndofs_minus}});

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"mesh"> interior_facets;
   auto beta =
      MakeVectorCoefficient<"beta">(
         [] GENDIL_HOST_DEVICE () -> std::array<Real, 2>
         {
            return {static_cast<Real>(beta_x_value), 0.0};
         });
   auto weak_form =
      integrate(interior_facets, upwind(beta, u) * jump(v));
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField<"u">(mixed),
         MakeIntegrationDomain<"mesh">(mixed));

   constexpr Integer q1d =
      (p_minus > p_plus ? p_minus : p_plus) + 3;
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<q1d, q1d>{});
   auto op =
      MakeGenericOperator<GlobalFaceKernelPolicy<q1d>>(
         weak_form,
         wf_context,
         integration_rule);

   Vector x(mixed.GetNumberOfFiniteElementDofs());
   x = 0.0;
   if constexpr (p_minus == 0 && p_plus == 0)
   {
      FillMinusDofs<p_minus, decltype(minus_space), ConstantState>(
         minus_space,
         x);
      FillPlusDofs<p_plus, decltype(plus_space), ConstantState>(
         plus_space,
         x);
   }
   else
   {
      FillMinusDofs<p_minus, decltype(minus_space), LinearState>(
         minus_space,
         x);
      FillPlusDofs<p_plus, decltype(plus_space), LinearState>(
         plus_space,
         x);
   }

   const Vector y_generic = ApplyOperator(op, x);
   const Vector y_oracle =
      ApplyManualOracle<p_minus, p_plus, beta_x_value, false>(
         minus_space,
         plus_space,
         x);

   bool success = CheckVectorClose(label, y_generic, y_oracle);

   if (check_p0_sums)
   {
      constexpr Real beta_n = static_cast<Real>(beta_x_value);
      constexpr Real u_upwind = beta_n >= 0.0 ? 1.0 : 2.0;
      const Real expected_minus = beta_n * u_upwind * interface_measure;
      const Real expected_plus = -expected_minus;
      const Real minus_sum = SumRange(y_generic, 0, ndofs_minus);
      const Real plus_sum =
         SumRange(y_generic, ndofs_minus, ndofs_minus + ndofs_plus);
      success =
         CheckNear("p0 upwind minus residual sum", minus_sum, expected_minus) &&
         success;
      success =
         CheckNear("p0 upwind plus residual sum", plus_sum, expected_plus) &&
         success;
      success =
         CheckNear(
            "p0 upwind residuals cancel globally",
            minus_sum + plus_sum,
            0.0) &&
         success;
   }

   if (check_dense)
   {
      const Integer size = mixed.GetNumberOfFiniteElementDofs();
      const auto dense_generic =
         BuildDenseMatrix(
            size,
            [&] (const Vector& basis)
            {
               return ApplyOperator(op, basis);
            });
      const auto dense_oracle =
         BuildDenseMatrix(
            size,
            [&] (const Vector& basis)
            {
               return ApplyManualOracle<p_minus, p_plus, beta_x_value, false>(
                  minus_space,
                  plus_space,
                  basis);
            });
      success =
         CheckDenseClose("facet-only dense matrix oracle", dense_generic, dense_oracle) &&
         success;
   }

   return success;
}

template <Integer p_minus, Integer p_plus, int beta_x_value>
bool TestFullCellAndFacetCase(
   const char* label,
   const bool check_dense)
{
   Cartesian2DMesh meshL(hx, hyL, nxL, nyL, Point<2>{0.0, 0.0});
   Cartesian2DMesh meshR(hx, hyR, nxR, nyR, Point<2>{1.0, 0.0});

   auto fe_minus =
      MakeLobattoFiniteElement(FiniteElementOrders<p_minus, p_minus>{});
   auto fe_plus =
      MakeLobattoFiniteElement(FiniteElementOrders<p_plus, p_plus>{});

   auto minus_space = MakeFiniteElementSpace(meshL, fe_minus, L2Restriction{0});
   const Integer ndofs_minus = minus_space.GetNumberOfFiniteElementDofs();
   auto plus_space =
      MakeFiniteElementSpace(meshR, fe_plus, L2Restriction{ndofs_minus});

   NonconformingCartesianIntermeshFaceConnectivity<2, 2>
      iface({nxL, nyL}, {nxR, nyR});
   auto partition =
      MakePartition(
         MakeCellPart(meshL),
         MakeCellPart(meshR),
         MakeInteriorFacePart<0, 1>(iface));
   auto mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{fe_minus, fe_plus},
         std::tuple{L2Restriction{0}, L2Restriction{ndofs_minus}});

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   Cells<"mesh"> cells;
   InteriorFacets<"mesh"> interior_facets;
   auto beta =
      MakeVectorCoefficient<"beta">(
         [] GENDIL_HOST_DEVICE () -> std::array<Real, 2>
         {
            return {static_cast<Real>(beta_x_value), 0.0};
         });
   auto weak_form =
      integrate(cells, -u * dot(beta, grad(v)))
    + integrate(interior_facets, upwind(beta, u) * jump(v));
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField<"u">(mixed),
         MakeIntegrationDomain<"mesh">(mixed));

   constexpr Integer q1d =
      (p_minus > p_plus ? p_minus : p_plus) + 3;
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<q1d, q1d>{});
   auto op =
      MakeGenericOperator<GlobalFaceKernelPolicy<q1d>>(
         weak_form,
         wf_context,
         integration_rule);

   Vector x(mixed.GetNumberOfFiniteElementDofs());
   x = 0.0;
   FillMinusDofs<p_minus, decltype(minus_space), LinearState>(
      minus_space,
      x);
   FillPlusDofs<p_plus, decltype(plus_space), LinearState>(
      plus_space,
      x);

   const Vector y_generic = ApplyOperator(op, x);
   const Vector y_oracle =
      ApplyManualOracle<p_minus, p_plus, beta_x_value, true>(
         minus_space,
         plus_space,
         x);

   bool success = CheckVectorClose(label, y_generic, y_oracle);

   if (check_dense)
   {
      const Integer size = mixed.GetNumberOfFiniteElementDofs();
      const auto dense_generic =
         BuildDenseMatrix(
            size,
            [&] (const Vector& basis)
            {
               return ApplyOperator(op, basis);
            });
      const auto dense_oracle =
         BuildDenseMatrix(
            size,
            [&] (const Vector& basis)
            {
               return ApplyManualOracle<p_minus, p_plus, beta_x_value, true>(
                  minus_space,
                  plus_space,
                  basis);
            });
      success =
         CheckDenseClose(
            "full cell+facet dense matrix oracle",
            dense_generic,
            dense_oracle) &&
         success;
   }

   return success;
}

} // namespace

int main()
{
   bool success = true;

   success =
      TestFacetOnlyCase<0, 0, 2>(
         "h-adaptive p0/p0 facet-only positive beta",
         true,
         true) &&
      success;
   success =
      TestFacetOnlyCase<0, 0, -3>(
         "h-adaptive p0/p0 facet-only negative beta",
         true,
         true) &&
      success;
   success =
      TestFacetOnlyCase<1, 1, 2>(
         "h-adaptive p1/p1 nonconstant facet-only positive beta",
         false,
         false) &&
      success;
   success =
      TestFacetOnlyCase<1, 1, -3>(
         "h-adaptive p1/p1 nonconstant facet-only negative beta",
         false,
         false) &&
      success;
   success =
      TestFacetOnlyCase<1, 2, 2>(
         "h+p p1-minus/p2-plus facet-only positive beta",
         false,
         true) &&
      success;
   success =
      TestFacetOnlyCase<1, 2, -3>(
         "h+p p1-minus/p2-plus facet-only negative beta",
         false,
         true) &&
      success;
   success =
      TestFacetOnlyCase<2, 1, 2>(
         "h+p p2-minus/p1-plus facet-only positive beta",
         false,
         true) &&
      success;
   success =
      TestFacetOnlyCase<2, 1, -3>(
         "h+p p2-minus/p1-plus facet-only negative beta",
         false,
         true) &&
      success;

   // Boundary facets are intentionally omitted in this full cell+facet check.
   // The oracle includes only the volume term plus the same nonconforming
   // global interface term as the weak form.
   success =
      TestFullCellAndFacetCase<1, 1, 2>(
         "h-adaptive p1/p1 full cell+facet positive beta",
         false) &&
      success;
   success =
      TestFullCellAndFacetCase<1, 1, -3>(
         "h-adaptive p1/p1 full cell+facet negative beta",
         false) &&
      success;
   success =
      TestFullCellAndFacetCase<1, 2, 2>(
         "h+p p1-minus/p2-plus full cell+facet positive beta",
         true) &&
      success;
   success =
      TestFullCellAndFacetCase<1, 2, -3>(
         "h+p p1-minus/p2-plus full cell+facet negative beta",
         true) &&
      success;
   success =
      TestFullCellAndFacetCase<2, 1, 2>(
         "h+p p2-minus/p1-plus full cell+facet positive beta",
         true) &&
      success;
   success =
      TestFullCellAndFacetCase<2, 1, -3>(
         "h+p p2-minus/p1-plus full cell+facet negative beta",
         true) &&
      success;

   if (!success)
   {
      return 1;
   }

   std::cout << "GenericOperator h/hp adaptive upwind tests passed.\n";
   return 0;
}

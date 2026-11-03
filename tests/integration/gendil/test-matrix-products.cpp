// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <tuple>
#include <type_traits>

namespace matrix_product_test
{
using namespace gendil;
using Matrix = std::array<std::array<Real, 2>, 2>;

struct ScalarProxy
{
   Real value;
   GENDIL_HOST_DEVICE operator Real() const { return value; }
   friend Real operator*(ScalarProxy, ScalarProxy) = delete;
   friend Real operator+(ScalarProxy, ScalarProxy) = delete;
};

// A read-only owning coefficient value: neither proxy arithmetic nor writable
// entries are available to multiplication or the pullback's transposes.
struct ReadableMatrix
{
   using value_type = Real;
   Matrix values;
   ReadableMatrix() = delete;
   GENDIL_HOST_DEVICE explicit ReadableMatrix(Matrix values_) : values(values_) {}
   template<class I, class J> requires (std::is_integral_v<I> && std::is_integral_v<J>)
   GENDIL_HOST_DEVICE ScalarProxy operator()(I i, J j) const
   { return {values[i][j]}; }
};
} // namespace matrix_product_test

namespace gendil
{
template<> struct static_num_rows<matrix_product_test::ReadableMatrix>
   : std::integral_constant<Integer, 2> {};
template<> struct static_num_cols<matrix_product_test::ReadableMatrix>
   : std::integral_constant<Integer, 2> {};
static_assert(StaticMatrix<matrix_product_test::ReadableMatrix>);
} // namespace gendil

namespace
{
using namespace gendil;
using matrix_product_test::Matrix;
using matrix_product_test::ReadableMatrix;
constexpr Real hx = 0.4, hy = 0.7, test_weight = 0.3;
enum class Storage { Standard, Recursive, Proxy };
enum class Region { Cell, Boundary, Interior };
enum class Contribution { TrialProduct, TestProduct, Both };

template<Integer Which>
GENDIL_HOST_DEVICE Matrix MatrixEntries()
{
   // Nonsymmetric, noncommuting factors distinguish product order and adjoints.
   if constexpr (Which == 0) return {{{2.0, 1.0}, {-0.5, 3.0}}};
   else return {{{1.0, -2.0}, {0.25, 0.5}}};
}

template<Storage Representation, Integer Which>
GENDIL_HOST_DEVICE auto MatrixValue()
{
   const auto entries = MatrixEntries<Which>();
   if constexpr (Representation == Storage::Standard) return entries;
   else if constexpr (Representation == Storage::Proxy) return ReadableMatrix{entries};
   else
   {
      SerialRecursiveArray<Real, 2, 2> result;
      for (Integer i = 0; i < 2; ++i)
         for (Integer j = 0; j < 2; ++j) result(i, j) = entries[i][j];
      return result;
   }
}

// Independent Q1 assembly. The oracle uses explicit basis derivatives and
// scalar index sums, never GenDiL matrix multiplication, transposes or traces.
Vector Expected(const Vector& input, Region region, Contribution contribution)
{
   Vector result(input.Size());
   result = 0.0;
   const Real* x = input.ReadHostData();
   Real* y = result.ReadWriteHostData();
   const Matrix A = MatrixEntries<0>(), B = MatrixEntries<1>();
   const std::array<Real, 2> points{0.5 - std::sqrt(1.0 / 12.0),
                                  0.5 + std::sqrt(1.0 / 12.0)};
   auto add_point = [&](Integer minus, Integer plus, Real xi, Real eta,
                        Real plus_xi, Real plus_eta, Real weight) {
      const Integer sides = region == Region::Interior ? 2 : 1;
      const std::array<Integer, 2> cells{minus, plus};
      const std::array<Real, 2> xis{xi, plus_xi}, etas{eta, plus_eta};
      std::array<std::array<std::array<Real, 2>, 4>, 2> derivative{};
      Matrix gradient{}, flux{};
      for (Integer side = 0; side < sides; ++side)
         for (Integer j = 0; j < 2; ++j)
            for (Integer i = 0; i < 2; ++i)
            {
               const Integer node = i + 2 * j;
               derivative[side][node] = {
                  (2.0 * i - 1.0) * (j == 0 ? 1.0 - etas[side] : etas[side]) / hx,
                  (i == 0 ? 1.0 - xis[side] : xis[side]) * (2.0 * j - 1.0) / hy};
               for (Integer c = 0; c < 2; ++c)
                  for (Integer d = 0; d < 2; ++d)
                     gradient[c][d] += (side == 0 ? 1.0 : -1.0) *
                        x[24 * c + 4 * cells[side] + node] * derivative[side][node][d];
            }
      for (Integer c = 0; c < 2; ++c)
         for (Integer d = 0; d < 2; ++d)
            for (Integer i = 0; i < 2; ++i)
               for (Integer j = 0; j < 2; ++j)
               {
                  if (contribution != Contribution::TestProduct)
                     flux[c][d] += A[c][i] * gradient[i][j] * B[j][d];
                  if (contribution != Contribution::TrialProduct)
                     flux[c][d] += test_weight * A[i][c] * gradient[i][j] * B[d][j];
               }
      for (Integer side = 0; side < sides; ++side)
         for (Integer c = 0; c < 2; ++c)
            for (Integer node = 0; node < 4; ++node)
               for (Integer d = 0; d < 2; ++d)
                  y[24 * c + 4 * cells[side] + node] += weight *
                     (side == 0 ? 1.0 : -1.0) * flux[c][d] * derivative[side][node][d];
   };
   for (Integer row = 0; row < 3; ++row)
      for (Integer column = 0; column < 2; ++column)
      {
         const Integer cell = column + 2 * row;
         if (region == Region::Cell)
            for (Real eta : points)
               for (Real xi : points) add_point(cell, 0, xi, eta, 0, 0, 0.25 * hx * hy);
         else if (region == Region::Boundary)
            for (Integer axis = 0; axis < 2; ++axis)
               for (Integer side = 0; side < 2; ++side)
               {
                  const Integer index = axis == 0 ? column : row;
                  const Integer size = axis == 0 ? 2 : 3;
                  if (index != (side == 0 ? 0 : size - 1)) continue;
                  for (Real q : points)
                     add_point(cell, 0, axis == 0 ? side : q, axis == 1 ? side : q,
                               0, 0, 0.5 * (axis == 0 ? hy : hx));
               }
         else
         {
            if (column == 0)
               for (Real q : points) add_point(cell, cell + 1, 1, q, 0, q, 0.5 * hy);
            if (row < 2)
               for (Real q : points) add_point(cell, cell + 2, q, 1, q, 0, 0.5 * hx);
         }
      }
   return result;
}

bool Check(const char* label, const Vector& actual, const Vector& expected)
{
   if (actual.Size() != expected.Size())
   {
      std::cerr << label << ": mismatched output sizes\n";
      return false;
   }
   const Real* a = actual.ReadHostData();
   const Real* e = expected.ReadHostData();
   Real error = 0.0, scale = 1.0;
   for (Integer i = 0; i < actual.Size(); ++i)
   {
      if (!std::isfinite(a[i]) || !std::isfinite(e[i]))
      {
         std::cerr << label << ": non-finite entry " << i << '\n';
         return false;
      }
      error = std::max(error, std::abs(a[i] - e[i]));
      scale = std::max(scale, std::abs(e[i]));
   }
   if (error <= 3.0e-11 * scale) return true;
   std::cerr << label << ": scaled error " << error / scale << '\n';
   return false;
}

template<class Policy, class Form, class Context, class Rule>
Vector Evaluate(const Form& form, const Context& context, const Rule& rule, const Vector& input)
{
   auto op = MakeGenericOperator<Policy>(form, context, rule);
   Vector output(input.Size());
   output = 0.0;
   op(input, output);
   GENDIL_DEVICE_SYNC;
   return output;
}

template<Storage Representation, Region Entity, class Policy, class Context, class Rule>
bool CheckForms(const Context& context, const Rule& rule, const Vector& input)
{
   VectorTrialSpace<"u"> u;
   VectorTestSpace<"v"> v;
   const auto domain = [] {
      if constexpr (Entity == Region::Cell) return Cells<"m">{};
      else if constexpr (Entity == Region::Boundary) return BoundaryFacets<"m">{};
      else return InteriorFacets<"m">{};
   }();
   const auto trial = [&] {
      if constexpr (Entity == Region::Interior) return jump(grad(u));
      else return grad(u);
   }();
   const auto test = [&] {
      if constexpr (Entity == Region::Interior) return jump(grad(v));
      else return grad(v);
   }();
   const auto A = MakeMatrixCoefficient<"A">(
      [] GENDIL_HOST_DEVICE () { return MatrixValue<Representation, 0>(); });
   const auto B = MakeMatrixCoefficient<"B">(
      [] GENDIL_HOST_DEVICE () { return MatrixValue<Representation, 1>(); });
   const auto trial_product = inner(A * trial * B, test);
   // Nested products exercise both adjoints: the test is the right operand
   // of A*test, and that product is the left operand of (A*test)*B.
   const auto test_product = test_weight * inner(trial, A * test * B);
   bool success = Check("trial matrix products / independent assembly",
      Evaluate<Policy>(integrate(domain, trial_product), context, rule, input),
      Expected(input, Entity, Contribution::TrialProduct));
   success = Check("both matrix pullbacks / independent assembly",
      Evaluate<Policy>(integrate(domain, test_product), context, rule, input),
      Expected(input, Entity, Contribution::TestProduct)) && success;
   const auto combined = Evaluate<Policy>(integrate(domain, trial_product + test_product),
      context, rule, input);
   success = Check("combined matrix products / independent assembly", combined,
      Expected(input, Entity, Contribution::Both)) && success;
   success = Check("split / combined matrix products", combined,
      Evaluate<Policy>(integrate(domain, trial_product) + integrate(domain, test_product),
         context, rule, input)) && success;
   if (!success)
      std::cerr << "region " << int(Entity) << ", storage " << int(Representation)
                << ", device " << Policy::is_device_configuration << '\n';
   return success;
}

template<Storage Representation, class LocalPolicy, class GlobalPolicy,
         class LocalContext, class GlobalContext, class Rule>
bool CheckDomains(const LocalContext& local, const GlobalContext& global,
                  const Rule& rule, const Vector& input)
{
   bool success = CheckForms<Representation, Region::Cell, LocalPolicy>(local, rule, input);
   success = CheckForms<Representation, Region::Boundary, LocalPolicy>(local, rule, input) && success;
   success = CheckForms<Representation, Region::Interior, LocalPolicy>(local, rule, input) && success;
   success = CheckForms<Representation, Region::Interior, GlobalPolicy>(global, rule, input) && success;
   return success;
}
} // namespace

int main()
{
   const CartesianMesh<2> mesh({2, 3}, {hx, hy}, {0.0, 0.0}, false);
   const auto scalar = MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{});
   const auto fe = MakeVectorFiniteElement(scalar, scalar);
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<3, 3>{});
   const auto local_space = MakeFiniteElementSpace(mesh, fe);
   const auto local_context = MakeWeakFormContext(MakeTrialField<"u">(local_space),
      MakeTestField<"v">(local_space), MakeIntegrationDomain<"m">(mesh));
   const auto partition = MakePartition(MakeCellPart(mesh),
      MakeInteriorFacePart<0, 0>(MakeCartesianInteriorFaceConnectivity<2>({2, 3})));
   const auto global_space = MakeMixedFiniteElementSpace(partition, std::tuple{fe}, DGDirectSumNumbering{});
   const auto global_context = MakeWeakFormContext(MakeTrialField<"u">(global_space),
      MakeTestField<"v">(global_space), MakeIntegrationDomain<"m">(partition));
   Vector input(local_space.GetNumberOfFiniteElementDofs());
   Real* x = input.WriteHostData();
   for (Integer i = 0; i < input.Size(); ++i) x[i] = 0.25 + 0.03 * i + 0.002 * i * i;

   auto run = [&]<Storage Representation>() {
      bool success = CheckDomains<Representation, SerialKernelConfiguration, SerialKernelConfiguration>(
         local_context, global_context, rule, input);
#ifdef GENDIL_USE_DEVICE
      // Six cells and seven interior faces both leave incomplete batches.
      using LocalPolicy = DeviceKernelConfiguration<ThreadBlockLayout<4, 4>, 4, 1>;
      using GlobalPolicy = DeviceKernelConfiguration<ThreadBlockLayout<4, 4>, 2, 4>;
      success = CheckDomains<Representation, LocalPolicy, GlobalPolicy>(
         local_context, global_context, rule, input) && success;
#endif
      return success;
   };
   bool success = run.template operator()<Storage::Standard>();
   success = run.template operator()<Storage::Recursive>() && success;
   success = run.template operator()<Storage::Proxy>() && success;
   if (!success) return 1;
   std::cout << "Matrix products and both pullbacks match independent assembly.\n";
   return 0;
}

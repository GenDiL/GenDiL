// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace facet_arithmetic_test
{
struct ScalarProxy
{
   gendil::Real value;
   GENDIL_HOST_DEVICE constexpr operator gendil::Real() const { return value; }
   friend gendil::Real operator+(ScalarProxy, ScalarProxy) = delete;
   friend gendil::Real operator-(ScalarProxy, ScalarProxy) = delete;
   friend gendil::Real operator-(ScalarProxy) = delete;
};

// Coefficients own their data, but expose no writable scalar accessor.
struct ReadableVector
{
   using value_type = gendil::Real;
   std::array<value_type, 2> values;
   GENDIL_HOST_DEVICE constexpr ScalarProxy operator()(gendil::Integer i) const
   { return {values[i]}; }
};

struct ReadableMatrix
{
   using value_type = gendil::Real;
   std::array<std::array<value_type, 2>, 2> values;
   GENDIL_HOST_DEVICE constexpr ScalarProxy operator()(gendil::Integer i, gendil::Integer j) const
   { return {values[i][j]}; }
};
} // namespace facet_arithmetic_test

namespace gendil
{
template<> struct static_extent<facet_arithmetic_test::ReadableVector>
   : std::integral_constant<Integer, 2> {};
template<> struct static_num_rows<facet_arithmetic_test::ReadableMatrix>
   : std::integral_constant<Integer, 2> {};
template<> struct static_num_cols<facet_arithmetic_test::ReadableMatrix>
   : std::integral_constant<Integer, 2> {};
static_assert(StaticVector<facet_arithmetic_test::ReadableVector>);
static_assert(StaticMatrix<facet_arithmetic_test::ReadableMatrix>);
} // namespace gendil

using namespace gendil;

namespace facet_arithmetic_test
{
// Value-owned geometry keeps the same unequal-width fixture usable on a GPU.
struct UnequalCell
{
   static constexpr Integer Dim = 2;
   using geometry = HyperCube<2>;
   using physical_coordinates = std::array<Real, 2>;
   using jacobian = std::array<Real, 2>;
   template<class Rule>
   using QuadData = typename HyperCubeCell<2>::template QuadData<Rule>;

   Point<2> origin;
   Real width;

   template<class Data>
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian(const TensorIndex<2>& qi, const Data& data,
                             physical_coordinates& x, jacobian& J) const
   {
      x = {origin[0] + width * GetCoord<0>(data, qi[0]),
           origin[1] + GetCoord<1>(data, qi[1])};
      J = {width, 1.0};
   }
};

GENDIL_HOST_DEVICE
void ApplyOrientationToCell(const IdentityOrientation<2>&, UnequalCell&) {}

struct UnequalMesh : CartesianMesh<2>
{
   using cell_type = UnequalCell;
   UnequalMesh() : CartesianMesh<2>({2, 3}, {1.0, 1.0}, {0.0, 0.0}, false) {}

   GENDIL_HOST_DEVICE
   cell_type GetCell(GlobalIndex element) const
   {
      const auto indices = GetStructuredSubIndices(element, this->sizes);
      return {{Real(indices[0]), Real(indices[1])}, indices[0] == 0 ? 1.0 : 2.0};
   }
};

enum class Representation { StandardArray, RecursiveArray, ReadableProxy };

template<Representation Storage>
auto MakeTerms()
{
   auto a = MakeVectorCoefficient<"a", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real r) {
         if constexpr (Storage == Representation::StandardArray)
            return std::array<Real, 2>{r, 2.0 * r + 1.0};
         else if constexpr (Storage == Representation::ReadableProxy)
            return facet_arithmetic_test::ReadableVector{{r, 2.0 * r + 1.0}};
         else
         {
            SerialRecursiveArray<Real, 2> result;
            result(0) = r; result(1) = 2.0 * r + 1.0;
            return result;
         }
      });
   auto b = MakeVectorCoefficient<"b", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real r) {
         SerialRecursiveArray<Real, 2> result;
         result(0) = 3.0 * r - 1.0; result(1) = -r;
         return result;
      });
   auto A = MakeMatrixCoefficient<"A", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real r) {
         if constexpr (Storage == Representation::StandardArray)
            return std::array<std::array<Real, 2>, 2>{{{r, 2.0 * r + 1.0}, {-r, r * r}}};
         else if constexpr (Storage == Representation::ReadableProxy)
            return facet_arithmetic_test::ReadableMatrix{{{{r, 2.0 * r + 1.0}, {-r, r * r}}}};
         else
         {
            SerialRecursiveArray<Real, 2, 2> result;
            result(0, 0) = r; result(0, 1) = 2.0 * r + 1.0;
            result(1, 0) = -r; result(1, 1) = r * r;
            return result;
         }
      });
   auto B = MakeMatrixCoefficient<"B", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real r) {
         SerialRecursiveArray<Real, 2, 2> result;
         result(0, 0) = 2.0 * r - 1.0; result(0, 1) = -r;
         result(1, 0) = r + 2.0; result(1, 1) = 3.0 * r;
         return result;
      });

   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   // Both orders of mixed storage addition/subtraction occur. Nonlinear r*r
   // also distinguishes averaging callback results from averaging their input.
   const auto vector_weight = dot(jump(a), jump(a))
      + dot(average(a + b) - average(a), average(a - b));
   const auto matrix_weight = inner(jump(A), jump(A))
      + inner(average(B + A) - average(A), -average(B - A));
   return std::tuple{vector_weight * jump(u) * jump(v),
                     matrix_weight * jump(u) * jump(v)};
}

enum class Contribution { Vector, Matrix, Both };

// Independent P0 edge assembly: physical widths are 1 and 2, row heights 1.
// These closed scalar formulas use no DSL or array-arithmetic implementation.
Vector Expected(const Vector& input, Contribution contribution)
{
   Vector result(input.Size());
   result = 0.0;
   const Real* x = input.ReadHostData();
   Real* y = result.ReadWriteHostData();
   auto add = [&](Integer left, Integer right, Real r, Real s, Real measure) {
      const Real mean = 0.5 * (r + s), delta = r - s;
      const Real mean_square = 0.5 * (r * r + s * s);
      const Real vector_weight = 5.0 * delta * delta
         + (3.0 * mean - 1.0) * (1.0 - 2.0 * mean)
         - mean * (3.0 * mean + 1.0);
      const Real matrix_weight = 6.0 * delta * delta
         + (r * r - s * s) * (r * r - s * s)
         + (2.0 * mean - 1.0) * (1.0 - mean)
         - mean * (3.0 * mean + 1.0)
         - (mean + 2.0) * (2.0 * mean + 2.0)
         + 3.0 * mean * (mean_square - 3.0 * mean);
      const Real weight = contribution == Contribution::Vector ? vector_weight
         : contribution == Contribution::Matrix ? matrix_weight : vector_weight + matrix_weight;
      const Real flux = measure * weight * (x[left] - x[right]);
      y[left] += flux;
      y[right] -= flux;
   };
   for (Integer row = 0; row < 3; ++row) add(2 * row, 2 * row + 1, 1.0, 0.5, 1.0);
   for (Integer row = 0; row < 2; ++row)
      for (Integer column = 0; column < 2; ++column)
         add(2 * row + column, 2 * (row + 1) + column,
             1.0, 1.0, column == 0 ? 1.0 : 2.0);
   return result;
}

bool Check(const char* label, const Vector& actual, const Vector& expected)
{
   const Real* a = actual.ReadHostData();
   const Real* e = expected.ReadHostData();
   Real error = 0.0, scale = 1.0;
   for (Integer i = 0; i < actual.Size(); ++i)
   {
      if (!std::isfinite(a[i]) || !std::isfinite(e[i]))
      {
         std::cerr << label << ": non-finite result\n";
         return false;
      }
      error = std::max(error, std::abs(a[i] - e[i]));
      scale = std::max(scale, std::abs(e[i]));
   }
   if (error <= 2.0e-11 * scale) return true;
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

template<Representation Storage, class Policy, class Context, class Rule>
auto CheckRepresentation(const Context& context, const Rule& rule, const Vector& input)
{
   InteriorFacets<"m"> faces;
   const auto [vector_term, matrix_term] = MakeTerms<Storage>();
   auto observed = Evaluate<Policy>(integrate(faces, vector_term + matrix_term), context, rule, input);
   bool success = Check("coefficient arithmetic / analytic edge assembly", observed,
                        Expected(input, Contribution::Both));
   success = Check("vector coefficient arithmetic", Evaluate<Policy>(
      integrate(faces, vector_term), context, rule, input), Expected(input, Contribution::Vector)) && success;
   success = Check("matrix coefficient arithmetic", Evaluate<Policy>(
      integrate(faces, matrix_term), context, rule, input), Expected(input, Contribution::Matrix)) && success;
   success = Check("split / combined integrands", observed, Evaluate<Policy>(
      integrate(faces, vector_term) + integrate(faces, matrix_term), context, rule, input)) && success;
   return std::pair{success, std::move(observed)};
}

template<class Policy, class Context, class Rule>
auto CheckForms(const Context& context, const Rule& rule, const Vector& input)
{
   auto [array_ok, array] = CheckRepresentation<Representation::StandardArray, Policy>(context, rule, input);
   const auto [proxy_ok, proxy] = CheckRepresentation<Representation::ReadableProxy, Policy>(context, rule, input);
   const auto [recursive_vector, recursive_matrix] = MakeTerms<Representation::RecursiveArray>();
   bool success = Check("array / proxy coefficients", array, proxy) && array_ok && proxy_ok;
   success = Check("array / recursive coefficients", array, Evaluate<Policy>(
      integrate(InteriorFacets<"m">{}, recursive_vector + recursive_matrix), context, rule, input)) && success;
   return std::pair{success, std::move(array)};
}
} // namespace facet_arithmetic_test

int main()
{
   using namespace facet_arithmetic_test;
   const UnequalMesh mesh;
   const auto fe = MakeLegendreFiniteElement(FiniteElementOrders<0, 0>{});
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
   for (Integer i = 0; i < input.Size(); ++i) x[i] = 0.25 + 0.4 * i + 0.07 * i * i;

   const auto [local_ok, local] = CheckForms<SerialKernelConfiguration>(local_context, rule, input);
   const auto [global_ok, global] = CheckForms<SerialKernelConfiguration>(global_context, rule, input);
   bool success = Check("local / global facet paths", local, global) && local_ok && global_ok;
#if defined(GENDIL_USE_DEVICE)
   using LocalPolicy = DeviceKernelConfiguration<ThreadBlockLayout<4, 4>, 2, 1>;
   using GlobalPolicy = DeviceKernelConfiguration<ThreadBlockLayout<4, 4>, 2, 4>;
   const auto [device_local_ok, device_local] = CheckForms<LocalPolicy>(local_context, rule, input);
   const auto [device_global_ok, device_global] = CheckForms<GlobalPolicy>(global_context, rule, input);
   success = Check("device / serial local facets", device_local, local) && success;
   success = Check("device / serial global facets", device_global, global) && success;
   success = device_local_ok && device_global_ok && success;
#endif
   if (!success) return 1;
   std::cout << "Facet coefficient arithmetic and independent edge assembly passed.\n";
   return 0;
}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{
constexpr Real curvature = 0.4;

// A curved tensor mesh stored entirely by value, so the exact same fixture is
// accessible to host and device kernels. Adjacent x widths are 1 and 2.
template<Integer D>
struct CurvedCell
{
   static constexpr Integer Dim = D;
   using geometry = HyperCube<D>;
   using physical_coordinates = std::array<Real, D>;
   using jacobian = std::array<std::array<Real, D>, D>;
   template<class Rule>
   using QuadData = typename HyperCubeCell<D>::template QuadData<Rule>;

   Point<D> origin;
   Real width;

   template<class Data>
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian(const TensorIndex<D>& qi, const Data& data,
                             physical_coordinates& x, jacobian& J) const
   {
      std::array<Real, D> q{};
      ConstexprLoop<D>([&](auto d) { q[d] = GetCoord<d>(data, qi[d]); });
      J = {};
      for (Integer d = 0; d < D; ++d)
      {
         x[d] = origin[d] + q[d];
         J[d][d] = 1.0;
      }
      const Real t = q[1];
      Real displacement = curvature * t * (1.0 - t);
      Real dt = curvature * (1.0 - 2.0 * t);
      if constexpr (D == 3)
      {
         const Real z = q[2];
         J[2][0] = displacement * (1.0 - 2.0 * z);
         displacement *= z * (1.0 - z);
         dt *= z * (1.0 - z);
      }
      x[0] = origin[0] + width * q[0] + displacement;
      J[0][0] = width;
      J[1][0] = dt;
   }
};

template<Integer D>
GENDIL_HOST_DEVICE
void ApplyOrientationToCell(const IdentityOrientation<D>&, CurvedCell<D>&) {}

template<Integer D>
struct CurvedMesh : CartesianMesh<D>
{
   using cell_type = CurvedCell<D>;
   explicit CurvedMesh(const std::array<GlobalIndex, D>& sizes)
      : CartesianMesh<D>(sizes, [] {
           std::array<Real, D> h{};
           h.fill(1.0);
           return h;
        }(), Point<D>{}, false) {}

   GENDIL_HOST_DEVICE
   cell_type GetCell(GlobalIndex element) const
   {
      const auto indices = GetStructuredSubIndices(element, this->sizes);
      Point<D> origin{};
      for (Integer d = 0; d < D; ++d) origin[d] = Real(indices[d]);
      return {origin, indices[0] == 0 ? 1.0 : 2.0};
   }
};

// Squared inverse normal scale on an x face before dividing by width^2.
// y/z faces have unit normal scale. The test evaluates at interior Gauss points,
// so transverse integer coordinates identify y/z faces without corner ambiguity.
template<Integer D>
GENDIL_HOST_DEVICE
Real NormalMetricSquared(const std::array<Real, D>& x)
{
   const Real t = x[1] - std::floor(x[1]);
   Real dt = curvature * (1.0 - 2.0 * t);
   Real dz = 0.0;
   if constexpr (D == 3)
   {
      const Real z = x[2] - std::floor(x[2]);
      dz = curvature * t * (1.0 - t) * (1.0 - 2.0 * z);
      dt *= z * (1.0 - z);
   }
   return 1.0 + dt * dt + dz * dz;
}

template<Integer D>
GENDIL_HOST_DEVICE
bool TransverseFace(const std::array<Real, D>& x)
{
   for (Integer d = 1; d < D; ++d)
      if (std::abs(x[d] - std::floor(x[d] + 0.5)) < 1.0e-10) return true;
   return false;
}

#if defined(GENDIL_USE_DEVICE)
template<class Layout, Integer Shared, Integer Batch>
using TestPolicy = DeviceKernelConfiguration<Layout, Shared, Batch>;
#else
template<class Layout, Integer Shared, Integer Batch>
using TestPolicy = SerialKernelConfiguration;
#endif

template<Integer D, size_t... I>
auto MakeOrders(std::index_sequence<I...>)
{ return FiniteElementOrders<(static_cast<void>(I), 2)...>{}; }

template<Integer D, size_t... I>
auto MakeRule(std::index_sequence<I...>)
{ return MakeIntegrationRule(IntegrationRuleNumPoints<(static_cast<void>(I), 4)...>{}); }

bool Check(const char* label, const Vector& actual, const Vector& expected)
{
   const auto* a = actual.ReadHostData();
   const auto* e = expected.ReadHostData();
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
   if (error > 1.0e-10 * scale)
   {
      std::cerr << label << ": scaled error " << error / scale << '\n';
      return false;
   }
   return true;
}

template<class Operator>
Vector Apply(const Operator& op, const Vector& input)
{
   Vector output(input.Size());
   output = 0.0;
   op(input, output);
   GENDIL_DEVICE_SYNC;
   output.ReadHostData();
   return output;
}

template<class Policy, class Form, class Context, class Rule>
Vector Evaluate(const Form& form, const Context& context, const Rule& rule,
                const Vector& input)
{
   return Apply(MakeGenericOperator<Policy>(form, context, rule), input);
}

template<Integer D, bool Global, class Policy, class Context, class Rule>
bool CheckForms(const Context& context, const Rule& rule, const Vector& input)
{
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"m"> faces;
   BoundaryFacets<"m"> boundary;
   auto rho_squared = MakeCoefficient<"rho_squared", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real rho) { return rho * rho; });
   auto expected_interior = MakeCoefficient<"expected_interior", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const std::array<Real, D>& x) {
         return TransverseFace<D>(x) ? 1.0 : 0.625 * NormalMetricSquared<D>(x);
      });
   auto expected_boundary = MakeCoefficient<"expected_boundary", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const std::array<Real, D>& x) {
         if (TransverseFace<D>(x)) return 1.0;
         return NormalMetricSquared<D>(x) * (x[0] < 1.0 ? 1.0 : 0.25);
      });
   const auto penalty = 90.0 * average(rho_squared) * jump(u) * jump(v);
   const auto flux = -average(dot(grad(u), Normal{})) * jump(v)
                     - jump(u) * average(dot(grad(v), Normal{}));
   const auto combined = integrate(faces, flux + penalty);
   const auto split = integrate(faces, flux) + integrate(faces, penalty);
   const auto observed = Evaluate<Policy>(combined, context, rule, input);
   bool success = Check("device/serial SIPDG", observed,
      Evaluate<SerialKernelConfiguration>(combined, context, rule, input));
   success = Check("split/combined SIPDG", observed,
      Evaluate<Policy>(split, context, rule, input)) && success;
   const auto flux_y = Evaluate<Policy>(integrate(faces, flux), context, rule, input);
   const auto penalty_y = Evaluate<Policy>(integrate(faces, penalty), context, rule, input);
   Vector sum(input.Size());
   Real* sum_data = sum.WriteHostData();
   const Real* flux_data = flux_y.ReadHostData();
   const Real* penalty_data = penalty_y.ReadHostData();
   for (Integer i = 0; i < input.Size(); ++i) sum_data[i] = flux_data[i] + penalty_data[i];
   success = Check("separate/combined SIPDG", observed, sum) && success;
   success = Check("nonlinear normal-scale average", penalty_y,
      Evaluate<SerialKernelConfiguration>(integrate(faces,
         90.0 * expected_interior * jump(u) * jump(v)), context, rule, input)) && success;
   success = Check("boundary normal scale",
      Evaluate<Policy>(integrate(boundary, rho_squared * u * v), context, rule, input),
      Evaluate<SerialKernelConfiguration>(integrate(boundary, expected_boundary * u * v),
         context, rule, input)) && success;
   if constexpr (Global)
   {
      const auto raw = GenericAssembly<MatrixAssemblyType::RawCOO, Policy>(combined, context, rule);
      const auto coo = FinalizeRawCOOToCOOHost(raw);
      success = Check("global assembled/matrix-free SIPDG", Apply(coo, input), observed) && success;
   }
   return success;
}

template<Integer D>
bool RunDimension()
{
   std::array<GlobalIndex, D> sizes{};
   sizes.fill(2);
   sizes[1] = 3; // The x-face family has 3 (2D) or 6 (3D) faces: partial batches.
   const CurvedMesh<D> mesh(sizes);
   const auto fe = MakeLobattoFiniteElement(MakeOrders<D>(std::make_index_sequence<D>{}));
   const auto rule = MakeRule<D>(std::make_index_sequence<D>{});
   const auto local_space = MakeFiniteElementSpace(mesh, fe);
   const auto local_context = MakeWeakFormContext(
      MakeTrialField<"u">(local_space), MakeIntegrationDomain<"m">(mesh));
   Vector input(local_space.GetNumberOfFiniteElementDofs());
   Real* data = input.WriteHostData();
   for (Integer i = 0; i < input.Size(); ++i)
      data[i] = 0.1 + 0.07 * i + 0.03 * ((i * 7) % 11);
   bool success = CheckForms<D, false, TestPolicy<ThreadBlockLayout<2>, 1, 1>>(
      local_context, rule, input);
   success = CheckForms<D, false, TestPolicy<ThreadBlockLayout<8, 4>, 2, 1>>(
      local_context, rule, input) && success;

   const auto partition = MakePartition(MakeCellPart(mesh),
      MakeInteriorFacePart<0, 0>(MakeCartesianInteriorFaceConnectivity<D>(sizes)),
      MakeBoundaryFacePart<0>(MakeCartesianBoundaryFaceConnectivity<D>(sizes)));
   const auto global_space = MakeMixedFiniteElementSpace(partition, std::tuple{fe},
      DGDirectSumNumbering{});
   const auto global_context = MakeWeakFormContext(
      MakeTrialField<"u">(global_space), MakeIntegrationDomain<"m">(partition));
   success = CheckForms<D, true, TestPolicy<ThreadBlockLayout<2>, 1, 4>>(
      global_context, rule, input) && success;
   success = CheckForms<D, true, TestPolicy<ThreadBlockLayout<8, 4>, 2, 4>>(
      global_context, rule, input) && success;
   return success;
}
} // namespace

int main()
{
   const bool two_dimensions = RunDimension<2>();
   const bool three_dimensions = RunDimension<3>();
   if (!two_dimensions || !three_dimensions) return 1;
#if defined(GENDIL_USE_DEVICE)
   std::cout << "Threaded device normal-cell-size regressions passed.\n";
#else
   std::cout << "Normal-cell-size fixture and CPU oracles passed; CUDA/HIP execution requires a device build.\n";
#endif
   return 0;
}

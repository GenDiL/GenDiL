// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <tuple>

using namespace gendil;

namespace
{
// Geometry is value-owned so the unequal-width cells also work on the device.
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

constexpr Real value_weight = 0.3;
constexpr std::array<Real, 2> direction{1.3, -0.7};
enum class Contribution { Gradient, Both, GradientTest };

// Independent Q1 edge assembly, with component-major DG numbering. Evaluate
// nodal basis values and physical derivatives explicitly, without DSL traces,
// field callbacks, mapping routines, or GenDiL's quadrature implementation.
Vector Expected(const Vector& input, Integer components, Real speed, Contribution contribution)
{
   Vector result(input.Size());
   result = 0.0;
   const Real* x = input.ReadHostData();
   Real* y = result.ReadWriteHostData();
   const std::array<Real, 2> points{0.5 - std::sqrt(1.0 / 12.0),
                                  0.5 + std::sqrt(1.0 / 12.0)};
   auto add_face = [&](Integer minus, Integer plus, Integer axis) {
      const std::array<Integer, 2> cells{minus, plus};
      const Real normal_speed = speed * (axis == 0 ? 1.0 : 0.5);
      const Integer selected = normal_speed >= 0.0 ? 0 : 1;
      const Real measure = axis == 0 || minus % 2 == 0 ? 1.0 : 2.0;
      for (Real point : points)
      {
         std::array<std::array<Real, 4>, 2> phi{};
         std::array<std::array<std::array<Real, 2>, 4>, 2> derivative{};
         for (Integer side = 0; side < 2; ++side)
         {
            const Real xi = axis == 0 ? 1.0 - side : point;
            const Real eta = axis == 1 ? 1.0 - side : point;
            const Real width = cells[side] % 2 == 0 ? 1.0 : 2.0;
            for (Integer j = 0; j < 2; ++j)
               for (Integer i = 0; i < 2; ++i)
               {
                  const Integer node = i + 2 * j;
                  const Real lx = i == 0 ? 1.0 - xi : xi;
                  const Real ly = j == 0 ? 1.0 - eta : eta;
                  phi[side][node] = lx * ly;
                  derivative[side][node] = {(2.0 * i - 1.0) * ly / width,
                                           lx * (2.0 * j - 1.0)};
               }
         }
         for (Integer component = 0; component < components; ++component)
         {
            Real value = 0.0;
            std::array<Real, 2> gradient{};
            const Integer selected_offset = 24 * component + 4 * cells[selected];
            for (Integer node = 0; node < 4; ++node)
            {
               value += x[selected_offset + node] * phi[selected][node];
               for (Integer d = 0; d < 2; ++d)
                  gradient[d] += x[selected_offset + node] * derivative[selected][node][d];
            }
            for (Integer side = 0; side < 2; ++side)
               for (Integer node = 0; node < 4; ++node)
               {
                  Real entry = 0.0;
                  for (Integer d = 0; d < 2; ++d)
                     entry += gradient[d] * (contribution == Contribution::GradientTest
                        ? derivative[side][node][d] : direction[d] * phi[side][node]);
                  if (contribution == Contribution::Both)
                     entry += value_weight * value * phi[side][node];
                  y[24 * component + 4 * cells[side] + node] +=
                     (side == 0 ? 1.0 : -1.0) * 0.5 * measure * normal_speed * entry;
               }
         }
      }
   };
   for (Integer row = 0; row < 3; ++row) add_face(2 * row, 2 * row + 1, 0);
   for (Integer row = 0; row < 2; ++row)
      for (Integer column = 0; column < 2; ++column)
         add_face(2 * row + column, 2 * (row + 1) + column, 1);
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

template<bool VectorField, class Policy, class Context, class Rule>
bool CheckForms(const Context& context, const Rule& rule, const Vector& input)
{
   constexpr auto shape = VectorField ? FieldShape::Vector : FieldShape::Scalar;
   constexpr Integer components = VectorField ? 2 : 1;
   TrialSpace<"u", shape> u;
   TestSpace<"v", shape> v;
   InteriorFacets<"m"> faces;
   const auto d = MakeVectorCoefficient<"direction">(
      [] GENDIL_HOST_DEVICE () { return std::array<Real, 2>{1.3, -0.7}; });
   bool success = true;
   for (const Real speed : {Real(2.0), Real(-2.0), Real(0.0)})
   {
      const auto beta = MakeVectorCoefficient<"beta">(
         [speed] GENDIL_HOST_DEVICE () { return std::array<Real, 2>{speed, 0.5 * speed}; });
      // Value tests leave the trial's transported gradient solely responsible
      // for requesting gradient interpolation and the plus-side Jacobian.
      const auto gradient_term = [&] {
         if constexpr (VectorField) return dot(upwind(beta, grad(u)) * d, jump(v));
         else return dot(upwind(beta, grad(u)), d) * jump(v);
      }();
      const auto value_term = [&] {
         if constexpr (VectorField) return value_weight * dot(upwind(beta, u), jump(v));
         else return value_weight * upwind(beta, u) * jump(v);
      }();
      const auto gradient_test_term = [&] {
         if constexpr (VectorField) return inner(upwind(beta, grad(u)), jump(grad(v)));
         else return dot(upwind(beta, grad(u)), jump(grad(v)));
      }();
      success = Check("gradient-only upwind / independent edge assembly",
         Evaluate<Policy>(integrate(faces, gradient_term), context, rule, input),
         Expected(input, components, speed, Contribution::Gradient)) && success;
      success = Check("upwind gradient test channel / independent edge assembly",
         Evaluate<Policy>(integrate(faces, gradient_test_term), context, rule, input),
         Expected(input, components, speed, Contribution::GradientTest)) && success;
      const auto combined = Evaluate<Policy>(integrate(faces, gradient_term + value_term),
         context, rule, input);
      success = Check("combined upwind / independent edge assembly", combined,
         Expected(input, components, speed, Contribution::Both)) && success;
      success = Check("split / combined upwind integrands", combined,
         Evaluate<Policy>(integrate(faces, gradient_term) + integrate(faces, value_term),
            context, rule, input)) && success;
   }
   return success;
}

template<bool VectorField>
bool Run()
{
   const UnequalMesh mesh;
   const auto scalar_fe = MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{});
   const auto fe = [&] {
      if constexpr (VectorField) return MakeVectorFiniteElement(scalar_fe, scalar_fe);
      else return scalar_fe;
   }();
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

   bool success = CheckForms<VectorField, SerialKernelConfiguration>(local_context, rule, input);
   success = CheckForms<VectorField, SerialKernelConfiguration>(global_context, rule, input) && success;
#ifdef GENDIL_USE_DEVICE
   // Six cells and seven faces exercise incomplete batches on both paths.
   using LocalPolicy = DeviceKernelConfiguration<ThreadBlockLayout<4, 4>, 4, 1>;
   using GlobalPolicy = DeviceKernelConfiguration<ThreadBlockLayout<4, 4>, 2, 4>;
   success = CheckForms<VectorField, LocalPolicy>(local_context, rule, input) && success;
   success = CheckForms<VectorField, GlobalPolicy>(global_context, rule, input) && success;
#endif
   return success;
}
} // namespace

int main()
{
   const bool scalar_ok = Run<false>();
   const bool vector_ok = Run<true>();
   if (!scalar_ok || !vector_ok) return 1;
   std::cout << "Scalar and vector upwind gradients match independent edge assembly.\n";
   return 0;
}

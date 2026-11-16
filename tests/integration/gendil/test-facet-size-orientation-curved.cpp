// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <gendil/Meshes/MeshDataStructures/UnstructuredMesh/quadmesh.hpp>
#include <gendil/Meshes/MeshDataStructures/UnstructuredMesh/hexmesh.hpp>
#include <gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/globalfacesfromlocalconnectivity.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace gendil;

namespace facet_size_orientation_curved_test
{

using KP = SerialKernelConfiguration;
using Matrix = std::vector<Real>;

bool CheckNear(const char* label, const Real actual, const Real expected,
               const Real tolerance = 2.0e-11)
{
   if (!std::isfinite(actual) || !std::isfinite(expected) ||
       std::abs(actual - expected) > tolerance * std::max(Real{1.0}, std::abs(expected)))
   {
      std::cerr << "FAILED: " << label << ": got " << actual
                << ", expected " << expected << '\n';
      return false;
   }
   return true;
}

bool CheckMatrix(const char* label, const Matrix& actual, const Matrix& expected)
{
   Real error = 0.0;
   Real scale = 1.0;
   for (size_t i = 0; i < actual.size(); ++i)
   {
      if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
         return CheckNear(label, actual[i], expected[i]);
      error = std::max(error, std::abs(actual[i] - expected[i]));
      scale = std::max(scale, std::abs(expected[i]));
   }
   return CheckNear(label, error / scale, 0.0);
}

template<class Form, class Context, class Rule>
Matrix Evaluate(const Form& form, const Context& context, const Rule& rule,
                const Integer size)
{
   const auto op = MakeGenericOperator<KP>(form, context, rule);
   Matrix matrix(size * size);
   Vector x(size), y(size);
   for (Integer column = 0; column < size; ++column)
   {
      // WriteHostData discards prior device contents: fill the whole basis vector.
      Real* basis = x.WriteHostData();
      std::fill_n(basis, size, 0.0);
      basis[column] = 1.0;
      y = 0.0;
      op(x, y);
      for (Integer row = 0; row < size; ++row)
         matrix[row * size + column] = y.ReadHostData()[row];
   }
   return matrix;
}

auto NormalSize()
{
   return MakeCoefficient<"normal_size", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real size) { return size; });
}

auto SquaredNormalSize()
{
   return MakeCoefficient<"squared_normal_size", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real size) { return size * size; });
}

template<Integer Dim>
auto BoundaryConnectivity()
{
   UnstructuredConformingConnectivity<HyperCube<Dim>> connectivity(2);
   for (Integer cell = 0; cell < 2; ++cell)
      for (Integer face = 0; face < 2 * Dim; ++face)
         connectivity[cell].faces[face] =
            {GlobalIndex{}, {}, MakeReferencePermutation<Dim>(), {}, {}, true};
   return connectivity;
}

template<class Context, class Rule, class Coefficient>
bool CheckGrouping(const Context& context, const Rule& rule, const Integer size,
                   const Coefficient& coefficient)
{
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"m"> faces;
   const auto flux = -average(dot(grad(u), Normal{})) * jump(v)
                     - jump(u) * average(dot(grad(v), Normal{}));
   const auto penalty = coefficient * jump(u) * jump(v);
   const auto combined = Evaluate(integrate(faces, flux + penalty), context, rule, size);
   const auto split = Evaluate(integrate(faces, flux) + integrate(faces, penalty),
                               context, rule, size);
   auto separate = Evaluate(integrate(faces, flux), context, rule, size);
   const auto penalty_matrix = Evaluate(integrate(faces, penalty), context, rule, size);
   Matrix transpose(size * size);
   for (Integer row = 0; row < size; ++row)
      for (Integer column = 0; column < size; ++column)
      {
         separate[row * size + column] += penalty_matrix[row * size + column];
         transpose[row * size + column] = combined[column * size + row];
      }
   bool success = CheckMatrix("combined versus split integrands", combined, split);
   success = CheckMatrix("combined versus separate operators", combined, separate) && success;
   success = CheckMatrix("SIPDG symmetry", combined, transpose) && success;
   return success;
}

bool TestRotatedNeighbor()
{
   // The right cell uses native coordinates (r,s) -> (1+2*s,3*(1-r)).
   // Its -s face meets the left cell's +x face, and its normal width is 2.
   const std::array<Real, 16> nodes{0, 0, 1, 0, 0, 3, 1, 3,
                                   1, 3, 1, 0, 3, 3, 3, 0};
   const std::array<int, 8> indices{0, 1, 2, 3, 4, 5, 6, 7};
   auto connectivity = BoundaryConnectivity<2>();
   connectivity[0].faces[2] = {1, {}, Permutation<2>{{-2, 1}}, {}, {}, false};
   connectivity[1].faces[1] = {0, {}, Permutation<2>{{2, -1}}, {}, {}, false};
   HostDevicePointer<const int> pointer;
   pointer.host_pointer = indices.data();
   QuadMesh<1> mesh{
      MakeFIFOView(nodes.data(), Integer(2), Integer(8)),
      MakeFIFOView(pointer, Integer(2), Integer(2), Integer(2)), connectivity, 2};
   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{});
   const auto space = MakeFiniteElementSpace(mesh, fe);
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space), MakeIntegrationDomain<"m">(mesh));
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<3, 3>{});
   const Integer size = space.GetNumberOfFiniteElementDofs();
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"m"> faces;
   auto matrix = [&](const auto& coefficient, const auto& ctx)
   {
      return Evaluate(integrate(faces, coefficient * jump(u) * jump(v)), ctx, rule, size);
   };
   bool success = CheckMatrix("rotated mean normal scale", matrix(average(NormalSize()), context),
                               matrix(0.75, context));
   success = CheckMatrix("rotated nonlinear mean normal scale",
      matrix(average(SquaredNormalSize()), context), matrix(0.625, context)) && success;
   success = CheckGrouping(context, rule, size, 90.0 * average(SquaredNormalSize())) && success;

   auto facets = MakeUnstructuredInteriorFaceConnectivity(connectivity, 2);
   auto partition = MakePartition(MakeCellPart(mesh),
      MakeInteriorFacePart<0, 0>(std::get<2>(facets)));
   auto mixed = MakeMixedFiniteElementSpace(partition, std::tuple{fe}, DGDirectSumNumbering{});
   const auto global_context = MakeWeakFormContext(
      MakeTrialField<"u">(mixed), MakeIntegrationDomain<"m">(partition));
   success = CheckMatrix("rotated global minus normal scale",
      matrix(minus(NormalSize()), global_context), matrix(1.0, global_context)) && success;
   success = CheckMatrix("rotated global plus normal scale",
      matrix(plus(NormalSize()), global_context), matrix(0.5, global_context)) && success;
   success = CheckMatrix("rotated local minus normal scale",
      matrix(minus(NormalSize()), context), matrix(1.0, context)) && success;
   success = CheckMatrix("rotated local plus normal scale",
      matrix(plus(NormalSize()), context), matrix(0.5, context)) && success;
   // This callback depends on reference axes, unlike the physical gradient.
   // In the canonical left-cell frame the two inverse Jacobians are diagonal:
   // diag(1,1/3) and diag(1/2,1/3), despite the neighbor's rotated native axes.
   const auto inverse = MakeCoefficient<"inverse_entries", JacobianInverse>(
      [] GENDIL_HOST_DEVICE (const auto& J) {
         return GetValue(J,0,0) + 2.0*GetValue(J,0,1)
            + 3.0*GetValue(J,1,0) + 4.0*GetValue(J,1,1);
      });
   success = CheckMatrix("rotated canonical minus inverse Jacobian",
      matrix(minus(inverse), context), matrix(7.0/3.0, global_context)) && success;
   success = CheckMatrix("rotated canonical plus inverse Jacobian",
      matrix(plus(inverse), context), matrix(11.0/6.0, global_context)) && success;
   const auto inverse_term = minus(inverse)*jump(u)*jump(v);
   const auto physical_gradient = 0.37*average(dot(grad(u),Normal{}))*jump(v);
   const auto combined_inverse = Evaluate(integrate(faces,inverse_term+physical_gradient),
      context,rule,size);
   success = CheckMatrix("rotated inverse and physical gradient local/global",
      combined_inverse,Evaluate(integrate(faces,inverse_term+physical_gradient),
         global_context,rule,size)) && success;
   success = CheckMatrix("rotated inverse and physical gradient split/combined",
      combined_inverse,Evaluate(integrate(faces,inverse_term)+integrate(faces,physical_gradient),
         context,rule,size)) && success;
   success = CheckMatrix("rotated local/global nonlinear penalty",
      matrix(average(SquaredNormalSize()), context),
      matrix(average(SquaredNormalSize()), global_context)) && success;
   success = CheckGrouping(global_context, rule, size,
                            90.0 * average(SquaredNormalSize())) && success;
   return success;
}

bool TestCurvedHexes()
{
   constexpr Real a = 0.8;
   constexpr Real b = 2.0;
   constexpr Real c = 3.0;
   std::array<Real, 48> nodes{};
   std::array<int, 16> indices{};
   auto fill_nodes = [&](const Real curvature)
   {
      for (Integer cell = 0; cell < 2; ++cell)
         for (Integer k = 0; k < 2; ++k)
            for (Integer j = 0; j < 2; ++j)
               for (Integer i = 0; i < 2; ++i)
               {
                  const Integer node = i + 2 * j + 4 * k + 8 * cell;
                  indices[node] = node;
                  nodes[3 * node] = (cell == 0 ? Real(i) : 1.0 + 2.0 * i)
                                    + curvature * j * k;
                  nodes[3 * node + 1] = b * j;
                  nodes[3 * node + 2] = c * k;
               }
   };
   fill_nodes(a);
   auto connectivity = BoundaryConnectivity<3>();
   connectivity[0].faces[3] = {1, {}, MakeReferencePermutation<3>(), {}, {}, false};
   connectivity[1].faces[0] = {0, {}, MakeReferencePermutation<3>(), {}, {}, false};
   HostDevicePointer<const int> pointer;
   pointer.host_pointer = indices.data();
   HexMesh<1> mesh{
      MakeFIFOView(static_cast<const Real*>(nodes.data()), Integer(3), Integer(16)),
      MakeFIFOView(pointer, Integer(2), Integer(2), Integer(2), Integer(2)), connectivity, 2};
   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<1, 1, 1>{});
   const auto space = MakeFiniteElementSpace(mesh, fe);
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space), MakeIntegrationDomain<"m">(mesh));
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<3, 3, 3>{});
   const Integer size = space.GetNumberOfFiniteElementDofs();
   const auto facets = MakeUnstructuredInteriorFaceConnectivity(connectivity, 2);
   const auto partition = MakePartition(MakeCellPart(mesh),
      MakeInteriorFacePart<0, 0>(std::get<3>(facets)));
   const auto mixed = MakeMixedFiniteElementSpace(partition, std::tuple{fe}, DGDirectSumNumbering{});
   const auto global_context = MakeWeakFormContext(
      MakeTrialField<"u">(mixed), MakeIntegrationDomain<"m">(partition));
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"m"> faces;
   auto matrix = [&](const auto& coefficient, const auto& ctx)
   {
      return Evaluate(integrate(faces, coefficient * jump(u) * jump(v)), ctx, rule, size);
   };
   const auto expected_normal = MakeCoefficient<"expected_normal", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const std::array<Real, 3>& x)
      {
         return 0.625 * (1.0 + a * a * (x[1] * x[1] + x[2] * x[2]) / (b * b * c * c));
      });
   const auto mean = matrix(average(SquaredNormalSize()), context);
   const auto global_mean = matrix(average(SquaredNormalSize()), global_context);
   bool success = CheckMatrix("curved hex true mean normal scales",
      mean, matrix(expected_normal, context));
   success = CheckMatrix("curved hex mean normal scales local/global",
      mean, global_mean) && success;

   // Canonical minus is cell 0, with normal width 1; plus has width 2.
   // Squaring their inverse scales gives factors 1 and 1/4, whose mean is 5/8.
   const auto minus_scale = matrix(minus(SquaredNormalSize()), context);
   const auto plus_scale = matrix(plus(SquaredNormalSize()), context);
   success = CheckMatrix("curved hex independent minus normal scale",
      minus_scale, matrix(1.6 * expected_normal, context)) && success;
   success = CheckMatrix("curved hex independent plus normal scale",
      plus_scale, matrix(0.4 * expected_normal, context)) && success;
   success = CheckMatrix("curved hex minus normal scale local/global",
      minus_scale, matrix(minus(SquaredNormalSize()), global_context)) && success;
   success = CheckMatrix("curved hex plus normal scale local/global",
      plus_scale, matrix(plus(SquaredNormalSize()), global_context)) && success;

   const auto penalty = 90.0 * average(SquaredNormalSize());
   success = CheckGrouping(context, rule, size, penalty) && success;
   success = CheckGrouping(global_context, rule, size, penalty) && success;
   const auto flux = -average(dot(grad(u), Normal{})) * jump(v)
                     - jump(u) * average(dot(grad(v), Normal{}));
   const auto combined = integrate(faces, flux + penalty * jump(u) * jump(v));
   success = CheckMatrix("curved hex grouped SIPDG local/global",
      Evaluate(combined, context, rule, size),
      Evaluate(combined, global_context, rule, size)) && success;

   return success;
}

} // namespace facet_size_orientation_curved_test

int main()
{
   using namespace facet_size_orientation_curved_test;
   const bool rotated = TestRotatedNeighbor();
   const bool curved = TestCurvedHexes();
   if (!rotated || !curved) return 1;
   std::cout << "Rotated and curved hex normal-scale regressions passed.\n";
   return 0;
}

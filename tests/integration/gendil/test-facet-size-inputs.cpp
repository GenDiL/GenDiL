// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <gendil/Meshes/MeshDataStructures/UnstructuredMesh/quadmesh.hpp>
#include <gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/globalfacesfromlocalconnectivity.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace gendil;

namespace facet_size_inputs_test
{

using KP = SerialKernelConfiguration;
using DenseMatrix = std::vector<Real>;

auto GridConnectivity(const Integer nx, const Integer ny)
{
   UnstructuredConformingConnectivity<HyperCube<2>> connectivity(nx * ny);
   for (Integer j = 0; j < ny; ++j)
   {
      for (Integer i = 0; i < nx; ++i)
      {
         const Integer cell = i + nx * j;
         for (Integer face = 0; face < 4; ++face)
         {
            connectivity[cell].faces[face] =
               {GlobalIndex{}, {}, MakeReferencePermutation<2>(), {}, {}, true};
         }
         if (i > 0) connectivity[cell].faces[0] =
            {cell - 1, {}, MakeReferencePermutation<2>(), {}, {}, false};
         if (j > 0) connectivity[cell].faces[1] =
            {cell - nx, {}, MakeReferencePermutation<2>(), {}, {}, false};
         if (i + 1 < nx) connectivity[cell].faces[2] =
            {cell + 1, {}, MakeReferencePermutation<2>(), {}, {}, false};
         if (j + 1 < ny) connectivity[cell].faces[3] =
            {cell + nx, {}, MakeReferencePermutation<2>(), {}, {}, false};
      }
   }
   return connectivity;
}

template<class Operator>
DenseMatrix Dense(const Operator& op, const Integer size)
{
   DenseMatrix result(size * size);
   Vector x(size), y(size);
   for (Integer column = 0; column < size; ++column)
   {
      // WriteHostData discards prior device contents: fill the whole basis vector.
      Real* basis = x.WriteHostData();
      std::fill_n(basis, size, 0.0);
      basis[column] = 1.0;
      y = 0.0;
      op(x, y);
      const Real* values = y.ReadHostData();
      for (Integer row = 0; row < size; ++row)
      {
         result[row * size + column] = values[row];
      }
   }
   return result;
}

bool Check(const char* label, const DenseMatrix& actual,
           const DenseMatrix& expected)
{
   Real error = 0.0;
   Real scale = 1.0;
   for (size_t i = 0; i < actual.size(); ++i)
   {
      if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
      {
         std::cerr << "FAILED: " << label << ", non-finite matrix entry\n";
         return false;
      }
      error = std::max(error, std::abs(actual[i] - expected[i]));
      scale = std::max(scale, std::abs(expected[i]));
   }
   if (error > 2.0e-11 * scale)
   {
      std::cerr << "FAILED: " << label << ", max error=" << error
                << ", scale=" << scale << '\n';
      return false;
   }
   return true;
}

bool CheckSymmetric(const char* label, const DenseMatrix& matrix,
                    const Integer size)
{
   DenseMatrix transpose(matrix.size());
   for (Integer i = 0; i < size; ++i)
   {
      for (Integer j = 0; j < size; ++j)
      {
         transpose[i * size + j] = matrix[j * size + i];
      }
   }
   return Check(label, matrix, transpose);
}

template<class Form, class Context, class Rule>
DenseMatrix Evaluate(const Form& form, const Context& context,
                     const Rule& rule, const Integer size)
{
   return Dense(MakeGenericOperator<KP>(form, context, rule), size);
}

template<class Form, class Context, class Rule>
DenseMatrix AssembleGlobal(const Form& form, const Context& context,
                           const Rule& rule, const Integer size)
{
   const auto coo = GenericAssembly<MatrixAssemblyType::RawCOO, KP>(
      form, context, rule);
   DenseMatrix result(size * size, 0.0);
   const auto* rows = ReadHost(coo.rows);
   const auto* columns = ReadHost(coo.cols);
   const auto* values = ReadHost(coo.values);
   for (size_t i = 0; i < coo.nnz_raw; ++i)
   {
      result[rows[i] * size + columns[i]] += values[i];
   }
   return result;
}

template<class Context, class Rule>
bool CheckConstantScales(const Context& context, const Rule& rule,
                         const Integer size)
{
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"m"> faces;
   const auto normal = MakeCoefficient<"normal_size", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real inverse_size) { return inverse_size; });
   const auto squared = MakeCoefficient<"squared_normal_size", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real inverse_size) { return inverse_size * inverse_size; });
   auto matrix = [&](const auto& coefficient)
   {
      return Evaluate(integrate(faces, coefficient * jump(u) * jump(v)),
                      context, rule, size);
   };

   // Every interface in these fixtures has normal cell widths 1 and 2.
   bool success = Check("mean normal scales", matrix(average(normal)), matrix(0.75));
   // Averaging a callback must evaluate it on both sides first: (1^2+.5^2)/2.
   success = Check("mean nonlinear normal-scale callback",
                   matrix(average(squared)), matrix(0.625)) && success;
   success = Check("product of distinct normal traces",
                   matrix(minus(normal) * plus(normal)), matrix(0.5)) && success;
   return success;
}

template<bool Global, class Context, class Rule, class Coefficient>
bool CheckSplitting(const Context& context, const Rule& rule,
                    const Integer size, const Coefficient& tau)
{
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"m"> faces;
   const auto consistency = -average(dot(grad(u), Normal{})) * jump(v);
   const auto adjoint = -jump(u) * average(dot(grad(v), Normal{}));
   const auto penalty = tau * jump(u) * jump(v);
   const auto combined = integrate(faces, consistency + adjoint + penalty);
   const auto split = integrate(faces, consistency)
                    + integrate(faces, adjoint)
                    + integrate(faces, penalty);
   const auto combined_matrix = Evaluate(combined, context, rule, size);
   bool success = Check("combined and split integrals",
                        combined_matrix, Evaluate(split, context, rule, size));

   auto separate = Evaluate(integrate(faces, consistency), context, rule, size);
   const auto adjoint_matrix = Evaluate(integrate(faces, adjoint), context, rule, size);
   const auto penalty_matrix = Evaluate(integrate(faces, penalty), context, rule, size);
   for (size_t i = 0; i < separate.size(); ++i)
   {
      separate[i] += adjoint_matrix[i] + penalty_matrix[i];
   }
   success = Check("combined and independently constructed operators",
                   combined_matrix, separate) && success;
   success = CheckSymmetric("symmetric interior SIPDG", combined_matrix, size) && success;
   success = CheckSymmetric("symmetric penalty alone", penalty_matrix, size) && success;
   if constexpr (Global)
   {
      success = Check("global assembly of combined integral", combined_matrix,
                      AssembleGlobal(combined, context, rule, size)) && success;
      success = Check("global assembly of split integrals", combined_matrix,
                      AssembleGlobal(split, context, rule, size)) && success;
   }
   else
   {
      const auto assembled = GenericAssembly<MatrixAssemblyType::BSR, KP>(
         combined, context, rule);
      success = Check("local assembly of combined integral", combined_matrix,
                      Dense(assembled, size)) && success;
      const auto assembled_split = GenericAssembly<MatrixAssemblyType::BSR, KP>(
         split, context, rule);
      success = Check("local assembly of split integrals", combined_matrix,
                      Dense(assembled_split, size)) && success;
   }
   return success;
}

bool TestUnequalRectangles()
{
   const std::array<Real, 12> vertices{0, 0, 1, 0, 3, 0, 0, 3, 1, 3, 3, 3};
   const std::array<int, 8> cells{0, 1, 3, 4, 1, 2, 4, 5};
   auto connectivity = GridConnectivity(2, 1);
   HostDevicePointer<const int> cells_pointer;
   cells_pointer.host_pointer = cells.data();
   QuadMesh<1> mesh{
      MakeFIFOView(vertices.data(), Integer(2), Integer(6)),
      MakeFIFOView(cells_pointer, Integer(2), Integer(2), Integer(2)),
      connectivity, Integer(2)};
   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   const auto space = MakeFiniteElementSpace(mesh, fe);
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{});
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space), MakeIntegrationDomain<"m">(mesh));
   const auto squared_normal = MakeCoefficient<"squared_normal_size", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real inverse_size) { return inverse_size * inverse_size; });
   const Integer size = space.GetNumberOfFiniteElementDofs();
   bool success = CheckConstantScales(context, rule, size);
   success = CheckSplitting<false>(context, rule, size,
                                  90.0 * average(squared_normal)) && success;

   auto facets = MakeUnstructuredInteriorFaceConnectivity(connectivity, 2);
   auto partition = MakePartition(MakeCellPart(mesh),
      MakeInteriorFacePart<0, 0>(std::get<2>(facets)));
   auto mixed = MakeMixedFiniteElementSpace(partition, std::tuple{fe},
                                            DGDirectSumNumbering{});
   const auto global_context = MakeWeakFormContext(
      MakeTrialField<"u">(mixed), MakeIntegrationDomain<"m">(partition));
   success = CheckConstantScales(global_context, rule, size) && success;
   success = CheckSplitting<true>(global_context, rule, size,
                                 90.0 * average(squared_normal)) && success;
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"m"> faces;
   const auto form = integrate(faces,
      average(squared_normal) * jump(u) * jump(v));
   success = Check("local and global penalties",
      Evaluate(form, context, rule, size),
      Evaluate(form, global_context, rule, size)) && success;
   return success;
}

bool TestNonconformingInterface()
{
   // Tangential refinement divides the coarse edge; normal widths remain 1 and 2.
   Cartesian2DMesh coarse(1.0, 3.0, 1, 1, Point<2>{0.0, 0.0});
   Cartesian2DMesh fine(2.0, 1.5, 1, 2, Point<2>{1.0, 0.0});
   NonconformingCartesianIntermeshFaceConnectivity<2, 2> faces({1, 1}, {1, 2});
   auto partition = MakePartition(MakeCellPart(coarse), MakeCellPart(fine),
                                  MakeInteriorFacePart<0, 1>(faces));
   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   auto space = MakeMixedFiniteElementSpace(partition, std::tuple{fe, fe},
                                            DGDirectSumNumbering{});
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space), MakeIntegrationDomain<"m">(partition));
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{});
   const auto squared_normal = MakeCoefficient<"squared_normal_size", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real inverse_size) { return inverse_size * inverse_size; });
   const Integer size = GetNumberOfGlobalDofs(space);
   bool success = CheckConstantScales(context, rule, size);
   success = CheckSplitting<true>(context, rule, size,
                                 90.0 * average(squared_normal)) && success;
   return success;
}

bool TestWarpedQuads()
{
   constexpr Integer n = 3;
   std::vector<Real> nodes(2 * (n + 1) * (n + 1));
   for (Integer j = 0; j <= n; ++j)
   {
      for (Integer i = 0; i <= n; ++i)
      {
         const Real x = Real(i) / n;
         const Real y = Real(j) / n;
         nodes[2 * (i + (n + 1) * j)] =
            x + 0.1 * std::sin(M_PI * x) * std::sin(M_PI * y);
         nodes[2 * (i + (n + 1) * j) + 1] = y;
      }
   }
   std::vector<int> indices(4 * n * n);
   for (Integer j = 0; j < n; ++j)
   {
      for (Integer i = 0; i < n; ++i)
      {
         for (Integer b = 0; b < 2; ++b)
         {
            for (Integer a = 0; a < 2; ++a)
            {
               const Integer cell = i + n * j;
               const Integer corner = a + 2 * b;
               const Integer node = (i + a) + (n + 1) * (j + b);
               indices[4 * cell + corner] = node;
            }
         }
      }
   }
   auto connectivity = GridConnectivity(n, n);
   HostDevicePointer<const int> indices_pointer;
   indices_pointer.host_pointer = indices.data();
   QuadMesh<1> mesh{
      MakeFIFOView(static_cast<const Real*>(nodes.data()), Integer(2), Integer((n + 1) * (n + 1))),
      MakeFIFOView(indices_pointer, Integer(2), Integer(2), Integer(n * n)),
      connectivity, Integer(n * n)};
   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   const auto space = MakeFiniteElementSpace(mesh, fe);
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space), MakeIntegrationDomain<"m">(mesh));
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{});
   const auto normal_size = MakeCoefficient<"normal_size", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real inverse_size) { return inverse_size; });
   const Integer size = space.GetNumberOfFiniteElementDofs();
   return CheckSplitting<false>(context, rule, size, 90.0 * average(normal_size));
}

bool TestCurvedEdge()
{
   // Both Q2 cells share x=1+0.8*t*(1-t), y=3*t. Their normal
   // scales vary pointwise, with a factor of two between adjacent cells.
   std::array<Real, 36> nodes{};
   std::array<int, 18> indices{};
   for (Integer cell = 0; cell < 2; ++cell)
   {
      for (Integer j = 0; j < 3; ++j)
      {
         for (Integer i = 0; i < 3; ++i)
         {
            const Real s = Real(i) / 2.0;
            const Real t = Real(j) / 2.0;
            const Integer node = i + 3 * j + 9 * cell;
            indices[node] = node;
            nodes[2 * node] = (cell == 0 ? s : 1.0 + 2.0 * s)
                           + 0.8 * t * (1.0 - t);
            nodes[2 * node + 1] = 3.0 * t;
         }
      }
   }
   auto connectivity = GridConnectivity(2, 1);
   HostDevicePointer<const int> indices_pointer;
   indices_pointer.host_pointer = indices.data();
   QuadMesh<2> mesh{
      MakeFIFOView(static_cast<const Real*>(nodes.data()), Integer(2), Integer(18)),
      MakeFIFOView(indices_pointer, Integer(3), Integer(3), Integer(2)),
      connectivity, Integer(2)};
   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   const auto space = MakeFiniteElementSpace(mesh, fe);
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space), MakeIntegrationDomain<"m">(mesh));
   const auto squared_normal = MakeCoefficient<"squared_normal_size", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real inverse_size) { return inverse_size * inverse_size; });
   const Integer size = space.GetNumberOfFiniteElementDofs();
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"m"> faces;
   const auto actual = integrate(faces, average(squared_normal) * jump(u) * jump(v));
   const auto initial_scale = MakeCoefficient<"expected_normal", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const std::array<Real, 2>& x)
      {
         const Real slope = 0.8 * (1.0 - 2.0 * x[1] / 3.0) / 3.0;
         // Mean of rho^2 for normal reference widths 1 and 2.
         return 0.625 * (1.0 + slope * slope);
      });
   const auto updated_scale = MakeCoefficient<"expected_normal", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const std::array<Real, 2>& x)
      {
         const Real slope = 1.2 * (1.0 - 2.0 * x[1] / 3.0) / 3.0;
         return 0.625 * (1.0 + slope * slope);
      });
   const auto expected_form = integrate(faces, initial_scale * jump(u) * jump(v));
   const auto updated_form = integrate(faces, updated_scale * jump(u) * jump(v));
   const auto rule4 = MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{});
   const auto rule6 = MakeIntegrationRule(IntegrationRuleNumPoints<6, 6>{});
   bool success = Check("curved normal scales with four-point operator rule",
      Evaluate(actual, context, rule4, size), Evaluate(expected_form, context, rule4, size));
   success = Check("curved normal scales with six-point operator rule",
      Evaluate(actual, context, rule6, size), Evaluate(expected_form, context, rule6, size)) && success;
   success = CheckSplitting<false>(context, rule4, size,
                                  90.0 * average(squared_normal)) && success;

   // An existing operator must observe changes in the mesh node storage.
   const auto existing_operator = MakeGenericOperator<KP>(actual, context, rule4);
   for (Integer cell = 0; cell < 2; ++cell)
   {
      for (Integer j = 0; j < 3; ++j)
      {
         for (Integer i = 0; i < 3; ++i)
         {
            const Real t = Real(j) / 2.0;
            nodes[2 * (i + 3 * j + 9 * cell)] += 0.4 * t * (1.0 - t);
         }
      }
   }
   success = Check("normal scales after mesh coordinates change",
      Dense(existing_operator, size),
      Evaluate(updated_form, context, rule4, size)) && success;
   return success;
}

bool TestThreeDimensionalScales()
{
   CartesianMesh<3> mesh({2, 1, 1}, {1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}, false);
   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<1, 1, 1>{});
   const auto space = MakeFiniteElementSpace(mesh, fe);
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space), MakeIntegrationDomain<"m">(mesh));
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<3, 3, 3>{});
   const auto normal_size = MakeCoefficient<"normal_size", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real inverse_size) { return inverse_size; });
   const Integer size = space.GetNumberOfFiniteElementDofs();
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"m"> faces;
   BoundaryFacets<"m"> boundary;
   const auto normal = integrate(faces, average(normal_size) * jump(u) * jump(v));
   const auto expected_normal = integrate(faces, jump(u) * jump(v));
   bool success = Check("three-dimensional normal cell width",
      Evaluate(normal, context, rule, size), Evaluate(expected_normal, context, rule, size));

   const auto boundary_normal = MakeCoefficient<"boundary_normal", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const std::array<Real, 3>& x)
      {
         if (x[0] < 1.0e-12 || x[0] > 2.0 - 1.0e-12) return 1.0;
         if (x[1] < 1.0e-12 || x[1] > 2.0 - 1.0e-12) return 0.5;
         return 1.0 / 3.0;
      });
   success = Check("boundary normal scale without a side selector",
      Evaluate(integrate(boundary, normal_size * u * v), context, rule, size),
      Evaluate(integrate(boundary, boundary_normal * u * v), context, rule, size)) && success;
   return success;
}

} // namespace facet_size_inputs_test

int main()
{
   using namespace facet_size_inputs_test;
   bool success = TestUnequalRectangles();
   success = TestNonconformingInterface() && success;
   success = TestWarpedQuads() && success;
   success = TestCurvedEdge() && success;
   success = TestThreeDimensionalScales() && success;
   if (!success) return 1;
   std::cout << "Side-specific normal sizes and SIPDG splitting passed.\n";
   return 0;
}

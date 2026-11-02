// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{
using Policy = SerialKernelConfiguration;

template<size_t N>
bool Check(const Vector& result, const std::array<Real, N>& expected, const char* label)
{
   const auto* values = result.ReadHostData();
   for (size_t i = 0; i < N; ++i)
   {
      if (!std::isfinite(values[i]) || std::abs(values[i] - expected[i]) > 1e-12)
      {
         std::cerr << label << ": row " << i << " is " << values[i]
                   << ", expected " << expected[i] << '\n';
         return false;
      }
   }
   return true;
}

bool CheckOneDimensionalBoundary()
{
   CartesianMesh<1> mesh({1}, {.25}, {0.}, false);
   auto space = MakeFiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<1>{}));
   auto context = MakeWeakFormContext(MakeTrialField<"u">(space),
      MakeIntegrationDomain<"mesh">(mesh));
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   auto scale = MakeCoefficient<"scale", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real value) { return value; });
   auto op = MakeWeakFormResidualOperator<Policy>(
      integrate(BoundaryFacets<"mesh">{}, scale*u*v), context,
      MakeIntegrationRule(IntegrationRuleNumPoints<3>{}));
   Vector x(2), y(2);
   x = 1.0;
   op(x, y);
   // Each endpoint has unit point measure and inverse normal cell size 4.
   return Check(y, std::array<Real, 2>{4., 4.}, "1D boundary normal size");
}

bool CheckOneDimensionalInterior()
{
   CartesianMesh<1> left({1}, {.4}, {0.}, false);
   CartesianMesh<1> right({1}, {.6}, {.4}, false);
   auto partition = MakePartition(MakeCellPart(left), MakeCellPart(right),
      MakeInteriorFacePart<0, 1>(CartesianIntermeshFaceConnectivity<1, 1>({1}, {1})));
   auto fe = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   auto space = MakeMixedFiniteElementSpace(partition, std::tuple{fe, fe}, DGDirectSumNumbering{});
   auto context = MakeWeakFormContext(MakeTrialField<"u">(space),
      MakeIntegrationDomain<"mesh">(partition));
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   auto scale = MakeCoefficient<"scale", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real value) { return value; });
   auto op = MakeWeakFormResidualOperator<Policy>(
      integrate(InteriorFacets<"mesh">{}, average(scale)*jump(u)*jump(v)), context,
      MakeIntegrationRule(IntegrationRuleNumPoints<3>{}));
   Vector x(4), y(4);
   auto* values = x.WriteHostData();
   values[0] = 0.; values[1] = 1.; values[2] = 3.; values[3] = 0.;
   op(x, y);
   // The interface jump is -2 and mean(1/.4, 1/.6) = 25/12.
   return Check(y, std::array<Real, 4>{0., -25./6., 25./6., 0.},
      "1D interior mean normal size");
}

bool CheckAnisotropicBoundary()
{
   CartesianMesh<2> mesh({1, 1}, {2., 3.}, {0., 0.}, false);
   auto space = MakeFiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{}));
   auto context = MakeWeakFormContext(MakeTrialField<"u">(space),
      MakeIntegrationDomain<"mesh">(mesh));
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   BoundaryFacets<"mesh"> boundary;
   auto normal = MakeCoefficient<"normal", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real value) { return value; });
   auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<3, 3>{});
   auto normal_op = MakeWeakFormResidualOperator<Policy>(integrate(boundary, normal*u*v), context, rule);
   Vector x(4), normal_y(4);
   x = 1.0;
   normal_op(x, normal_y);
   // Each vertex receives half of each adjacent edge's scaled measure.
   // Normal: (2/3 + 3/2)/2 = 13/12.
   return Check(normal_y, std::array<Real, 4>{13./12., 13./12., 13./12., 13./12.},
         "2D boundary normal size");
}
} // namespace

int main()
{
   bool ok = CheckOneDimensionalBoundary();
   ok = CheckOneDimensionalInterior() && ok;
   ok = CheckAnisotropicBoundary() && ok;
   return ok ? 0 : 1;
}

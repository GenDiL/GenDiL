// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>

#include <array>
#include <tuple>

#ifndef GENDIL_TRANSPOSE_REJECT_CASE
#define GENDIL_TRANSPOSE_REJECT_CASE 0
#endif

namespace
{
using namespace gendil;

// Returning a valid scalar for every supported input isolates the admission
// diagnostic from the callback's invocation representation.
struct TraversalCoefficient
{
   template<class T>
   Real operator()(const T& value) const
   {
      if constexpr (std::is_arithmetic_v<T>) return 1+Real(value);
      else return 1+GetValue(value, 0);
   }
};

template<CoefficientInput Input, bool Transposed, bool Global = false, bool Traced = false,
         bool RoundTrip = false>
void ConstructFacet()
{
   CartesianMesh<1> mesh({2}, {1.}, {0.}, false);
   const auto fe = MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<2>{});
   const auto coefficient = MakeCoefficient<"traversal", Input>(TraversalCoefficient{});
   const auto source = [&]
   {
      if constexpr (Traced)
         return integrate(InteriorFacets<"mesh">{}, minus(coefficient)*minus(TrialSpace<"u">{})*plus(TestSpace<"v">{}));
      else
         return integrate(InteriorFacets<"mesh">{}, coefficient*minus(TrialSpace<"u">{})*plus(TestSpace<"v">{}));
   }();
   const auto form = [&]
   {
      if constexpr (RoundTrip) return transpose(transpose(source));
      else if constexpr (Transposed) return transpose(source);
      else return source;
   }();
   if constexpr (Global)
   {
      const auto faces = MakeCartesianInteriorFaceConnectivity<1>({2});
      const auto partition = MakePartition(MakeCellPart(mesh), MakeInteriorFacePart<0, 0>(faces));
      const auto space = MakeMixedFiniteElementSpace(partition, std::tuple{fe}, DGDirectSumNumbering{});
      const auto context = MakeWeakFormContext(MakeTrialField<"u">(space),
         MakeTestField<"v">(space), MakeIntegrationDomain<"mesh">(partition));
      const auto op = MakeGenericOperator<SerialKernelConfiguration>(form, context, rule);
      (void)op;
   }
   else
   {
      const auto space = MakeFiniteElementSpace(mesh, fe);
      const auto context = MakeWeakFormContext(MakeTrialField<"u">(space),
         MakeTestField<"v">(space), MakeIntegrationDomain<"mesh">(mesh));
      const auto op = MakeGenericOperator<SerialKernelConfiguration>(form, context, rule);
      (void)op;
   }
}
}

int main()
{
   [[maybe_unused]] const Cells<"mesh"> cells;
   [[maybe_unused]] const TrialSpace<"u"> u;
   [[maybe_unused]] const TrialSpace<"p"> p;
   [[maybe_unused]] const TestSpace<"v"> v;
#if GENDIL_TRANSPOSE_REJECT_CASE == 0
   ConstructFacet<ElementIndex, false>();
   ConstructFacet<ElementIndex, false, false, true>();
   ConstructFacet<QuadraturePointIndex, false>();
   ConstructFacet<ReferenceCoordinates, false>();
   ConstructFacet<PhysicalCoordinates, true>();
   ConstructFacet<ElementIndex, true, true>();
   ConstructFacet<QuadraturePointIndex, true, true>();
   ConstructFacet<ReferenceCoordinates, true, true>();
   ConstructFacet<ElementIndex, true, false, false, true>();
   ConstructFacet<QuadraturePointIndex, true, false, false, true>();
   ConstructFacet<ReferenceCoordinates, true, false, false, true>();
#elif GENDIL_TRANSPOSE_REJECT_CASE == 1
   (void)transpose(integrate(cells, v));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 2
   (void)transpose(rhs(integrate(cells, (u+ScaleExpr{Real{1}})*v)));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 3
   (void)transpose(integrate(cells, (u+ScaleExpr{Real{1}})*v));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 4
   (void)transpose(integrate(cells, u*p*v));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 5
   const auto opaque = MakeCoefficient<"opaque", FieldValue<"u">>([](Real value) { return value; });
   // The declaration appears later: eligibility must use the complete form.
   (void)transpose(integrate(cells, opaque*v)+integrate(cells, u*v));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 6
   const auto beta = MakeVectorCoefficient<"beta", FieldValue<"p">>(
      [](Real value) { return std::array<Real, 1>{value}; });
   (void)transpose(integrate(InteriorFacets<"mesh">{}, upwind(average(beta), u)*jump(v))
      + integrate(cells, p*v));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 7
   const auto prescribed_v = MakeCoefficient<"prescribed_v", FieldValue<"v">>([](Real value) { return 1+value; });
   (void)transpose(integrate(cells, prescribed_v*u*v));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 8
   (void)transpose(integrate(cells, u*v*v));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 9
   (void)transpose(rhs(integrate(cells, u*v)));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 10
   ConstructFacet<ElementIndex, true>();
#elif GENDIL_TRANSPOSE_REJECT_CASE == 11
   ConstructFacet<ElementIndex, true, false, true>();
#elif GENDIL_TRANSPOSE_REJECT_CASE == 12
   ConstructFacet<QuadraturePointIndex, true>();
#elif GENDIL_TRANSPOSE_REJECT_CASE == 13
   ConstructFacet<ReferenceCoordinates, true>();
#elif GENDIL_TRANSPOSE_REJECT_CASE == 14
   (void)transpose(integrate(cells, Real{0}*(u*p*v)));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 15
   const auto load = MakeCoefficient<"load">([] { return Real{2}; });
   (void)transpose(integrate(cells, u*v)+integrate(cells, load*v)+integrate(cells, -load*v));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 16
   (void)transpose(integrate(cells, FiniteElementField<"v">{}*u*v));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 17
   (void)transpose(lhs(rhs(integrate(cells, u*v))));
#elif GENDIL_TRANSPOSE_REJECT_CASE == 18
   CartesianMesh<1> mesh({1}, {1.}, {0.}, false);
   const auto fe = MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   const auto scalar = MakeFiniteElementSpace(mesh, fe);
   const auto vector = MakeFiniteElementSpace(mesh, MakeVectorFiniteElement(fe, fe));
   Vector values(2);
   values = 1;
   const auto context = MakeWeakFormContext(MakeTrialField<"u">(scalar),
      MakeTestField<"v">(scalar), MakeFiniteElementField<"f">(vector, values),
      MakeIntegrationDomain<"mesh">(mesh));
   // Deferred multiplication may dispatch as test-linear, but binding f to a
   // vector must still reject this integrand's required scalar result.
   const auto form = transpose(integrate(cells, -(2.5*(FiniteElementField<"f">{}*u))*v));
   const auto op = MakeGenericOperator<SerialKernelConfiguration>(form, context,
      MakeIntegrationRule(IntegrationRuleNumPoints<2>{}));
   (void)op;
#else
#error Unknown form-transpose rejection case
#endif
}

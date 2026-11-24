// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>

#include <tuple>
#include <type_traits>

#ifndef GENDIL_SHAPED_REJECT_CASE
#define GENDIL_SHAPED_REJECT_CASE 0
#endif

namespace
{
using namespace gendil;

// Accepting any rank cannot discharge a field-shape declaration, including
// the scalar declaration provided by a one-argument input tag.
// Every case only constructs the operator: no operator() or getter is called.
struct AcceptAny
{
   template<class T>
   GENDIL_HOST_DEVICE
   Real operator()(const T&) const { return 1; }
};

template<bool Gradient, bool VectorField, bool Mismatch, bool DefaultScalar = false>
using Input = std::conditional_t<DefaultScalar,
   std::conditional_t<Gradient, FieldGradient<"w">, FieldValue<"w">>,
   std::conditional_t<Gradient,
      FieldGradient<"w", VectorField != Mismatch ? FieldShape::Vector : FieldShape::Scalar>,
      FieldValue<"w", VectorField != Mismatch ? FieldShape::Vector : FieldShape::Scalar>>>;

template<bool VectorField, bool Gradient, bool Active, bool Mismatch, bool DefaultScalar = false>
void CellConstructor()
{
   CartesianMesh<2> mesh({1, 1}, {2., 3.}, {0., 0.}, false);
   const auto scalar_fe = MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{});
   const auto vector_fe = MakeVectorFiniteElement(scalar_fe, scalar_fe, scalar_fe);
   const auto scalar_space = MakeFiniteElementSpace(mesh, scalar_fe);
   const auto field_space = [&] {
      if constexpr (VectorField) return MakeFiniteElementSpace(mesh, vector_fe);
      else return scalar_space;
   }();
   Vector prescribed(64);
   prescribed = 0;
   const auto field = [&] {
      if constexpr (Active) return MakeTrialField<"w">(field_space);
      else return MakeFiniteElementField<"w">(field_space, prescribed);
   }();
   const auto context = MakeWeakFormContext(field, MakeTestField<"v">(scalar_space),
      MakeIntegrationDomain<"mesh">(mesh));
   const auto coefficient = MakeCoefficient<"c", Input<Gradient, VectorField, Mismatch, DefaultScalar>>(AcceptAny{});
   const TestSpace<"v"> v;
   const auto form = [&] {
      if constexpr (!Active) return integrate(Cells<"mesh">{}, coefficient * v);
      else if constexpr (VectorField)
      {
         const VectorTrialSpace<"w"> w;
         return integrate(Cells<"mesh">{}, coefficient * dot(w, w) * v);
      }
      else return integrate(Cells<"mesh">{}, coefficient * TrialSpace<"w">{} * v);
   }();
   const auto op = MakeGenericOperator<SerialKernelConfiguration>(form, context,
      MakeIntegrationRule(IntegrationRuleNumPoints<3, 3>{}));
   (void)op;
}

template<bool PlusIsVector, bool Gradient, bool Mismatch, bool DefaultScalar = false>
void PlusConstructor()
{
   CartesianMesh<1> left({1}, {.4}, {0.}, false);
   CartesianMesh<1> right({1}, {.6}, {.4}, false);
   const auto partition = MakePartition(MakeCellPart(left), MakeCellPart(right),
      MakeInteriorFacePart<0, 1>(CartesianIntermeshFaceConnectivity<1, 1>({1}, {1})));
   const auto scalar_fe = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   const auto vector_fe = MakeVectorFiniteElement(scalar_fe, scalar_fe, scalar_fe);
   const auto field_space = [&] {
      if constexpr (PlusIsVector)
         return MakeMixedFiniteElementSpace(partition, std::tuple{scalar_fe, vector_fe}, DGDirectSumNumbering{});
      else
         return MakeMixedFiniteElementSpace(partition, std::tuple{vector_fe, scalar_fe}, DGDirectSumNumbering{});
   }();
   const auto test_space = MakeMixedFiniteElementSpace(partition,
      std::tuple{scalar_fe, scalar_fe}, DGDirectSumNumbering{});
   Vector prescribed(8);
   prescribed = 0;
   const auto context = MakeWeakFormContext(MakeFiniteElementField<"w">(field_space, prescribed),
      MakeTestField<"v">(test_space), MakeIntegrationDomain<"mesh">(partition));
   const auto coefficient = MakeCoefficient<"c", Input<Gradient, PlusIsVector, Mismatch, DefaultScalar>>(AcceptAny{});
   // Only plus reads w. The unused minus space has the opposite rank, making
   // validating the wrong endpoint detectable by both the pass and fail cases.
   const auto form = integrate(InteriorFacets<"mesh">{}, plus(coefficient) * minus(TestSpace<"v">{}));
   const auto op = MakeGenericOperator<SerialKernelConfiguration>(form, context,
      MakeIntegrationRule(IntegrationRuleNumPoints<3>{}));
   (void)op;
}

template<bool VectorField, bool Gradient, bool Active>
void CellPositiveControls()
{
   CellConstructor<VectorField, Gradient, Active, false>();
   if constexpr (!VectorField)
      CellConstructor<VectorField, Gradient, Active, false, true>();
}
}

int main()
{
#if GENDIL_SHAPED_REJECT_CASE == 0
   CellPositiveControls<false, false, false>();
   CellPositiveControls<false, true, false>();
   CellPositiveControls<true, false, false>();
   CellPositiveControls<true, true, false>();
   CellPositiveControls<false, false, true>();
   CellPositiveControls<false, true, true>();
   CellPositiveControls<true, false, true>();
   CellPositiveControls<true, true, true>();
   PlusConstructor<false, false, false>();
   PlusConstructor<false, true, false>();
   PlusConstructor<true, false, false>();
   PlusConstructor<true, true, false>();
   PlusConstructor<false, false, false, true>();
   PlusConstructor<false, true, false, true>();
#elif GENDIL_SHAPED_REJECT_CASE <= 8
   // Bits select actual vector/scalar space, gradient/value input and role.
   constexpr int mode = GENDIL_SHAPED_REJECT_CASE - 1;
   CellConstructor<(mode & 1) != 0, (mode & 2) != 0, (mode & 4) != 0, true>();
#elif GENDIL_SHAPED_REJECT_CASE <= 12
   constexpr int mode = GENDIL_SHAPED_REJECT_CASE - 9;
   PlusConstructor<(mode & 1) != 0, (mode & 2) != 0, true>();
#elif GENDIL_SHAPED_REJECT_CASE <= 16
   // Default scalar tags must reject vector fields for both input kinds and
   // both active and prescribed resolution, even with a generic callback.
   constexpr int mode = GENDIL_SHAPED_REJECT_CASE - 13;
   CellConstructor<true, (mode & 1) != 0, (mode & 2) != 0, true, true>();
#elif GENDIL_SHAPED_REJECT_CASE <= 18
   constexpr int mode = GENDIL_SHAPED_REJECT_CASE - 17;
   PlusConstructor<true, mode != 0, true, true>();
#else
#error "Unknown shaped coefficient rejection case"
#endif
}

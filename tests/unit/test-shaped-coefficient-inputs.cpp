// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>

#include <type_traits>

namespace
{
using namespace gendil;

struct AcceptAny
{
   template<class... Args>
   GENDIL_HOST_DEVICE
   Real operator()(const Args&...) const { return 1; }
};

using ScalarValue = FieldValue<"w", FieldShape::Scalar>;
using ScalarGradient = FieldGradient<"w", FieldShape::Scalar>;
using VectorValue = VectorFieldValue<"w">;
using VectorGradient = VectorFieldGradient<"w">;
static_assert(std::is_same_v<VectorValue, FieldValue<"w", FieldShape::Vector>>);
static_assert(std::is_same_v<VectorGradient, FieldGradient<"w", FieldShape::Vector>>);
static_assert(std::is_same_v<FieldValue<"w">, ScalarValue>);
static_assert(std::is_same_v<FieldGradient<"w">, ScalarGradient>);
static_assert(ScalarValue::field_shape == FieldShape::Scalar && ScalarValue::argument_rank == 0);
static_assert(ScalarGradient::field_shape == FieldShape::Scalar && ScalarGradient::argument_rank == 1);
static_assert(VectorValue::field_shape == FieldShape::Vector && VectorValue::argument_rank == 1);
static_assert(VectorGradient::field_shape == FieldShape::Vector && VectorGradient::argument_rank == 2);

template<class Input, OperatorMask Mask>
void CheckRequirements()
{
   using Tag = std::remove_cvref_t<Input>;
   const auto coefficient = MakeCoefficient<"c", Input>(AcceptAny{});
   using CoefficientType = decltype(coefficient);
   using Requirements = coefficient_named_field_requirements_t<CoefficientType>;
   using Read = find_named_field_requirement_t<Requirements, "w">;
   static_assert(need_values(Read::mask) == need_values(Mask));
   static_assert(need_gradients(Read::mask) == need_gradients(Mask));
   static_assert(has_provenance(Read::provenance, NamedFieldProvenance::CoefficientInput));

   using Bound = expression_semantics_t<CoefficientType, form_symbols_t<CoefficientType>>;
   static_assert(is_empty_type_list_v<typename Bound::active_inputs>);
   static_assert(contains_named_field_requirement_v<typename Bound::bound_inputs, "w">);
   static_assert(Bound::active_affinity == ActiveAffinity::Independent);
   // A scalar callback result is independent of the requested argument's rank.
   static_assert(Bound::symbolic_shape == SemanticShape::Scalar);
   static_assert(has_geometry(Bound::geometry, GeometryRequirement::JacobianInverse) == need_gradients(Mask));

   constexpr FieldShape shape = Tag::field_shape;
   const TrialSpace<"w", shape> w;
   const TestSpace<"v", shape> v;
   const auto form = [&] {
      if constexpr (shape == FieldShape::Vector)
         return integrate(Cells<"mesh">{}, coefficient * dot(w, v));
      else
         return integrate(Cells<"mesh">{}, coefficient * w * v);
   }();
   using Active = form_signature_t<decltype(form)>;
   static_assert(Active::valid);
   static_assert(contains_named_field_requirement_v<typename Active::active_inputs, "w">);
   static_assert(is_empty_type_list_v<typename Active::bound_inputs>);
   using ActiveRead = find_named_field_requirement_t<typename Active::active_inputs, "w">;
   static_assert(need_values(ActiveRead::mask));
   static_assert(need_gradients(ActiveRead::mask) == need_gradients(Mask));
   static_assert(has_provenance(ActiveRead::provenance, NamedFieldProvenance::CoefficientInput));
   static_assert(Active::active_affinity == ActiveAffinity::Unknown);

   const auto interior = integrate(InteriorFacets<"mesh">{},
      plus(coefficient) * minus(TestSpace<"v">{}));
   using Side = form_signature_t<decltype(interior)>;
   static_assert(Side::valid);
   static_assert(contains_named_field_requirement_v<
      typename Side::template bound_inputs_on<SemanticSide::Plus>, "w">);
   static_assert(is_empty_type_list_v<typename Side::template bound_inputs_on<SemanticSide::Minus>>);
   static_assert(has_geometry(Side::template geometry_on<SemanticSide::Plus>,
      GeometryRequirement::JacobianInverse) == need_gradients(Mask));
   static_assert(!has_geometry(Side::template geometry_on<SemanticSide::Minus>,
      GeometryRequirement::JacobianInverse));
   static_assert(requires_plus_side_jacobian_v<decltype(plus(coefficient))> == need_gradients(Mask));
   static_assert(!requires_plus_side_jacobian_v<decltype(minus(coefficient))>);
}
}

int main()
{
   CheckRequirements<FieldValue<"w">, OperatorMask::Values>();
   CheckRequirements<FieldGradient<"w">, OperatorMask::Gradients>();
   CheckRequirements<ScalarValue, OperatorMask::Values>();
   CheckRequirements<ScalarGradient, OperatorMask::Gradients>();
   CheckRequirements<VectorValue, OperatorMask::Values>();
   CheckRequirements<VectorGradient, OperatorMask::Gradients>();
   CheckRequirements<const VectorValue&, OperatorMask::Values>();
   CheckRequirements<const VectorGradient&, OperatorMask::Gradients>();
}

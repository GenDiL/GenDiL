// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <tuple>
#include <type_traits>

namespace
{
using namespace gendil;

template<class Form, StaticString Input, StaticString Output>
constexpr void CheckSinglePair()
{
   using Signature = form_signature_t<Form>;
   static_assert(Signature::valid);
   static_assert(Signature::result_kind == FormResultKind::Residual);
   static_assert(weak_form_traits::list_size<typename Signature::callable_inputs>::value == 1);
   static_assert(weak_form_traits::list_size<typename Signature::logical_outputs>::value == 1);
   static_assert(contains_named_field_requirement_v<typename Signature::callable_inputs, Input>);
   static_assert(contains_named_field_requirement_v<typename Signature::logical_outputs, Output>);
}

// Construct a typed zero with an explicit callable contract. rhs deliberately
// does not supply such a contract, even when it retains trial declarations.
template<FormExpr Form, DomainExpr Domain>
auto ZeroWithContract(const Form&, const Domain& domain)
{
   using Signature = form_signature_t<Form>;
   using Original = typename form_symbols_t<Form>::plain_declarations;
   using Declarations = weak_form_traits::symbol_summary<Original::supported,
      typename Original::active, typename Original::tests, typename Original::bound,
      typename Signature::callable_input_relations>;
   using Zero = ZeroExpr<Original, typename Signature::logical_output_sides>;
   auto term = integrate(domain, Zero{});
   return SumFormExpr<std::tuple<decltype(term)>, Declarations>{std::tuple{term}};
}

void CheckSignatures()
{
   const Cells<"mesh"> cells;
   const InteriorFacets<"mesh"> faces;
   const TrialSpace<"u"> u;
   const TrialSpace<"p"> p;
   const TestSpace<"v"> v;

   const auto form = integrate(cells, u*v);
   const auto transposed = transpose(form);
   CheckSinglePair<decltype(transposed), "v", "u">();
   static_assert(FormExpr<decltype(transposed)>);
   static_assert(!FieldExpr<decltype(transposed)>);
   static_assert(form_symbols_t<decltype(transposed)>::template is_active<"v">);
   static_assert(!form_symbols_t<decltype(transposed)>::template is_active<"u">);
   static_assert(form_symbols_t<decltype(transposed)>::template is_test<"u">);
   static_assert(!form_symbols_t<decltype(transposed)>::template is_test<"v">);

   const auto round_trip = transpose(transposed);
   CheckSinglePair<decltype(round_trip), "u", "v">();

   const auto coupled = transpose(integrate(cells, (u+p)*v));
   using Coupled = form_signature_t<decltype(coupled)>;
   static_assert(Coupled::valid);
   static_assert(Coupled::active_affinity == ActiveAffinity::Linear);
   static_assert(weak_form_traits::list_size<typename Coupled::callable_inputs>::value == 1);
   static_assert(contains_named_field_requirement_v<typename Coupled::callable_inputs, "v">);
   static_assert(weak_form_traits::list_size<typename Coupled::logical_outputs>::value == 2);
   static_assert(contains_named_field_requirement_v<typename Coupled::logical_outputs, "u">);
   static_assert(contains_named_field_requirement_v<typename Coupled::logical_outputs, "p">);
   ForEachIntegrand(coupled, [](const auto& term)
   {
      using Term = std::remove_cvref_t<decltype(term)>;
      using Signature = integrand_signature_t<Term, form_symbols_t<decltype(coupled)>>;
      static_assert(Signature::valid);
      static_assert(weak_form_traits::list_size<typename Signature::logical_outputs>::value == 1);
   });

   const auto combined = coupled + integrate(cells, TrialSpace<"v">{}*TestSpace<"u">{});
   static_assert(form_signature_t<decltype(combined)>::valid);

   const auto cross = transpose(integrate(faces, minus(u)*plus(v)));
   using Cross = form_signature_t<decltype(cross)>;
   CheckSinglePair<decltype(cross), "v", "u">();
   static_assert(contains_named_field_requirement_v<
      typename Cross::template active_inputs_on<SemanticSide::Plus>, "v">);
   static_assert(is_empty_type_list_v<typename Cross::template active_inputs_on<SemanticSide::Minus>>);
   static_assert(contains_named_field_requirement_v<
      typename Cross::template logical_outputs_on<SemanticSide::Minus>, "u">);
   static_assert(is_empty_type_list_v<typename Cross::template logical_outputs_on<SemanticSide::Plus>>);

   const auto derivative = transpose(integrate(cells, dot(grad(u), grad(v))));
   using Derivative = form_signature_t<decltype(derivative)>;
   using Input = find_named_field_requirement_t<typename Derivative::active_inputs, "v">;
   using Output = find_named_field_requirement_t<typename Derivative::logical_outputs, "u">;
   static_assert(need_gradients(Input::mask) && need_gradients(Output::mask));

   const auto vector_form = transpose(integrate(cells,
      dot(VectorTrialSpace<"U">{}, VectorTestSpace<"V">{})));
   CheckSinglePair<decltype(vector_form), "V", "U">();
   using VectorSymbols = form_symbols_t<decltype(vector_form)>;
   using VectorInput = weak_form_traits::only_field_t<typename VectorSymbols::active_declarations>;
   using VectorOutput = weak_form_traits::only_field_t<typename VectorSymbols::test_declarations>;
   static_assert(VectorInput::shape == FieldShape::Vector && VectorOutput::shape == FieldShape::Vector);

   const auto same_name = transpose(integrate(cells, TrialSpace<"w">{}*TestSpace<"w">{}));
   CheckSinglePair<decltype(same_name), "w", "w">();

   const auto frozen = MakeCoefficient<"read_f", FieldValue<"f">>([](Real f) { return 2+f; });
   const auto prescribed = transpose(integrate(cells, frozen*u*v));
   using Prescribed = form_signature_t<decltype(prescribed)>;
   static_assert(contains_named_field_requirement_v<typename Prescribed::bound_inputs, "f">);
   static_assert(!contains_named_field_requirement_v<typename Prescribed::active_inputs, "f">);

   const auto residual = integrate(cells, (u+FiniteElementField<"f">{})*v);
   const auto linear_transpose = transpose(lhs(residual));
   CheckSinglePair<decltype(linear_transpose), "v", "u">();
   static_assert(is_empty_type_list_v<typename form_signature_t<decltype(linear_transpose)>::bound_inputs>);

   const auto zero = ZeroWithContract(form, cells);
   const auto zero_transpose = transpose(zero);
   using ZeroSignature = form_signature_t<decltype(zero_transpose)>;
   CheckSinglePair<decltype(zero_transpose), "v", "u">();
   static_assert(ZeroSignature::is_zero);
   static_assert(is_empty_type_list_v<typename ZeroSignature::active_inputs>);
   static_assert(is_empty_type_list_v<typename ZeroSignature::bound_inputs>);
   CheckSinglePair<decltype(transpose(zero_transpose)), "u", "v">();

   const auto cross_zero = transpose(ZeroWithContract(integrate(faces, minus(u)*plus(v)), faces));
   using CrossZero = form_signature_t<decltype(cross_zero)>;
   using CrossZeroInputs = typename weak_form_traits::callable_relations_for<
      typename CrossZero::callable_input_relations, InteriorFacets<"mesh">>::type;
   static_assert(contains_named_field_requirement_v<typename CrossZeroInputs::template on<SemanticSide::Plus>, "v">);
   static_assert(contains_named_field_requirement_v<typename CrossZero::template logical_outputs_on<SemanticSide::Minus>, "u">);

   const auto rhs_zero = rhs(form);
   static_assert(form_signature_t<decltype(rhs_zero)>::is_zero);
   static_assert(is_empty_type_list_v<typename form_signature_t<decltype(rhs_zero)>::callable_inputs>);
   static_assert(form_symbols_t<decltype(rhs_zero)>::template is_active<"u">);
   const auto still_input_free = lhs(rhs_zero);
   static_assert(is_empty_type_list_v<typename form_signature_t<decltype(still_input_free)>::callable_inputs>);

   const auto pointwise = transpose(ScaleExpr{Real{3}});
   static_assert(FieldExpr<decltype(pointwise)> && !FormExpr<decltype(pointwise)>);

   // Deferred scalar multiplication must retain its test name when nested.
   // Binding checks its eventual shape; no context-free shape is invented.
   const FiniteElementField<"f"> f;
   const auto left_multiplier = -(2.5*(f*v));
   const auto right_multiplier = -(2.5*(v*f));
   static_assert(is_test_linear_v<decltype(left_multiplier)>);
   static_assert(is_test_linear_v<decltype(right_multiplier)>);
   static_assert(test_name_v<decltype(left_multiplier)> == StaticString{"v"});
   static_assert(test_name_v<decltype(right_multiplier)> == StaticString{"v"});
   static_assert(!has_field_shape_v<decltype(f)>);
   static_assert(!has_field_shape_v<decltype(f*v)>);
   static_assert(is_nonlinear_in_test_v<decltype((f*v)*v)>);
   static_assert(is_unsupported_test_linear_v<decltype(VectorTestSpace<"V">{}*Normal{})>);
   static_assert(is_unsupported_test_linear_v<decltype(Normal{}*VectorTestSpace<"V">{})>);
   static_assert(is_unsupported_test_linear_v<decltype(Normal{}*grad(VectorTestSpace<"V">{}))>);
   using MatrixFactor = decltype(grad(VectorTrialSpace<"U">{}));
   static_assert(is_unsupported_test_linear_v<MultFieldExpr<MatrixFactor,TestSpace<"v">>>);
}

void CheckAffineAdapters()
{
   using namespace weak_form_detail;
   const TrialSpace<"u"> u;
   const TestSpace<"v"> v;
   auto state = u+ScaleExpr{Real{2}};
   auto residual = state*v;
   constexpr auto provenance = GeometryRequirement::ElementIndex;
   auto wrapped = TransposedBilinearExpr<decltype(residual),provenance>{residual};
   auto nested = -wrapped;
   using Symbols = form_symbols_t<decltype(integrate(Cells<"mesh">{},nested))>;

   // Ordinary transposed forms are homogeneous and bypass structural splitting.
   // An affine child exercises lookup of the adapters defined in transposeform.
   auto linear = AffineSplit<AffinePart::Linear,Symbols,SemanticEntity::Cell>(nested);
   auto constant = AffineSplit<AffinePart::Constant,Symbols,SemanticEntity::Cell>(nested);
   using LinearChild = decltype(u*v);
   using ConstantChild = decltype(ScaleExpr{Real{2}}*v);
   static_assert(std::is_same_v<decltype(linear),NegExpr<TransposedBilinearExpr<LinearChild,provenance>>>);
   static_assert(std::is_same_v<decltype(constant),NegExpr<TransposedBilinearExpr<ConstantChild,provenance>>>);
   static_assert(is_split_zero_v<decltype(SplitAffineNode<AffinePart::Constant,Symbols,SemanticEntity::Cell>(linear.expr))>);

   auto beta = MakeVectorCoefficient<"beta">([] { return std::array<Real,1>{3}; });
   constexpr auto branch = FluxBranch::BoundaryOutflow;
   auto flux = UpwindFluxBranchExpr<decltype(beta),decltype(state),branch>{beta,state};
   using FluxSymbols = form_symbols_t<decltype(integrate(BoundaryFacets<"mesh">{},flux*v))>;
   auto linear_flux = AffineSplit<AffinePart::Linear,FluxSymbols,SemanticEntity::BoundaryFacet>(flux);
   auto constant_flux = AffineSplit<AffinePart::Constant,FluxSymbols,SemanticEntity::BoundaryFacet>(flux);
   static_assert(std::is_same_v<decltype(linear_flux),UpwindFluxBranchExpr<decltype(beta),TrialSpace<"u">,branch>>);
   static_assert(std::is_same_v<decltype(constant_flux),UpwindFluxBranchExpr<decltype(beta),ScaleExpr,branch>>);
   static_assert(is_split_zero_v<decltype(SplitAffineNode<AffinePart::Linear,FluxSymbols,SemanticEntity::BoundaryFacet>(constant_flux))>);

   auto interior = upwind(beta,state);
   using InteriorSymbols = form_symbols_t<decltype(integrate(InteriorFacets<"mesh">{},interior*jump(v)))>;
   auto interior_linear = AffineSplit<AffinePart::Linear,InteriorSymbols,SemanticEntity::InteriorFacet>(interior);
   auto interior_constant = AffineSplit<AffinePart::Constant,InteriorSymbols,SemanticEntity::InteriorFacet>(interior);
   static_assert(std::is_same_v<decltype(interior_linear),UpwindExpr<decltype(beta),TrialSpace<"u">>>);
   static_assert(std::is_same_v<decltype(interior_constant),UpwindExpr<decltype(beta),ScaleExpr>>);
}

bool CheckZeroOperator()
{
   CartesianMesh<1> mesh({2}, {1.}, {0.}, false);
   const auto u_space = MakeFiniteElementSpace(mesh,
      MakeLegendreFiniteElement(FiniteElementOrders<0>{}), ContiguousL2RestrictionSpecification{1, 5});
   const auto v_space = MakeFiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<1>{}), ContiguousL2RestrictionSpecification{2, 9});
   const auto context = MakeWeakFormContext(MakeTrialField<"u">(u_space),
      MakeTestField<"v">(v_space), MakeIntegrationDomain<"mesh">(mesh));
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<2>{});
   const auto source = integrate(Cells<"mesh">{},
      FiniteElementField<"discarded">{}*TrialSpace<"u">{}*TestSpace<"v">{});
   const auto op = MakeGenericOperator<SerialKernelConfiguration>(
      transpose(ZeroWithContract(source, Cells<"mesh">{})), context, rule);
   bool ok = op.InputSize() == 9 && op.OutputSize() == 5;
   Vector input(9), output(5);
   input = 7;
   output = -3;
   op.ApplyAdd(input, output);
   for (Integer i = 0; i < 5; ++i) ok = output.ReadHostData()[i] == -3 && ok;
   op(input, output);
   for (Integer i = 0; i < 5; ++i) ok = output.ReadHostData()[i] == 0 && ok;

   // Dead traversal-sensitive callbacks do not constrain a zero's execution.
   const auto index = MakeCoefficient<"dead_index", ElementIndex>(
      [](auto cell) { return Real{1}+Real(cell); });
   const auto facet_source = integrate(InteriorFacets<"mesh">{},
      index*minus(TrialSpace<"u">{})*plus(TestSpace<"v">{}));
   const auto facet_op = MakeGenericOperator<SerialKernelConfiguration>(
      transpose(ZeroWithContract(facet_source, InteriorFacets<"mesh">{})), context, rule);
   ok = facet_op.InputSize() == 9 && facet_op.OutputSize() == 5 && ok;
   output = 4;
   facet_op.ApplyAdd(input, output);
   for (Integer i = 0; i < 5; ++i) ok = output.ReadHostData()[i] == 4 && ok;
   facet_op(input, output);
   for (Integer i = 0; i < 5; ++i) ok = output.ReadHostData()[i] == 0 && ok;
   if (!ok) std::cerr << "typed-zero transpose lost its contract or output semantics\n";
   return ok;
}

struct StoredCoefficient
{
   const Real* value;
   int* calls;
   StoredCoefficient() = delete;
   StoredCoefficient(const Real& value_, int& calls_) : value(&value_), calls(&calls_) {}
   Real operator()() const { ++*calls; return Real{2}+Real{3} * *value; }
};

bool CheckStoredCoefficient()
{
   Real parameter = 2;
   int calls = 0;
   const auto coefficient = MakeCoefficient<"stored">(StoredCoefficient{parameter, calls});
   const auto form = integrate(Cells<"mesh">{}, coefficient*TrialSpace<"u">{}*TestSpace<"v">{});
   const auto transposed = transpose(form);
   if (calls != 0) return false;
   CartesianMesh<1> mesh({1}, {1.}, {0.}, false);
   const auto space = MakeFiniteElementSpace(mesh, MakeLegendreFiniteElement(FiniteElementOrders<0>{}));
   const auto context = MakeWeakFormContext(MakeTrialField<"u">(space),
      MakeTestField<"v">(space), MakeIntegrationDomain<"mesh">(mesh));
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<2>{});
   const auto forward = MakeGenericOperator<SerialKernelConfiguration>(form, context, rule);
   const auto reverse = MakeGenericOperator<SerialKernelConfiguration>(transposed, context, rule);
   if (calls != 0) return false;
   Vector x(1), result(1);
   x = 4;
   for (const Real value : {Real{2}, Real{-0.5}})
   {
      parameter = value;
      const Real expected = 4*(2+3*value);
      forward(x, result);
      if (!std::isfinite(result.ReadHostData()[0]) || std::abs(result.ReadHostData()[0]-expected) > 1e-13) return false;
      reverse(x, result);
      if (!std::isfinite(result.ReadHostData()[0]) || std::abs(result.ReadHostData()[0]-expected) > 1e-13) return false;
   }
   return calls > 0;
}
}

int main()
{
   CheckSignatures();
   CheckAffineAdapters();
   if (!CheckZeroOperator()) return 1;
   if (!CheckStoredCoefficient())
   {
      std::cerr << "transpose lost a stored callback or read its parameter at construction\n";
      return 1;
   }
   std::cout << "Bilinear form transpose preserves symbolic roles, zero contracts, and stored callbacks.\n";
}

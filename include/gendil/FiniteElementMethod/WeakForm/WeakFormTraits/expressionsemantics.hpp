// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/FiniteElementMethod/WeakForm/coefficientinputs.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/formsymbols.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/semantictypes.hpp"
#include "gendil/FiniteElementMethod/WeakForm/zero.hpp"

namespace gendil
{

namespace weak_form_traits
{
// Keep declaration lookup and callback-role resolution identical while making
// entity-specific trace rules available throughout the recursive analysis.
template<class Symbols, SemanticEntity Entity>
struct entity_symbols : Symbols
{
   static constexpr auto semantic_entity = Entity;
};
template<class Symbols>
inline constexpr auto analysis_entity_v = []
{
   if constexpr (requires { Symbols::semantic_entity; })
      return Symbols::semantic_entity;
   else
      return SemanticEntity::Unspecified;
}();

template<class U = type_list<>, class M = type_list<>, class P = type_list<>>
struct side_requirements
{
   template<SemanticSide Side>
   using on = std::conditional_t<Side == SemanticSide::Unqualified, U,
              std::conditional_t<Side == SemanticSide::Minus, M, P>>;
   using all = merge_named_field_requirement_list_t<concat_many_t<U, M, P>>;
};

template<SemanticSide Side, class Requirement>
using one_side_requirement = side_requirements<
   std::conditional_t<Side == SemanticSide::Unqualified, type_list<Requirement>, type_list<>>,
   std::conditional_t<Side == SemanticSide::Minus, type_list<Requirement>, type_list<>>,
   std::conditional_t<Side == SemanticSide::Plus, type_list<Requirement>, type_list<>>>;

template<class... Sides>
using merge_sides = side_requirements<
   merge_named_field_requirement_list_t<concat_many_t<typename Sides::template on<SemanticSide::Unqualified>...>>,
   merge_named_field_requirement_list_t<concat_many_t<typename Sides::template on<SemanticSide::Minus>...>>,
   merge_named_field_requirement_list_t<concat_many_t<typename Sides::template on<SemanticSide::Plus>...>>>;

template<GeometryRequirement U = GeometryRequirement::None,
         GeometryRequirement M = GeometryRequirement::None,
         GeometryRequirement P = GeometryRequirement::None>
struct side_geometry
{
   template<SemanticSide Side>
   static constexpr auto on = Side == SemanticSide::Unqualified ? U : Side == SemanticSide::Minus ? M : P;
   static constexpr auto all = U | M | P;
};

template<SemanticSide Side, GeometryRequirement Geometry>
using one_side_geometry = side_geometry<
   Side == SemanticSide::Unqualified ? Geometry : GeometryRequirement::None,
   Side == SemanticSide::Minus ? Geometry : GeometryRequirement::None,
   Side == SemanticSide::Plus ? Geometry : GeometryRequirement::None>;

template<class... Sides>
using merge_geometry = side_geometry<
   (GeometryRequirement::None | ... | Sides::template on<SemanticSide::Unqualified>),
   (GeometryRequirement::None | ... | Sides::template on<SemanticSide::Minus>),
   (GeometryRequirement::None | ... | Sides::template on<SemanticSide::Plus>)>;

// Keep candidate ranks as well as their public summary. An unresolved field
// may have any rank, but a product involving a matrix cannot become scalar.
constexpr unsigned shape_options(SemanticShape shape)
{
   return shape == SemanticShape::Scalar ? 1u :
          shape == SemanticShape::Vector ? 2u :
          shape == SemanticShape::Matrix ? 4u :
          shape == SemanticShape::Unresolved ? 7u : 0u;
}
constexpr SemanticShape summarize_shape(unsigned options)
{
   return options == 1u ? SemanticShape::Scalar :
          options == 2u ? SemanticShape::Vector :
          options == 4u ? SemanticShape::Matrix :
          options == 0u ? SemanticShape::Unsupported : SemanticShape::Unresolved;
}

struct empty_semantics
{
   using active_sides = side_requirements<>;
   using bound_sides = side_requirements<>;
   using output_sides = side_requirements<>;
   using geometry_sides = side_geometry<>;
   using shape_obligations = type_list<>;
   static constexpr bool supported = true;
   static constexpr bool requires_facet = false;
   static constexpr bool requires_two_sides = false;
   static constexpr auto symbolic_shape = SemanticShape::Scalar;
   static constexpr unsigned possible_shapes = shape_options(symbolic_shape);
   static constexpr auto test_linearity = SemanticTestLinearity::TestFree;
   static constexpr auto active_affinity = ActiveAffinity::Independent;
};

template<class... Children>
struct merge_information : empty_semantics
{
   using active_sides = merge_sides<typename Children::active_sides...>;
   using bound_sides = merge_sides<typename Children::bound_sides...>;
   using output_sides = merge_sides<typename Children::output_sides...>;
   using geometry_sides = merge_geometry<typename Children::geometry_sides...>;
   using shape_obligations = unique_t<concat_many_t<typename Children::shape_obligations...>>;
   static constexpr bool supported = (Children::supported && ...);
   static constexpr bool requires_facet = (Children::requires_facet || ...);
   static constexpr bool requires_two_sides = (Children::requires_two_sides || ...);
};

constexpr auto semantic_shape(FieldShape shape)
{
   return shape == FieldShape::Scalar ? SemanticShape::Scalar :
          shape == FieldShape::Vector ? SemanticShape::Vector : SemanticShape::Matrix;
}
constexpr auto gradient_shape(SemanticShape shape)
{
   return shape == SemanticShape::Scalar ? SemanticShape::Vector :
          shape == SemanticShape::Vector ? SemanticShape::Matrix :
          shape == SemanticShape::Unresolved ? SemanticShape::Unresolved : SemanticShape::Unsupported;
}
constexpr auto sum_affinity(ActiveAffinity a, ActiveAffinity b)
{
   if (a == ActiveAffinity::Unknown || b == ActiveAffinity::Unknown) return ActiveAffinity::Unknown;
   if (a == ActiveAffinity::Nonlinear || b == ActiveAffinity::Nonlinear) return ActiveAffinity::Nonlinear;
   if (a == b) return a;
   return ActiveAffinity::Affine;
}
constexpr auto product_affinity(ActiveAffinity a, ActiveAffinity b)
{
   if (a == ActiveAffinity::Independent) return b;
   if (b == ActiveAffinity::Independent) return a;
   if (a == ActiveAffinity::Unknown || b == ActiveAffinity::Unknown) return ActiveAffinity::Unknown;
   return ActiveAffinity::Nonlinear;
}
constexpr auto product_linearity(SemanticTestLinearity a, SemanticTestLinearity b)
{
   if (a == SemanticTestLinearity::TestFree) return b;
   if (b == SemanticTestLinearity::TestFree) return a;
   if (a == SemanticTestLinearity::Unsupported || b == SemanticTestLinearity::Unsupported)
      return SemanticTestLinearity::Unsupported;
   return SemanticTestLinearity::Nonlinear;
}
constexpr auto sum_linearity(SemanticTestLinearity a, SemanticTestLinearity b, size_t output_count)
{
   if (a == SemanticTestLinearity::Unsupported || b == SemanticTestLinearity::Unsupported)
      return SemanticTestLinearity::Unsupported;
   if (a == SemanticTestLinearity::Nonlinear || b == SemanticTestLinearity::Nonlinear)
      return SemanticTestLinearity::Nonlinear;
   if (output_count > 1) return SemanticTestLinearity::MultipleOutputs;
   if (a == b) return a;
   return SemanticTestLinearity::Inhomogeneous;
}
constexpr auto combined_shape(ShapeOperation operation, SemanticShape a, SemanticShape b)
{
   using S = SemanticShape;
   if (a == S::Unsupported || b == S::Unsupported) return S::Unsupported;
   if (operation == ShapeOperation::Sum) return a == b ? a : S::Unsupported;
   if (operation == ShapeOperation::Dot) return a == S::Vector && b == S::Vector ? S::Scalar : S::Unsupported;
   if (operation == ShapeOperation::Inner)
      return a == b && (a == S::Vector || a == S::Matrix) ? S::Scalar : S::Unsupported;
   if (operation == ShapeOperation::Outer) return a == S::Vector && b == S::Vector ? S::Matrix : S::Unsupported;
   if (a == S::Scalar) return b;
   if (b == S::Scalar) return a;
   if (a == S::Matrix && (b == S::Vector || b == S::Matrix)) return b;
   return S::Unsupported;
}
constexpr unsigned combined_shape_options(ShapeOperation operation, unsigned lhs, unsigned rhs)
{
   unsigned result = 0u;
   for (unsigned a = 1u; a <= 4u; a <<= 1u)
      for (unsigned b = 1u; b <= 4u; b <<= 1u)
         if ((lhs & a) && (rhs & b))
            result |= shape_options(combined_shape(operation, summarize_shape(a), summarize_shape(b)));
   return result;
}

template<class E, class Symbols, SemanticSide Side>
struct analyze : empty_semantics
{
   static constexpr bool supported = false;
   static constexpr auto symbolic_shape = SemanticShape::Unsupported;
   static constexpr unsigned possible_shapes = 0u;
   static constexpr auto test_linearity = SemanticTestLinearity::Unsupported;
   static constexpr auto active_affinity = ActiveAffinity::Unknown;
};

template<class Declarations, class Sides, class Symbols, SemanticSide Side>
struct analyze<ZeroExpr<Declarations, Sides>, Symbols, Side> : empty_semantics
{
   using logical_output_sides = Sides;
   static constexpr auto test_linearity = SemanticTestLinearity::Linear;
};

enum class ReadRole { Active, Bound, Test };
template<StaticString Name, SemanticShape Shape, ReadRole Role, OperatorMask Mask,
         NamedFieldProvenance Provenance, SemanticSide Side>
struct field_semantics : empty_semantics
{
   using read = one_side_requirement<Side, NamedFieldRequirement<Name, Mask, Provenance>>;
   using active_sides = std::conditional_t<Role == ReadRole::Active, read, side_requirements<>>;
   using bound_sides = std::conditional_t<Role == ReadRole::Bound, read, side_requirements<>>;
   using output_sides = std::conditional_t<Role == ReadRole::Test, read, side_requirements<>>;
   using geometry_sides = one_side_geometry<Side,
      need_gradients(Mask) ? GeometryRequirement::JacobianInverse : GeometryRequirement::None>;
   static constexpr auto symbolic_shape = Shape;
   static constexpr unsigned possible_shapes = shape_options(Shape);
   static constexpr auto test_linearity = Role == ReadRole::Test ? SemanticTestLinearity::Linear : SemanticTestLinearity::TestFree;
   static constexpr auto active_affinity = Role == ReadRole::Active ? ActiveAffinity::Linear : ActiveAffinity::Independent;
};

template<StaticString N, FieldShape Shape, class Symbols, SemanticSide Side>
struct analyze<TrialSpace<N, Shape>, Symbols, Side>
 : field_semantics<N, semantic_shape(Shape), ReadRole::Active, OperatorMask::Values, NamedFieldProvenance::ActiveTrial, Side> {};
template<StaticString N, FieldShape Shape, class Symbols, SemanticSide Side>
struct analyze<TestSpace<N, Shape>, Symbols, Side>
 : field_semantics<N, semantic_shape(Shape), ReadRole::Test, OperatorMask::Values, NamedFieldProvenance::ActiveTest, Side> {};
template<StaticString N, class Symbols, SemanticSide Side>
struct analyze<FiniteElementField<N>, Symbols, Side>
 : field_semantics<N, SemanticShape::Unresolved, ReadRole::Bound, OperatorMask::Values, NamedFieldProvenance::FiniteElementExpression, Side>
{
   using shape_obligations = type_list<ShapeObligation<ShapeOperation::BoundField, NameTag<N>>>;
};

template<StaticString N, FieldShape Shape, class Symbols, SemanticSide Side>
struct analyze<GradientExpr<TrialSpace<N, Shape>>, Symbols, Side>
 : field_semantics<N, gradient_shape(semantic_shape(Shape)), ReadRole::Active, OperatorMask::Gradients, NamedFieldProvenance::ActiveTrial, Side> {};
template<StaticString N, FieldShape Shape, class Symbols, SemanticSide Side>
struct analyze<GradientExpr<TestSpace<N, Shape>>, Symbols, Side>
 : field_semantics<N, gradient_shape(semantic_shape(Shape)), ReadRole::Test, OperatorMask::Gradients, NamedFieldProvenance::ActiveTest, Side> {};
template<StaticString N, class Symbols, SemanticSide Side>
struct analyze<GradientExpr<FiniteElementField<N>>, Symbols, Side>
 : field_semantics<N, SemanticShape::Unresolved, ReadRole::Bound, OperatorMask::Gradients, NamedFieldProvenance::FiniteElementExpression, Side>
{
   static constexpr unsigned possible_shapes = shape_options(SemanticShape::Vector) | shape_options(SemanticShape::Matrix);
   using shape_obligations = type_list<ShapeObligation<ShapeOperation::Gradient, NameTag<N>>>;
};

// The result rank is known before binding, but vector component counts must
// match the physical dimension of each selected finite-element space.
template<StaticString N, ReadRole Role, NamedFieldProvenance Provenance, SemanticSide Side>
struct divergence_semantics
 : field_semantics<N, SemanticShape::Scalar, Role, OperatorMask::Gradients, Provenance, Side>
{
   using shape_obligations = type_list<ShapeObligation<ShapeOperation::Divergence,
      NameTag<N>, std::integral_constant<SemanticSide, Side>>>;
};
template<StaticString N, class Symbols, SemanticSide Side>
struct analyze<DivergenceExpr<TrialSpace<N, FieldShape::Vector>>, Symbols, Side>
 : divergence_semantics<N, ReadRole::Active, NamedFieldProvenance::ActiveTrial, Side> {};
template<StaticString N, class Symbols, SemanticSide Side>
struct analyze<DivergenceExpr<TestSpace<N, FieldShape::Vector>>, Symbols, Side>
 : divergence_semantics<N, ReadRole::Test, NamedFieldProvenance::ActiveTest, Side> {};
template<StaticString N, class Symbols, SemanticSide Side>
struct analyze<DivergenceExpr<FiniteElementField<N>>, Symbols, Side>
 : divergence_semantics<N, ReadRole::Bound, NamedFieldProvenance::FiniteElementExpression, Side> {};

template<StaticString N, ReadRole Role, NamedFieldProvenance Provenance,
         SemanticSide Side>
struct curl_semantics
 : field_semantics<N, SemanticShape::Vector, Role, OperatorMask::Gradients, Provenance, Side>
{
   using shape_obligations = type_list<ShapeObligation<ShapeOperation::Curl,
      NameTag<N>, std::integral_constant<SemanticSide, Side>>>;
};
template<StaticString N, class Symbols, SemanticSide Side>
struct analyze<CurlExpr<TrialSpace<N, FieldShape::Vector>>, Symbols, Side>
 : curl_semantics<N, ReadRole::Active, NamedFieldProvenance::ActiveTrial, Side> {};
template<StaticString N, class Symbols, SemanticSide Side>
struct analyze<CurlExpr<TestSpace<N, FieldShape::Vector>>, Symbols, Side>
 : curl_semantics<N, ReadRole::Test, NamedFieldProvenance::ActiveTest, Side> {};
template<StaticString N, class Symbols, SemanticSide Side>
struct analyze<CurlExpr<FiniteElementField<N>>, Symbols, Side>
 : curl_semantics<N, ReadRole::Bound, NamedFieldProvenance::FiniteElementExpression, Side> {};

template<class Symbols, SemanticSide Side>
struct analyze<ScaleExpr, Symbols, Side> : empty_semantics {};
template<class Symbols, SemanticSide Side>
struct analyze<Normal, Symbols, Side> : empty_semantics
{
   using geometry_sides = one_side_geometry<Side, GeometryRequirement::Normal>;
   static constexpr auto symbolic_shape = SemanticShape::Vector;
   static constexpr unsigned possible_shapes = shape_options(symbolic_shape);
   static constexpr bool requires_facet = true;
};

template<class Input, class Symbols, SemanticSide Side>
struct input_semantics : empty_semantics
{
   static constexpr bool supported = false;
};

#define GENDIL_GEOMETRY_INPUT(Input, Requirement) \
template<class Symbols, SemanticSide Side> \
struct input_semantics<Input, Symbols, Side> : empty_semantics { \
   using geometry_sides = one_side_geometry<Side, GeometryRequirement::Requirement>; \
};
GENDIL_GEOMETRY_INPUT(ElementIndex, ElementIndex)
GENDIL_GEOMETRY_INPUT(QuadraturePointIndex, QuadratureIndex)
GENDIL_GEOMETRY_INPUT(PhysicalCoordinates, PhysicalCoordinates)
GENDIL_GEOMETRY_INPUT(ReferenceCoordinates, ReferenceCoordinates)
GENDIL_GEOMETRY_INPUT(JacobianDeterminant, Determinant)
GENDIL_GEOMETRY_INPUT(Jacobian, Jacobian)
GENDIL_GEOMETRY_INPUT(JacobianInverse, JacobianInverse)
#undef GENDIL_GEOMETRY_INPUT

template<class Symbols, SemanticSide Side>
struct input_semantics<InverseFacetSize, Symbols, Side> : empty_semantics
{
   using geometry_sides = one_side_geometry<Side, GeometryRequirement::InverseFacetSize | GeometryRequirement::JacobianInverse>;
   static constexpr bool requires_facet = true;
};

template<StaticString N, OperatorMask Mask, class Symbols, SemanticSide Side>
struct named_input_semantics : field_semantics<N, SemanticShape::Unresolved,
   Symbols::template is_active<N> ? ReadRole::Active : ReadRole::Bound,
   Mask, NamedFieldProvenance::CoefficientInput, Side>
{};
template<StaticString N, class Symbols, SemanticSide Side>
struct input_semantics<FieldValue<N>, Symbols, Side>
 : named_input_semantics<N, OperatorMask::Values, Symbols, Side> {};
template<StaticString N, class Symbols, SemanticSide Side>
struct input_semantics<FieldGradient<N>, Symbols, Side>
 : named_input_semantics<N, OperatorMask::Gradients, Symbols, Side> {};

template<StaticString N, FieldShape Shape, class Fn, CoefficientInput... Inputs, class Symbols, SemanticSide Side>
struct analyze<Coefficient<N, Shape, Fn, Inputs...>, Symbols, Side>
 : merge_information<input_semantics<std::remove_cvref_t<Inputs>, Symbols, Side>...>
{
   using base = merge_information<input_semantics<std::remove_cvref_t<Inputs>, Symbols, Side>...>;
   static constexpr auto symbolic_shape = semantic_shape(Shape);
   static constexpr unsigned possible_shapes = shape_options(symbolic_shape);
   static constexpr auto active_affinity = is_empty_type_list_v<typename base::active_sides::all>
      ? ActiveAffinity::Independent : ActiveAffinity::Unknown;
   static constexpr auto test_linearity = is_empty_type_list_v<typename base::output_sides::all>
      ? SemanticTestLinearity::TestFree : SemanticTestLinearity::Unsupported;
};

template<FieldExpr E, class Symbols, SemanticSide Side>
struct analyze<NegExpr<E>, Symbols, Side> : analyze<E, Symbols, Side> {};
template<FieldExpr E, class Symbols, SemanticSide Side>
struct analyze<TransposeExpr<E>, Symbols, Side> : analyze<E, Symbols, Side>
{
   using child = analyze<E, Symbols, Side>;
   static constexpr unsigned possible_shapes = child::possible_shapes & shape_options(SemanticShape::Matrix);
   static constexpr auto symbolic_shape = summarize_shape(possible_shapes);
   using shape_obligations = std::conditional_t<child::symbolic_shape == SemanticShape::Unresolved,
      concat_t<typename child::shape_obligations, type_list<ShapeObligation<ShapeOperation::Transpose, E>>>,
      typename child::shape_obligations>;
};

template<FieldExpr E, class Symbols, SemanticSide Side>
struct square_matrix_semantics : analyze<E, Symbols, Side>
{
   using child = analyze<E, Symbols, Side>;
   static constexpr unsigned possible_shapes = child::possible_shapes & shape_options(SemanticShape::Matrix);
   static constexpr auto symbolic_shape = summarize_shape(possible_shapes);
   // Matrix rank alone does not prove equal extents. Validate each occurrence
   // in its evaluation-side context, including statically known matrix ranks.
   using shape_obligations = concat_t<typename child::shape_obligations,
      type_list<ShapeObligation<ShapeOperation::SquareMatrix, E,
         std::integral_constant<SemanticSide, Side>>>>;
};

template<FieldExpr E, class Symbols, SemanticSide Side>
struct analyze<SymExpr<E>, Symbols, Side> : square_matrix_semantics<E, Symbols, Side> {};

template<FieldExpr E, class Symbols, SemanticSide Side>
struct analyze<SkewExpr<E>, Symbols, Side> : square_matrix_semantics<E, Symbols, Side> {};

template<class A, class B, ShapeOperation Operation, class L, class R>
struct binary_semantics : merge_information<A, B>
{
   using base = merge_information<A, B>;
   static constexpr unsigned possible_shapes = combined_shape_options(Operation, A::possible_shapes, B::possible_shapes);
   static constexpr auto symbolic_shape = summarize_shape(possible_shapes);
   static constexpr auto active_affinity = Operation == ShapeOperation::Sum
      ? sum_affinity(A::active_affinity, B::active_affinity)
      : product_affinity(A::active_affinity, B::active_affinity);
   static constexpr auto test_linearity = Operation == ShapeOperation::Sum
      ? sum_linearity(A::test_linearity, B::test_linearity, list_size<typename base::output_sides::all>::value)
      : product_linearity(A::test_linearity, B::test_linearity);
   using shape_obligations = std::conditional_t<
      A::symbolic_shape == SemanticShape::Unresolved || B::symbolic_shape == SemanticShape::Unresolved,
      concat_t<typename base::shape_obligations, type_list<ShapeObligation<Operation, L, R>>>,
      typename base::shape_obligations>;
};

#define GENDIL_BINARY_SEMANTICS(Node, Operation) \
template<FieldExpr L, FieldExpr R, class Symbols, SemanticSide Side> \
struct analyze<Node<L, R>, Symbols, Side> \
 : binary_semantics<analyze<L, Symbols, Side>, analyze<R, Symbols, Side>, ShapeOperation::Operation, L, R> {};
GENDIL_BINARY_SEMANTICS(ProductExpr, Product)
GENDIL_BINARY_SEMANTICS(MultFieldExpr, Product)
GENDIL_BINARY_SEMANTICS(DotExpr, Dot)
GENDIL_BINARY_SEMANTICS(InnerExpr, Inner)
GENDIL_BINARY_SEMANTICS(OuterExpr, Outer)
#undef GENDIL_BINARY_SEMANTICS

template<FieldExpr Head, class Symbols, SemanticSide Side>
struct analyze<SumExpr<Head>, Symbols, Side> : analyze<Head, Symbols, Side> {};
template<FieldExpr Head, FieldExpr Next, FieldExpr... Tail, class Symbols, SemanticSide Side>
struct analyze<SumExpr<Head, Next, Tail...>, Symbols, Side>
 : binary_semantics<analyze<Head, Symbols, Side>, analyze<SumExpr<Next, Tail...>, Symbols, Side>,
                    ShapeOperation::Sum, Head, SumExpr<Next, Tail...>> {};

template<FieldExpr E, class Symbols, SemanticSide Side>
struct analyze<MinusTraceExpr<E>, Symbols, Side> : analyze<E, Symbols, SemanticSide::Minus>
{
   static constexpr bool supported = Side == SemanticSide::Unqualified &&
      analysis_entity_v<Symbols> != SemanticEntity::BoundaryFacet &&
      analyze<E, Symbols, SemanticSide::Minus>::supported;
   // Explicit minus/plus traces always describe a two-sided interface.
   static constexpr bool requires_facet = true;
   static constexpr bool requires_two_sides = true;
};
template<FieldExpr E, class Symbols, SemanticSide Side>
struct analyze<PlusTraceExpr<E>, Symbols, Side> : analyze<E, Symbols, SemanticSide::Plus>
{
   static constexpr bool supported = Side == SemanticSide::Unqualified &&
      analysis_entity_v<Symbols> != SemanticEntity::BoundaryFacet &&
      analyze<E, Symbols, SemanticSide::Plus>::supported;
   static constexpr bool requires_facet = true;
   static constexpr bool requires_two_sides = true;
};
template<FieldExpr E, class Symbols>
struct both_sides : merge_information<analyze<E, Symbols, SemanticSide::Minus>, analyze<E, Symbols, SemanticSide::Plus>>
{
   using child = analyze<E, Symbols, SemanticSide::Minus>;
   static constexpr bool requires_facet = true;
   static constexpr bool requires_two_sides = true;
   static constexpr auto symbolic_shape = child::symbolic_shape;
   static constexpr unsigned possible_shapes = child::possible_shapes;
   static constexpr auto test_linearity = child::test_linearity;
   static constexpr auto active_affinity = child::active_affinity;
};
// Boundary jump/average are one-sided identities. Recursion still examines
// every operand, so interior-only operators cannot hide inside a wrapper.
template<FieldExpr E, class Symbols,
         bool Boundary = analysis_entity_v<Symbols> == SemanticEntity::BoundaryFacet>
struct trace_sides : both_sides<E, Symbols> {};
template<FieldExpr E, class Symbols>
struct trace_sides<E, Symbols, true> : analyze<E, Symbols, SemanticSide::Unqualified>
{
   static constexpr bool requires_facet = true;
};
template<FieldExpr E, class Symbols, SemanticSide Side>
struct analyze<AverageExpr<E>, Symbols, Side> : trace_sides<E, Symbols>
{
   static constexpr bool supported = Side == SemanticSide::Unqualified && trace_sides<E, Symbols>::supported;
};
template<FieldExpr E, class Symbols, SemanticSide Side>
struct analyze<JumpExpr<E>, Symbols, Side> : trace_sides<E, Symbols>
{
   static constexpr bool supported = Side == SemanticSide::Unqualified && trace_sides<E, Symbols>::supported;
};
template<FieldExpr Adv, FieldExpr E, class Symbols, SemanticSide Side>
struct analyze<UpwindExpr<Adv, E>, Symbols, Side>
 : merge_information<analyze<Adv, Symbols, Side>, both_sides<E, Symbols>>
{
   using selector = analyze<Adv, Symbols, Side>;
   using value = both_sides<E, Symbols>;
   using base = merge_information<selector, value>;
   using geometry_sides = merge_geometry<typename base::geometry_sides,
      one_side_geometry<Side, GeometryRequirement::Normal>>;
   static constexpr bool supported = Side == SemanticSide::Unqualified && base::supported &&
      analysis_entity_v<Symbols> != SemanticEntity::BoundaryFacet &&
      (selector::possible_shapes & shape_options(SemanticShape::Vector)) != 0u;
   static constexpr auto symbolic_shape = value::symbolic_shape;
   static constexpr unsigned possible_shapes = value::possible_shapes;
   static constexpr auto test_linearity = selector::test_linearity == SemanticTestLinearity::TestFree
      ? value::test_linearity : SemanticTestLinearity::Nonlinear;
   static constexpr auto active_affinity = selector::active_affinity == ActiveAffinity::Independent
      ? value::active_affinity : ActiveAffinity::Unknown;
   using shape_obligations = std::conditional_t<selector::symbolic_shape == SemanticShape::Unresolved,
      concat_t<typename base::shape_obligations, type_list<ShapeObligation<ShapeOperation::UpwindSelector, Adv>>>,
      typename base::shape_obligations>;
};

// Only affine splitting omits a boundary state. Its retained rank constrains
// the remaining state without making either zero marker a standalone AST node.
template<typename State, class Symbols, SemanticSide Side>
struct boundary_upwind_state : analyze<State, Symbols, Side> {};
template<FieldShape Shape, class Symbols, SemanticSide Side>
struct boundary_upwind_state<weak_form_detail::BoundaryUpwindZero<Shape>, Symbols, Side> : empty_semantics
{
   static constexpr auto symbolic_shape = semantic_shape(Shape);
   static constexpr unsigned possible_shapes = shape_options(symbolic_shape);
};

template<typename Interior, typename Exterior, class Symbols, SemanticSide Side>
using boundary_upwind_states_t = std::conditional_t<std::is_same_v<Interior, Empty>,
   analyze<Exterior, Symbols, Side>,
   std::conditional_t<std::is_same_v<Exterior, Empty>, analyze<Interior, Symbols, Side>,
      binary_semantics<boundary_upwind_state<Interior, Symbols, Side>,
         boundary_upwind_state<Exterior, Symbols, Side>,
         ShapeOperation::Sum, Interior, Exterior>>>;

template<FieldExpr Adv, typename Interior, typename Exterior, class Symbols, SemanticSide Side>
struct analyze<BoundaryUpwindExpr<Adv, Interior, Exterior>, Symbols, Side>
   : merge_information<analyze<Adv, Symbols, Side>,
      boundary_upwind_states_t<Interior, Exterior, Symbols, Side>>
{
   using selector = analyze<Adv, Symbols, Side>;
   using states = boundary_upwind_states_t<Interior, Exterior, Symbols, Side>;
   using base = merge_information<selector, states>;
   using geometry_sides = merge_geometry<typename base::geometry_sides,
      one_side_geometry<Side, GeometryRequirement::Normal>>;
   static constexpr bool supported = Side == SemanticSide::Unqualified && base::supported &&
      (analysis_entity_v<Symbols> == SemanticEntity::BoundaryFacet ||
       analysis_entity_v<Symbols> == SemanticEntity::Unspecified) &&
      (selector::possible_shapes & shape_options(SemanticShape::Vector)) != 0u;
   static constexpr bool requires_facet = true;
   static constexpr bool requires_two_sides = false;
   static constexpr auto symbolic_shape = states::symbolic_shape;
   static constexpr unsigned possible_shapes = states::possible_shapes;
   static constexpr auto test_linearity = selector::test_linearity == SemanticTestLinearity::TestFree &&
      states::test_linearity == SemanticTestLinearity::TestFree
         ? SemanticTestLinearity::TestFree : SemanticTestLinearity::Unsupported;
   static constexpr auto active_affinity = []
   {
      if constexpr (selector::active_affinity != ActiveAffinity::Independent)
         return ActiveAffinity::Unknown;
      else if constexpr (is_boundary_upwind_zero_v<Interior>)
         return analyze<Exterior, Symbols, Side>::active_affinity;
      else if constexpr (is_boundary_upwind_zero_v<Exterior>)
         return analyze<Interior, Symbols, Side>::active_affinity;
      else return states::active_affinity;
   }();
   using shape_obligations = std::conditional_t<selector::symbolic_shape == SemanticShape::Unresolved,
      concat_t<typename base::shape_obligations, type_list<ShapeObligation<ShapeOperation::UpwindSelector, Adv>>>,
      typename base::shape_obligations>;
};

template<class Analysis>
struct expose_semantics : Analysis
{
   // Reserved at the analysis boundary; current expressions use real algebra.
   static constexpr auto variance = Analysis::supported
      ? SemanticVariance::Real : SemanticVariance::Unsupported;
   using active_inputs = typename Analysis::active_sides::all;
   using bound_inputs = typename Analysis::bound_sides::all;
   using test_outputs = typename Analysis::output_sides::all;
   template<SemanticSide Side> using active_inputs_on = typename Analysis::active_sides::template on<Side>;
   template<SemanticSide Side> using bound_inputs_on = typename Analysis::bound_sides::template on<Side>;
   template<SemanticSide Side> using test_outputs_on = typename Analysis::output_sides::template on<Side>;
   static constexpr auto geometry = Analysis::geometry_sides::all;
   template<SemanticSide Side> static constexpr auto geometry_on = Analysis::geometry_sides::template on<Side>;
   static constexpr bool valid = Analysis::supported && Analysis::symbolic_shape != SemanticShape::Unsupported &&
      (Analysis::test_linearity == SemanticTestLinearity::TestFree || Analysis::test_linearity == SemanticTestLinearity::Linear);
};
} // namespace weak_form_traits

/** Context-free semantic query. Symbols must describe the whole original form. */
template<class Expression, class Symbols, SemanticEntity Entity = SemanticEntity::Unspecified>
struct expression_semantics : weak_form_traits::expose_semantics<
   weak_form_traits::analyze<std::remove_cvref_t<Expression>,
      weak_form_traits::entity_symbols<Symbols, Entity>, SemanticSide::Unqualified>>
{
   using base = weak_form_traits::expose_semantics<weak_form_traits::analyze<
      std::remove_cvref_t<Expression>, weak_form_traits::entity_symbols<Symbols, Entity>, SemanticSide::Unqualified>>;
   static constexpr bool valid = Symbols::valid && base::valid;
};
template<class Expression, class Symbols, SemanticEntity Entity = SemanticEntity::Unspecified>
using expression_semantics_t = expression_semantics<Expression, Symbols, Entity>;

} // namespace gendil

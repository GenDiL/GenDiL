// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/signatures.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/restrictedweakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/executioncontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globalfacefieldbinding.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/weakformrequirements.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/domainiteration.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"

namespace gendil::weak_form_validation_detail
{
// Deferred shape checking is confined to obligations retained by analysis.
// This resolves rank from selected spaces without recollecting dependencies or
// changing callback roles. All actual reads continue to come from Signature.
template<StaticString Name, class... Fields>
consteval bool ContainsField(type_list<Fields...>)
{ return ((Name == Fields::name) || ...); }

template<StaticString Name, class Signature, SemanticSide Requested>
consteval SemanticSide ShapeSide()
{
   if constexpr (std::is_void_v<Signature>) return Requested;
   else
   {
      constexpr bool Minus = ContainsField<Name>(concat_many_t<
         typename Signature::template active_inputs_on<SemanticSide::Minus>,
         typename Signature::template bound_inputs_on<SemanticSide::Minus>,
         typename Signature::template test_outputs_on<SemanticSide::Minus>>{});
      constexpr bool Plus = ContainsField<Name>(concat_many_t<
         typename Signature::template active_inputs_on<SemanticSide::Plus>,
         typename Signature::template bound_inputs_on<SemanticSide::Plus>,
         typename Signature::template test_outputs_on<SemanticSide::Plus>>{});
      if constexpr (Plus && !Minus) return SemanticSide::Plus;
      else if constexpr (Minus && !Plus) return SemanticSide::Minus;
      else return Requested;
   }
}

template<class E, class Context, SemanticSide Side = SemanticSide::Unqualified, class Signature = void>
struct bound_shape
{
   static constexpr auto value = weak_form_traits::semantic_shape(E::field_shape);
};
template<class Context, SemanticSide Side, class Signature> struct bound_shape<ScaleExpr, Context, Side, Signature>
{ static constexpr auto value = SemanticShape::Scalar; };
template<class Context, SemanticSide Side, class Signature> struct bound_shape<Normal, Context, Side, Signature>
{ static constexpr auto value = SemanticShape::Vector; };
template<StaticString Name, class Context, SemanticSide Side, class Signature>
struct bound_shape<FiniteElementField<Name>, Context, Side, Signature>
{
   using Registered = std::remove_cvref_t<decltype(std::declval<const Context&>().template GetFiniteElementField<Name>().space)>;
   using Space = std::remove_cvref_t<decltype([](const Registered& registered) -> decltype(auto) {
      if constexpr (ShapeSide<Name, Signature, Side>() == SemanticSide::Plus && is_interior_face_field_binding_v<Registered>)
         return registered.GetPlusFiniteElementSpace();
      else if constexpr (is_boundary_face_field_binding_v<Registered> || is_interior_face_field_binding_v<Registered>)
         return registered.GetMinusFiniteElementSpace();
      else return (registered);
   }(std::declval<const Registered&>()))>;
   static constexpr auto value = is_vector_finite_element_space_v<Space>
      ? SemanticShape::Vector : SemanticShape::Scalar;
};
template<StaticString Name, class Context, SemanticSide Side, class Signature>
struct bound_shape<NameTag<Name>, Context, Side, Signature> : bound_shape<FiniteElementField<Name>, Context, Side, Signature> {};
template<FieldExpr E, class Context, SemanticSide Side, class Signature>
struct bound_shape<GradientExpr<E>, Context, Side, Signature>
{ static constexpr auto value = weak_form_traits::gradient_shape(bound_shape<E, Context, Side, Signature>::value); };
template<FieldExpr E, class Context, SemanticSide Side, class Signature>
struct bound_shape<DivergenceExpr<E>, Context, Side, Signature>
{ static constexpr auto value = SemanticShape::Scalar; };
template<FieldExpr E, class Context, SemanticSide Side, class Signature>
struct bound_shape<NegExpr<E>, Context, Side, Signature> : bound_shape<E, Context, Side, Signature> {};
template<FieldExpr E, class Context, SemanticSide Side, class Signature>
struct bound_shape<TransposeExpr<E>, Context, Side, Signature>
{ static constexpr auto value = bound_shape<E, Context, Side, Signature>::value == SemanticShape::Matrix
   ? SemanticShape::Matrix : SemanticShape::Unsupported; };
#define GENDIL_BOUND_BINARY_SHAPE(Node, Operation) \
template<FieldExpr L, FieldExpr R, class Context, SemanticSide Side, class Signature> \
struct bound_shape<Node<L, R>, Context, Side, Signature> { \
 static constexpr auto value = weak_form_traits::combined_shape(ShapeOperation::Operation, \
    bound_shape<L, Context, Side, Signature>::value, bound_shape<R, Context, Side, Signature>::value); };
GENDIL_BOUND_BINARY_SHAPE(ProductExpr, Product)
GENDIL_BOUND_BINARY_SHAPE(MultFieldExpr, Product)
GENDIL_BOUND_BINARY_SHAPE(DotExpr, Dot)
GENDIL_BOUND_BINARY_SHAPE(InnerExpr, Inner)
GENDIL_BOUND_BINARY_SHAPE(OuterExpr, Outer)
#undef GENDIL_BOUND_BINARY_SHAPE
template<FieldExpr E, class Context, SemanticSide Side, class Signature>
struct bound_shape<SumExpr<E>, Context, Side, Signature> : bound_shape<E, Context, Side, Signature> {};
template<FieldExpr E, FieldExpr F, FieldExpr... Rest, class Context, SemanticSide Side, class Signature>
struct bound_shape<SumExpr<E, F, Rest...>, Context, Side, Signature>
{ static constexpr auto value = weak_form_traits::combined_shape(ShapeOperation::Sum,
   bound_shape<E, Context, Side, Signature>::value, bound_shape<SumExpr<F, Rest...>, Context, Side, Signature>::value); };

template<FieldExpr E, class Context, SemanticSide Side, class Signature>
struct bound_shape<JumpExpr<E>, Context, Side, Signature> : bound_shape<E, Context, Side, Signature> {};
template<FieldExpr E, class Context, SemanticSide Side, class Signature>
struct bound_shape<AverageExpr<E>, Context, Side, Signature> : bound_shape<E, Context, Side, Signature> {};

template<FieldExpr E, class Context, SemanticSide Side, class Signature>
struct bound_shape<MinusTraceExpr<E>, Context, Side, Signature>
   : bound_shape<E, Context, SemanticSide::Minus, Signature> {};
template<FieldExpr E, class Context, SemanticSide Side, class Signature>
struct bound_shape<PlusTraceExpr<E>, Context, Side, Signature>
   : bound_shape<E, Context, SemanticSide::Plus, Signature> {};
template<FieldExpr A, FieldExpr E, class Context, SemanticSide Side, class Signature>
struct bound_shape<UpwindExpr<A, E>, Context, Side, Signature> : bound_shape<E, Context, Side, Signature> {};

template<FieldShape Shape, class Context, SemanticSide Side, class Signature>
struct bound_shape<weak_form_detail::BoundaryUpwindZero<Shape>, Context, Side, Signature>
{
   static constexpr auto value = weak_form_traits::semantic_shape(Shape);
};

template<FieldExpr Adv, typename Interior, typename Exterior,
         class Context, SemanticSide Side, class Signature>
struct bound_shape<BoundaryUpwindExpr<Adv, Interior, Exterior>, Context, Side, Signature>
{
   static constexpr auto value = []
   {
      if constexpr (std::is_same_v<Interior, Empty>)
         return bound_shape<Exterior, Context, Side, Signature>::value;
      else if constexpr (std::is_same_v<Exterior, Empty>)
         return bound_shape<Interior, Context, Side, Signature>::value;
      else
         return weak_form_traits::combined_shape(ShapeOperation::Sum,
            bound_shape<Interior, Context, Side, Signature>::value,
            bound_shape<Exterior, Context, Side, Signature>::value);
   }();
};

template<class Obligation, class Context, SemanticSide Side = SemanticSide::Unqualified, class Signature = void>
struct shape_obligation;
template<ShapeOperation Op, class E, class Context, SemanticSide Side, class Signature>
struct shape_obligation<ShapeObligation<Op, E>, Context, Side, Signature>
{
   static constexpr auto shape = bound_shape<E, Context, Side, Signature>::value;
   static constexpr bool value = Op == ShapeOperation::ScalarIntegrand
      ? shape == SemanticShape::Scalar
      : Op == ShapeOperation::UpwindSelector ? shape == SemanticShape::Vector
      : Op == ShapeOperation::Transpose ? shape == SemanticShape::Matrix
      : Op == ShapeOperation::Gradient ? weak_form_traits::gradient_shape(shape) != SemanticShape::Unsupported
      : shape != SemanticShape::Unsupported;
};
template<ShapeOperation Op, class L, class R, class Context, SemanticSide Side, class Signature>
struct shape_obligation<ShapeObligation<Op, L, R>, Context, Side, Signature>
{ static constexpr bool value = weak_form_traits::combined_shape(Op,
   bound_shape<L, Context, Side, Signature>::value, bound_shape<R, Context, Side, Signature>::value) != SemanticShape::Unsupported; };

// A divergence occurrence constrains only its selected side. A value-only read
// of the same field on the other side must not acquire this extent obligation.
template<StaticString Name, SemanticSide OccurrenceSide, class Context,
         SemanticSide Side, class Signature>
struct shape_obligation<ShapeObligation<ShapeOperation::Divergence, NameTag<Name>,
   std::integral_constant<SemanticSide, OccurrenceSide>>, Context, Side, Signature>
{
   using Bound = bound_shape<FiniteElementField<Name>, Context,
      OccurrenceSide == SemanticSide::Unqualified ? Side : OccurrenceSide,
      std::conditional_t<OccurrenceSide == SemanticSide::Unqualified, Signature, void>>;
   using Space = typename Bound::Space;
   using Coordinates = typename Space::mesh_type::cell_type::physical_coordinates;
   static constexpr bool vector = is_vector_finite_element_space_v<Space>;
   static constexpr bool matching_dimension = num_comp_v<Space> == std::tuple_size_v<Coordinates>;
   static_assert(vector, "div requires a vector finite-element space.");
   static_assert(!vector || matching_dimension,
      "div requires one vector component per physical dimension.");
   static constexpr bool value = vector && matching_dimension;
};

template<class Context, SemanticSide Side = SemanticSide::Unqualified, class Signature = void, class... Obligations>
consteval void ValidateShapes(type_list<Obligations...>)
{
   static_assert((shape_obligation<Obligations, Context, Side, Signature>::value && ...),
      "Weak-form field shape obligations are incompatible with the selected finite element spaces.");
}
template<class Context, SemanticSide Side = SemanticSide::Unqualified, class Signature = void, class... Declarations>
consteval void ValidateDeclarations(type_list<Declarations...>)
{
   static_assert(((weak_form_traits::semantic_shape(Declarations::shape) ==
      bound_shape<NameTag<Declarations::name>, Context, Side, Signature>::value) && ...),
      "Weak-form declaration shape does not match its selected finite element space.");
}

template<class Context, SemanticSide Side, class Fields, class... Declarations>
consteval void ValidateRequiredDeclarations(type_list<Declarations...>)
{
   ([] {
      if constexpr (ContainsField<Declarations::name>(Fields{}))
         ValidateDeclarations<Context, Side>(type_list<Declarations>{});
   }(), ...);
}

// A field that occurs on both sides must have compatible symbolic shapes, but
// may use different basis orders and restriction types on those sides.
template<class Signature, class Context, class... Fields>
consteval void ValidateSideShapes(type_list<Fields...>)
{
   ([] {
      if constexpr (ContainsField<Fields::name>(concat_many_t<
         typename Signature::template active_inputs_on<SemanticSide::Minus>,
         typename Signature::template bound_inputs_on<SemanticSide::Minus>,
         typename Signature::template test_outputs_on<SemanticSide::Minus>>{}) &&
         ContainsField<Fields::name>(concat_many_t<
         typename Signature::template active_inputs_on<SemanticSide::Plus>,
         typename Signature::template bound_inputs_on<SemanticSide::Plus>,
         typename Signature::template test_outputs_on<SemanticSide::Plus>>{}))
         static_assert(bound_shape<NameTag<Fields::name>, Context, SemanticSide::Minus>::value ==
                       bound_shape<NameTag<Fields::name>, Context, SemanticSide::Plus>::value,
            "A field read or written on both facet sides must have compatible symbolic shapes.");
   }(), ...);
}

// Formal dimensions validate spaces without acquiring storage or reviving
// reads removed by an affine transformation.
template<class Sides, class Symbols, class Context>
consteval void ValidateDimensionSpaces()
{
   using Declarations = concat_t<typename Symbols::active_declarations,
                                 typename Symbols::test_declarations>;
   ValidateRequiredDeclarations<Context, SemanticSide::Unqualified,
      typename Sides::template on<SemanticSide::Unqualified>>(Declarations{});
   ValidateRequiredDeclarations<Context, SemanticSide::Minus,
      typename Sides::template on<SemanticSide::Minus>>(Declarations{});
   ValidateRequiredDeclarations<Context, SemanticSide::Plus,
      typename Sides::template on<SemanticSide::Plus>>(Declarations{});
}

template<class Signature, class Symbols = void, class Context>
consteval void ValidateSignature()
{
   using Declarations = typename decltype([] {
      if constexpr (std::is_void_v<Symbols>)
         return std::type_identity<concat_t<typename Signature::declared_inputs,
                                           typename Signature::declared_outputs>>{};
      else
         return std::type_identity<concat_t<typename Symbols::active_declarations,
                                           typename Symbols::test_declarations>>{};
   }())::type;
   using Current = concat_many_t<typename Signature::template active_inputs_on<SemanticSide::Unqualified>,
      typename Signature::template bound_inputs_on<SemanticSide::Unqualified>,
      typename Signature::template logical_outputs_on<SemanticSide::Unqualified>>;
   using Minus = concat_many_t<typename Signature::template active_inputs_on<SemanticSide::Minus>,
      typename Signature::template bound_inputs_on<SemanticSide::Minus>,
      typename Signature::template logical_outputs_on<SemanticSide::Minus>>;
   using Plus = concat_many_t<typename Signature::template active_inputs_on<SemanticSide::Plus>,
      typename Signature::template bound_inputs_on<SemanticSide::Plus>,
      typename Signature::template logical_outputs_on<SemanticSide::Plus>>;
   ValidateRequiredDeclarations<Context, SemanticSide::Unqualified, Current>(Declarations{});
   ValidateRequiredDeclarations<Context, SemanticSide::Minus, Minus>(Declarations{});
   ValidateRequiredDeclarations<Context, SemanticSide::Plus, Plus>(Declarations{});
   ValidateSideShapes<Signature, Context>(concat_many_t<Current, Minus, Plus>{});
   // Obligations containing explicit traces select their own sides. Bare
   // obligations are checked on every side used by the integrand.
   if constexpr (!is_empty_type_list_v<Current> || !is_empty_type_list_v<Minus>)
      ValidateShapes<Context, SemanticSide::Minus, Signature>(typename Signature::shape_obligations{});
   if constexpr (!is_empty_type_list_v<Plus>)
      ValidateShapes<Context, SemanticSide::Plus, Signature>(typename Signature::shape_obligations{});
}

// Borrow each registered field for topology and storage checks. Selection of
// types below must not construct a registry or acquire persistent vector data.
template<StaticString Name, class Context, class Batch, class... Reads>
void ValidateSelectedField(
   const Context& context, const Batch& batch, type_list<Reads...>)
{
   if constexpr (Context::template HasFiniteElementField<Name>())
   {
      decltype(auto) space = RestrictFiniteElementFieldSpace<
         FiniteElementFieldKey<Name>, restricted_execution_domain_kind_v<Batch>,
         restricted_execution_domain_index_v<Batch>>(
            context.template GetFiniteElementField<Name>().space, batch);

      ([&] {
         if constexpr (Reads::name == Name &&
                       Reads::storage_role == FieldStorageRole::Bound &&
                       Context::template HasBoundField<Reads::name>())
         {
            execution_context_detail::ValidateBoundStorageExtent(
               execution_context_detail::RequiredVolumeSpace<Reads::side>(space),
               context.template GetBoundField<Reads::name>());
         }
      }(), ...);
   }
}

template<class Symbols, class Context, class Batch, class Dimensions, class... Signatures>
void ValidateExecutionBatch(
   const Context& context, const Batch& batch,
   type_list<Signatures...>, Dimensions)
{
   using Requirements = typename weak_form_execution_detail::signature_requirements<
      type_list<Signatures...>>::type;
   using Outputs = weak_form_traits::merge_sides<typename Signatures::logical_output_sides...>;
   using DimensionReads = typename weak_form_execution_detail::read_fields<
      typename Dimensions::all, FieldStorageRole::Input, FieldSide::Current>::type;
   using OutputSpaces = typename weak_form_execution_detail::write_fields<
      typename Outputs::all, FieldSide::Current>::type;
   using Selection = FieldAccessRequirements<
      unique_t<concat_t<typename Requirements::reads, DimensionReads>>,
      unique_t<concat_t<typename Requirements::writes, OutputSpaces>>>;

   using Selected = decltype(MakeRestrictedWeakFormContext<Selection>(context, batch));
   (ValidateSignature<Signatures, Symbols, Selected>(), ...);
   ValidateDimensionSpaces<Dimensions, Symbols, Selected>();
   if constexpr (is_selected_interior_face_execution_domain_v<Batch>)
   {
      (ValidateNonconformingGlobalInteriorFacetTransformSupport<
         Batch, Signatures, Selected>(), ...);
   }

   using Names = decltype([]<class... Fields>(type_list<Fields...>) {
      return unique_t<type_list<NameTag<Fields::name>...>>{};
   }(concat_t<typename Selection::reads, typename Selection::writes>{}));
   [&]<class... Fields>(type_list<Fields...>) {
      (ValidateSelectedField<Fields::name>(context, batch, typename Requirements::reads{}), ...);
   }(Names{});
}

template<class Signature, generic_operator_detail::GenericOperatorDomainKind DomainKind,
         StaticString Name, class Context>
void ValidateDomain(const Context& context)
{
   using generic_operator_detail::GenericOperatorDomainKind;
   using Retained = typename Signature::symbols::declarations::retained_callable_input_relations;
   using CellDimensions = typename weak_form_traits::callable_relations_for<Retained, Cells<Name>>::type;
   using InteriorDimensions = typename weak_form_traits::callable_relations_for<Retained, InteriorFacets<Name>>::type;
   using BoundaryDimensions = typename weak_form_traits::callable_relations_for<Retained, BoundaryFacets<Name>>::type;

   if constexpr (DomainKind == GenericOperatorDomainKind::Mesh)
   {
      using Signatures = typename weak_form_traits::domain_signatures<
         typename Signature::integrands, Name, SemanticEntity::Unspecified, true>::type;
      ForEachCellExecutionDomain(context, Cells<Name>{}, [&](const auto& batch) {
         ValidateExecutionBatch<typename Signature::symbols>(context, batch, Signatures{},
            weak_form_traits::merge_sides<CellDimensions, InteriorDimensions, BoundaryDimensions>{});
      });
   }
   else
   {
      using CellSignatures = typename weak_form_traits::domain_signatures<
         typename Signature::integrands, Name, SemanticEntity::Cell, true>::type;
      using InteriorSignatures = typename weak_form_traits::domain_signatures<
         typename Signature::integrands, Name, SemanticEntity::InteriorFacet, true>::type;
      using BoundarySignatures = typename weak_form_traits::domain_signatures<
         typename Signature::integrands, Name, SemanticEntity::BoundaryFacet, true>::type;

      if constexpr (!is_empty_type_list_v<CellSignatures> || !is_empty_type_list_v<typename CellDimensions::all>)
      {
         ForEachCellExecutionDomain(context, Cells<Name>{}, [&](const auto& batch) {
            ValidateExecutionBatch<typename Signature::symbols>(context, batch, CellSignatures{}, CellDimensions{});
         });
      }
      if constexpr (!is_empty_type_list_v<InteriorSignatures> || !is_empty_type_list_v<typename InteriorDimensions::all>)
      {
         static_assert(Context::template HasInteriorFaceDomain<Name>(),
            "InteriorFacets<Name>: PartitionIntegrationDomain has no interior face parts; "
            "partition domains do not fall back to local-facet traversal.");
         if constexpr (Context::template HasInteriorFaceDomain<Name>())
         {
            ForEachInteriorFaceExecutionDomain(context, InteriorFacets<Name>{}, [&](const auto& batch) {
               ValidateExecutionBatch<typename Signature::symbols>(context, batch, InteriorSignatures{}, InteriorDimensions{});
            });
         }
      }
      if constexpr (!is_empty_type_list_v<BoundarySignatures> || !is_empty_type_list_v<typename BoundaryDimensions::all>)
      {
         static_assert(Context::template HasBoundaryFaceDomain<Name>(),
            "BoundaryFacets<Name>: PartitionIntegrationDomain has no boundary face parts; "
            "partition domains do not fall back to local-facet traversal.");
         if constexpr (Context::template HasBoundaryFaceDomain<Name>())
         {
            ForEachBoundaryFaceExecutionDomain(context, BoundaryFacets<Name>{}, [&](const auto& batch) {
               ValidateExecutionBatch<typename Signature::symbols>(context, batch, BoundarySignatures{}, BoundaryDimensions{});
            });
         }
      }
   }
}

} // namespace gendil::weak_form_validation_detail

namespace gendil
{

template<class WeakForm, class WFContext>
void ValidateWeakFormContext(const WeakForm&, const WFContext& wf_ctx)
{
   using namespace generic_operator_detail;
   constexpr auto DomainKind = generic_operator_domain_kind_v<WeakForm, WFContext>;
   if constexpr (DomainKind == GenericOperatorDomainKind::Invalid)
   {
      static_assert(DomainKind != GenericOperatorDomainKind::Invalid,
         "Weak form integrand requires an integration domain registered "
         "under its domain name.");
   }
   else if constexpr (DomainKind == GenericOperatorDomainKind::Mixed)
   {
      static_assert(DomainKind != GenericOperatorDomainKind::Mixed,
         "GenericOperator does not support mixing MeshIntegrationDomain and "
         "PartitionIntegrationDomain entries in one weak form.");
   }
   else
   {
      using Signature = form_signature_t<WeakForm>;
      static_assert(Signature::valid,
         "Weak form has invalid result, output, field, or entity semantics.");
      if constexpr (Signature::valid)
      {
         static_assert(Signature::result_kind == FormResultKind::Residual,
            "Local residual execution does not accept scalar forms.");
         if constexpr (Signature::result_kind == FormResultKind::Residual)
         {
            using Retained = typename Signature::symbols::declarations::retained_callable_input_relations;
            // Keep dimension-only domains in validation, not numerical scheduling.
            using DomainNames = decltype([]<class... Relations>(type_list<Relations...>) {
               return unique_t<concat_t<generic_operator_domain_names_t<WeakForm>,
                  type_list<NameTag<Relations::key_type::name>...>>>{};
            }(Retained{}));
            ForEachGenericOperatorDomainName(DomainNames{}, [&](auto name) {
               weak_form_validation_detail::ValidateDomain<Signature, DomainKind, decltype(name)::name>(wf_ctx);
            });
         }
      }
   }
}

} // namespace gendil

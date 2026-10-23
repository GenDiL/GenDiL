// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/fieldrequirements.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/signatures.hpp"

namespace gendil {
namespace weak_form_execution_detail
{
template<class List, FieldStorageRole Role, FieldSide Side>
struct read_fields;
template<class... Fields, FieldStorageRole Role, FieldSide Side>
struct read_fields<type_list<Fields...>, Role, Side>
{
   using type = type_list<FieldRead<Fields::name, Role, Side>...>;
};

template<class List, FieldSide Side>
struct write_fields;
template<class... Fields, FieldSide Side>
struct write_fields<type_list<Fields...>, Side>
{
   using type = type_list<FieldWrite<Fields::name, Side>...>;
};

template<class Signature, SemanticSide Side, FieldSide EntitySide>
using reads_on_t = concat_t<
   typename read_fields<typename Signature::template active_inputs_on<Side>,
                        FieldStorageRole::Input, EntitySide>::type,
   typename read_fields<typename Signature::template bound_inputs_on<Side>,
                        FieldStorageRole::Bound, EntitySide>::type>;

template<class Signature, SemanticSide Side, FieldSide EntitySide>
using writes_on_t = typename write_fields<
   typename Signature::template test_outputs_on<Side>, EntitySide>::type;

template<class Signature>
using reads_t = concat_many_t<
   reads_on_t<Signature, SemanticSide::Unqualified, FieldSide::Current>,
   reads_on_t<Signature, SemanticSide::Minus, FieldSide::Minus>,
   reads_on_t<Signature, SemanticSide::Plus, FieldSide::Plus>>;

template<class Signature>
using writes_t = concat_many_t<
   writes_on_t<Signature, SemanticSide::Unqualified, FieldSide::Current>,
   writes_on_t<Signature, SemanticSide::Minus, FieldSide::Minus>,
   writes_on_t<Signature, SemanticSide::Plus, FieldSide::Plus>>;

// Binding unions contain no numerical storage or preparation objects.
template<class Signatures> struct signature_requirements;
template<class... Signatures>
struct signature_requirements<type_list<Signatures...>>
{
   using reads = unique_t<concat_many_t<reads_t<Signatures>...>>;
   using writes = unique_t<concat_many_t<writes_t<Signatures>...>>;
   using type = FieldAccessRequirements<reads, writes>;
   using declared_inputs = unique_t<concat_many_t<typename Signatures::declared_inputs...>>;
   using declared_outputs = unique_t<concat_many_t<typename Signatures::declared_outputs...>>;
   // Reuse a current input/output row only for the existing conservative
   // single-pair case. Plus-only reads must not cause a current-cell gather.
   static constexpr bool can_reuse_current_element_dofs = sizeof...(Signatures) != 0 &&
      (weak_form_traits::has_single_field_pair<Signatures>::value && ...) &&
      weak_form_traits::list_size<declared_inputs>::value == 1 &&
      weak_form_traits::list_size<declared_outputs>::value == 1 &&
      ((!is_empty_type_list_v<typename Signatures::template active_inputs_on<SemanticSide::Unqualified>> ||
        !is_empty_type_list_v<typename Signatures::template active_inputs_on<SemanticSide::Minus>>) || ...);
};

template<StaticString Name, class Tuple, SemanticEntity Entity = SemanticEntity::Unspecified, size_t... I>
consteval auto DomainTermIndices(std::index_sequence<I...>)
{
   // Index carriers are compile-time values; inspection never copies terms.
   return std::tuple_cat([]<size_t J>() {
      using Term = std::remove_cvref_t<std::tuple_element_t<J, Tuple>>;
      if constexpr (!weak_form_traits::is_zero_expr_v<typename Term::expression_type> &&
         Term::domain_type::name == Name &&
         (Entity == SemanticEntity::Unspecified ||
          weak_form_traits::domain_semantic_entity_v<typename Term::domain_type> == Entity))
         return std::tuple{std::integral_constant<size_t, J>{}};
      else return std::tuple{};
   }.template operator()<I>()...);
}

} // namespace weak_form_execution_detail

// Lower the resolved signature; no AST traversal is performed here.
template<class Signature>
using lower_signature_requirements_t = FieldAccessRequirements<
   unique_t<weak_form_execution_detail::reads_t<Signature>>,
   std::conditional_t<Signature::result_kind == FormResultKind::Scalar,
      type_list<>, unique_t<weak_form_execution_detail::writes_t<Signature>>>>;

// Probe overlays select a semantic side, then bind by name. Side selection is
// still owned by the facet routine and never enters endpoint-map identity.
template<class Signature, SemanticSide Side>
using facet_side_requirements_t = FieldAccessRequirements<
   unique_t<weak_form_execution_detail::reads_on_t<Signature, Side, FieldSide::Current>>,
   unique_t<weak_form_execution_detail::writes_on_t<Signature, Side, FieldSide::Current>>>;

} // namespace gendil

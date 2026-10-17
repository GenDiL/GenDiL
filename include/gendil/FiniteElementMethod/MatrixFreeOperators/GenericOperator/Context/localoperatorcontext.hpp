// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontextcommon.hpp"

namespace gendil
{

template<class QD>
struct LocalFacetQuadratureData
{
   QD qd;

   // Cell-owned traversal selects the current or neighbor face from this
   // all-local-face tuple. Global face batches prepare each side separately.
   GENDIL_HOST_DEVICE
   constexpr const QD& MinusSide() const { return qd; }

   GENDIL_HOST_DEVICE
   constexpr const QD& PlusSide() const { return qd; }
};

template<
   class IntegrationRule,
   class MeshQDMap,
   class FEQDMap,
   class MeshFacetQDMap,
   class FEFacetQDMap,
   class FormSignature>
struct OperatorContext
{
   using integration_rule_type = IntegrationRule;
   using form_signature = FormSignature;
   template<class Integrand> using signature_for =
      typename operator_context_detail::signature_for<FormSignature, Integrand>::type;
   using face_integration_rules_type =
      decltype(GetFaceIntegrationRules(IntegrationRule{}));

   IntegrationRule int_rule;
   face_integration_rules_type face_int_rules;
   MeshQDMap mesh_qd;
   FEQDMap fe_qd;
   MeshFacetQDMap mesh_facet_qd;
   FEFacetQDMap fe_facet_qd;

   GENDIL_HOST_DEVICE
   const IntegrationRule& GetIntegrationRule() const { return int_rule; }

   GENDIL_HOST_DEVICE
   const face_integration_rules_type& GetFacetIntegrationRules() const
   {
      return face_int_rules;
   }

   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetMeshQuadData() const
   {
      return mesh_qd.template Get<DomainKey<Name>>();
   }

   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetFiniteElementQuadData() const
   {
      return fe_qd.template Get<FiniteElementFieldKey<Name>>();
   }

   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetMeshFacetQuadData() const
   {
      return mesh_facet_qd.template Get<DomainKey<Name>>();
   }

   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetFiniteElementFacetQuadData() const
   {
      return fe_facet_qd.template Get<FiniteElementFieldKey<Name>>();
   }
};

template<class IntegrationRule, class Domain>
constexpr auto MakeMeshFacetQuadData(
   const CellIntegrationDomain<Domain>& domain)
{
   using FaceIRs = decltype(GetFaceIntegrationRules(IntegrationRule{}));
   const auto& mesh = GetCellIntegrationDomainMesh(domain);
   using Mesh = std::remove_cvref_t<decltype(mesh)>;
   operator_context_detail::ValidateIntegrationRuleGeometry<
      IntegrationRule,
      Mesh>();
   auto qd = MakeMeshFaceQuadData<Mesh>(FaceIRs{});
   using QD = std::remove_cvref_t<decltype(qd)>;
   // Local traversal may visit any local face of the active domain Cell.
   return LocalFacetQuadratureData<QD>{ static_cast<QD>(qd) };
}

template<class IntegrationRule, class SpaceView>
constexpr auto MakeVolumeFiniteElementFacetQuadData(const SpaceView& space)
{
   using Space   = std::remove_cvref_t<SpaceView>;
   using FE      = typename Space::finite_element_type;
   using Shape   = typename FE::shape_functions;
   using FaceIRs = decltype(GetFaceIntegrationRules(IntegrationRule{}));
   return MakeFaceDofToQuad<Shape, FaceIRs>();
}

template<class IntegrationRule, class SpaceView>
constexpr auto MakeFiniteElementFacetQuadData(const SpaceView& space)
{
   using Space = std::remove_cvref_t<SpaceView>;
   if constexpr (is_cell_finite_element_space_v<Space>)
   {
      // Local traversal may select any local face. Global face-domain contexts
      // use side-selected qdata for the exact minus/plus face family.
      auto qd =
         MakeVolumeFiniteElementFacetQuadData<IntegrationRule>(space);
      using QD = std::remove_cvref_t<decltype(qd)>;
      return LocalFacetQuadratureData<QD>{ static_cast<QD>(qd) };
   }
   else
   {
      static_assert(
         dependent_false_v<Space>,
         "MakeFiniteElementFacetQuadData builds local/cell-owned facet qdata "
         "from a selected homogeneous finite element space. Mixed finite "
         "element spaces must be restricted to a selected cell batch; global "
         "facet contexts use side-selected field bindings.");
   }
}

template<class IR, class DomainEntry>
constexpr auto MakeLocalFacetMeshQuadDataEntryTuple(const DomainEntry& e)
{
   using Key = typename DomainEntry::key_type;
   auto qd   = MakeMeshFacetQuadData<IR>(e.value);
   using QD  = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

template<class IR, class FEFieldEntry>
constexpr auto MakeLocalFacetFiniteElementQuadDataEntryTuple(
   const FEFieldEntry& e)
{
   using Key       = typename FEFieldEntry::key_type;
   const auto& fev = e.value;
   auto qd         = MakeFiniteElementFacetQuadData<IR>(fev.space);
   using QD        = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

namespace operator_context_detail
{
template<class DomainEntry, bool Facet>
consteval bool PreparesLocalEntity()
{
   using Domain = std::remove_cvref_t<decltype(std::declval<DomainEntry>().value.domain)>;
   // Partition faces are prepared by their own selected face batches. Mesh
   // batches retain the fused cell/local-facet preparation route.
   return !Facet || !is_partitioned_selected_execution_domain_v<Domain>;
}

template<class Context, class Signature>
inline constexpr bool signature_in_context_v = [] {
   using Domains = std::remove_cvref_t<decltype(std::declval<Context>().domains)>;
   constexpr auto Name = Signature::domain_type::name;
   if constexpr (!Domains::template Contains<DomainKey<Name>>()) return false;
   else
   {
      using EntryValue = std::remove_cvref_t<decltype(std::declval<Domains>().template Get<DomainKey<Name>>())>;
      using Domain = std::remove_cvref_t<decltype(std::declval<EntryValue>().domain)>;
      return is_cells_domain<typename Signature::domain_type>::value ||
             !is_partitioned_selected_execution_domain_v<Domain>;
   }
}();

template<class Context, bool Boundary, class Signatures> struct required_entity_fields;
template<class Context, bool Boundary, class... Signatures>
struct required_entity_fields<Context, Boundary, type_list<Signatures...>>
{
   using type = concat_many_t<std::conditional_t<
      !Signatures::is_zero && signature_in_context_v<Context, Signatures> &&
      (is_cells_domain<typename Signatures::domain_type>::value != Boundary),
      concat_many_t<typename Signatures::active_inputs, typename Signatures::bound_inputs,
                    typename Signatures::test_outputs>, type_list<>>...>;
};
template<StaticString Name, class... Fields>
consteval bool ContainsName(type_list<Fields...>)
{ return ((Fields::name == Name) || ...); }

template<class Rule, bool Boundary, class Fields, class Entry>
auto PrepareRequiredField(const Entry& entry)
{
   if constexpr (ContainsName<Entry::key_type::name>(Fields{}))
   {
      if constexpr (Boundary) return MakeLocalFacetFiniteElementQuadDataEntryTuple<Rule>(entry);
      else return MakeFiniteElementQuadDataEntryTuple<Rule>(entry);
   }
   else return std::tuple<>{};
}

template<class DomainEntry, bool Boundary, class... Signatures>
consteval bool RequiresEntity(type_list<Signatures...>)
{
   return PreparesLocalEntity<DomainEntry, Boundary>() &&
      ((!Signatures::is_zero && (DomainEntry::key_type::name == Signatures::domain_type::name) &&
      (is_cells_domain<typename Signatures::domain_type>::value != Boundary)) || ...);
}
template<class Rule, bool Boundary, class Signatures, class Entry>
auto PrepareRequiredDomain(const Entry& entry)
{
   if constexpr (RequiresEntity<Entry, Boundary>(Signatures{}))
   {
      if constexpr (Boundary) return MakeLocalFacetMeshQuadDataEntryTuple<Rule>(entry);
      else return MakeMeshQuadDataEntryTuple<Rule>(entry);
   }
   else return std::tuple<>{};
}
// Signature retains complete-form roles; Signatures limits preparation to the
// original terms executed by this batch or assembly phase.
template<class Signature, class Signatures, class Context, class Rule>
auto MakeOperatorContextImpl(const Context& context, const Rule& rule)
{
   using IR = std::remove_cvref_t<Rule>;
   using CellFields = typename operator_context_detail::required_entity_fields<Context, false, Signatures>::type;
   using BoundaryFields = typename operator_context_detail::required_entity_fields<Context, true, Signatures>::type;
   auto cell_fields = std::apply([](const auto&... entries) {
      return ToStaticMap(std::tuple_cat(
         operator_context_detail::PrepareRequiredField<IR, false, CellFields>(entries)...));
   }, context.fe_fields.entries);
   auto cell_mesh = std::apply([](const auto&... entries) {
      return ToStaticMap(std::tuple_cat(
         operator_context_detail::PrepareRequiredDomain<IR, false, Signatures>(entries)...));
   }, context.domains.entries);
   // Keep the established cell-only capture shape: no face-rule carrier or
   // facet maps are constructed for a selected batch without facet terms.
   constexpr bool has_facets = []<class... Entries>(const StaticMap<Entries...>*) {
      return (operator_context_detail::RequiresEntity<Entries, true>(Signatures{}) || ...);
   }(static_cast<const std::remove_cvref_t<decltype(context.domains)>*>(nullptr));
   if constexpr (!has_facets)
      return CellOnlyOperatorContext<IR, decltype(cell_mesh), decltype(cell_fields), Signature>{
         rule, std::move(cell_mesh), std::move(cell_fields)};
   else
   {
      auto boundary_fields = std::apply([](const auto&... entries) {
         return ToStaticMap(std::tuple_cat(
            operator_context_detail::PrepareRequiredField<IR, true, BoundaryFields>(entries)...));
      }, context.fe_fields.entries);
      auto boundary_mesh = std::apply([](const auto&... entries) {
         return ToStaticMap(std::tuple_cat(
            operator_context_detail::PrepareRequiredDomain<IR, true, Signatures>(entries)...));
      }, context.domains.entries);
      return OperatorContext<IR, decltype(cell_mesh), decltype(cell_fields),
         decltype(boundary_mesh), decltype(boundary_fields), Signature>{rule, GetFaceIntegrationRules(IR{}),
         std::move(cell_mesh), std::move(cell_fields), std::move(boundary_mesh), std::move(boundary_fields)};
   }
}

} // namespace operator_context_detail

// The metadata has no runtime storage. Basis and geometry maps contain only
// the selected domains and the cell/facet fields that actually use them.
template<FormExpr Form, class Context, class Rule>
auto MakeOperatorContext(const Form&, const Context& context, const Rule& rule)
{
   using Signature = form_signature_t<Form>;
   return operator_context_detail::MakeOperatorContextImpl<
      Signature, typename Signature::integrands>(context, rule);
}

// Selected integrands may be owned or borrowed. Resolve their existing signatures
// from the complete form, without building or analyzing a combined expression.
template<class FormSignature, class... Integrands, class Context, class Rule>
auto MakeOperatorContext(const std::tuple<Integrands...>&,
                         const Context& context, const Rule& rule)
{
   using Signatures = type_list<typename operator_context_detail::signature_lookup<
      std::remove_cvref_t<Integrands>, typename FormSignature::integrands>::type...>;
   return operator_context_detail::MakeOperatorContextImpl<
      FormSignature, Signatures>(context, rule);
}

} // namespace gendil

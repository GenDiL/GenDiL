// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/selectedexecutiondomain.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontextcommon.hpp"

namespace gendil
{

template<class QD>
struct BoundaryFacetQData
{
   QD qd;

   // Boundary face MinusSide is the adjacent/interior Cell side. The boundary
   // face normal is outward from that Cell, and there is no PlusSide finite
   // element space.
   GENDIL_HOST_DEVICE
   constexpr const QD& MinusSide() const { return qd; }
};

template<class MinusQD, class PlusQD = MinusQD>
struct InteriorFacetQuadratureData
{
   MinusQD minus_qd;
   PlusQD plus_qd;

   GENDIL_HOST_DEVICE
   constexpr const MinusQD& MinusSide() const { return minus_qd; }

   GENDIL_HOST_DEVICE
   constexpr const PlusQD& PlusSide() const { return plus_qd; }
};

// Global facet contexts side-select exactly one local-face family from a
// concrete face mesh. That is only valid when the face-info side types expose
// static local-face indices; otherwise MakeFacetOperatorContext must reject the
// mesh instead of guessing from runtime topology.
template<class T, class = void>
struct global_face_mesh_has_static_face_family : std::false_type {};

template<class T>
struct global_face_mesh_has_static_face_family<
   T,
   std::void_t<
      typename std::remove_cvref_t<T>::face_info_type,
      typename std::remove_cvref_t<T>::face_info_type::minus_side_type,
      typename std::remove_cvref_t<T>::face_info_type::plus_side_type,
      typename std::remove_cvref_t<T>::face_info_type::
         minus_side_type::local_face_index_type,
      typename std::remove_cvref_t<T>::face_info_type::
         plus_side_type::local_face_index_type>> : std::true_type {};

template<class T>
inline constexpr bool global_face_mesh_has_static_face_family_v =
   global_face_mesh_has_static_face_family<std::remove_cvref_t<T>>::value;

template<class FaceMesh>
using global_face_mesh_minus_side_type_t =
   typename std::remove_cvref_t<FaceMesh>::face_info_type::minus_side_type;

template<class FaceMesh>
using global_face_mesh_plus_side_type_t =
   typename std::remove_cvref_t<FaceMesh>::face_info_type::plus_side_type;

template<class FaceMesh>
inline constexpr size_t global_face_mesh_minus_local_face_index_v =
   static_cast<size_t>(
      global_face_mesh_minus_side_type_t<
         FaceMesh>::local_face_index_type::value);

template<class FaceMesh>
inline constexpr size_t global_face_mesh_plus_local_face_index_v =
   static_cast<size_t>(
      global_face_mesh_plus_side_type_t<
         FaceMesh>::local_face_index_type::value);

// Facet contractions index the local-face tuple. A global batch prepares only
// its statically selected face family; other tuple entries carry no data.
template<size_t SelectedIndex, size_t I, class QD>
constexpr auto MakeSelectedFaceQDataTupleElement(const QD& qd)
{
   if constexpr (I == SelectedIndex)
   {
      return QD{ qd };
   }
   else
   {
      return Empty{};
   }
}

template<size_t SelectedIndex, class QD, size_t... I>
constexpr auto MakeSelectedFaceQDataTuple(
   const QD& qd,
   std::index_sequence<I...>)
{
   return std::tuple{
      MakeSelectedFaceQDataTupleElement<SelectedIndex, I>(qd)... };
}

template<class FaceIRs, size_t SelectedIndex, class QD>
constexpr auto MakeSelectedFaceQDataTuple(const QD& qd)
{
   static_assert(
      SelectedIndex < std::tuple_size_v<FaceIRs>,
      "Selected global face local-face index is out of bounds for the "
      "facet integration-rule tuple.");
   return MakeSelectedFaceQDataTuple<SelectedIndex>(
      qd,
      std::make_index_sequence<std::tuple_size_v<FaceIRs>>{});
}

template<class IntegrationRule, class MeshFacetQDMap, class FEFacetQDMap, class FormSignature>
struct FacetOperatorContext
{
   using integration_rule_type = IntegrationRule;
   using form_signature = FormSignature;
   template<class Integrand> using signature_for =
      typename operator_context_detail::signature_for<FormSignature, Integrand>::type;
   using face_integration_rules_type =
      decltype(GetFaceIntegrationRules(IntegrationRule{}));

   IntegrationRule int_rule;
   face_integration_rules_type face_int_rules;
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

template<class IntegrationRule, size_t SelectedIndex, class CellMesh>
constexpr auto MakeSelectedMeshFacetQuadDataForSide(
   const CellMesh& /*mesh*/)
{
   using Mesh = std::remove_cvref_t<CellMesh>;
   operator_context_detail::ValidateIntegrationRuleGeometry<
      IntegrationRule,
      Mesh>();
   using FaceIRs = decltype(GetFaceIntegrationRules(IntegrationRule{}));
   using FaceIR = std::tuple_element_t<SelectedIndex, FaceIRs>;
   using QD = typename Mesh::cell_type::template QuadData<FaceIR>;
   return MakeSelectedFaceQDataTuple<FaceIRs, SelectedIndex>(QD{});
}

template<class IntegrationRule, size_t SelectedIndex, class VolumeSpace>
constexpr auto MakeSelectedFiniteElementFacetQuadDataForSide(
   const VolumeSpace& /*space*/)
{
   using Space = std::remove_cvref_t<VolumeSpace>;
   using FE = typename Space::finite_element_type;
   using Shape = typename FE::shape_functions;
   using FaceIRs = decltype(GetFaceIntegrationRules(IntegrationRule{}));
   using FaceIR = std::tuple_element_t<SelectedIndex, FaceIRs>;
   auto qd = MakeDofToQuad<Shape, FaceIR>();
   using QD = std::remove_cvref_t<decltype(qd)>;
   return MakeSelectedFaceQDataTuple<FaceIRs, SelectedIndex>(
      static_cast<QD>(qd));
}

template<class IntegrationRule, bool NeedPlus = true, class FaceDomain>
constexpr auto MakeGlobalFacetMeshQuadData(const FaceDomain& face_domain)
{
   using Domain = std::remove_cvref_t<FaceDomain>;
   using Mesh = std::remove_cvref_t<decltype(face_domain.GetFaceMesh())>;
   static_assert(
      global_face_mesh_has_static_face_family_v<Mesh>,
      "MakeFacetOperatorContext requires a global face execution domain whose "
      "concrete face mesh type statically fixes its minus/plus local face "
      "family.");

   constexpr size_t MinusI = global_face_mesh_minus_local_face_index_v<Mesh>;
   constexpr size_t PlusI = global_face_mesh_plus_local_face_index_v<Mesh>;

   if constexpr (is_selected_boundary_face_execution_domain_v<Domain>)
   {
      auto minus_qd =
         MakeSelectedMeshFacetQuadDataForSide<IntegrationRule, MinusI>(
            face_domain.GetCellMesh());
      using MinusQD = std::remove_cvref_t<decltype(minus_qd)>;
      return BoundaryFacetQData<MinusQD>{ static_cast<MinusQD>(minus_qd) };
   }
   else if constexpr (is_selected_interior_face_execution_domain_v<Domain>)
   {
      auto minus_qd =
         MakeSelectedMeshFacetQuadDataForSide<IntegrationRule, MinusI>(
            face_domain.GetMinusCellMesh());
      auto plus_qd = [&]() {
         if constexpr (NeedPlus)
            return MakeSelectedMeshFacetQuadDataForSide<IntegrationRule, PlusI>(
               face_domain.GetPlusCellMesh());
         else return Empty{};
      }();
      using MinusQD = std::remove_cvref_t<decltype(minus_qd)>;
      using PlusQD = std::remove_cvref_t<decltype(plus_qd)>;
      return InteriorFacetQuadratureData<MinusQD, PlusQD>{
         static_cast<MinusQD>(minus_qd),
         static_cast<PlusQD>(plus_qd) };
   }
   else
   {
      static_assert(
         dependent_false_v<Domain>,
         "MakeFacetOperatorContext requires a boundary or interior face "
         "execution batch.");
   }
}

template<class IntegrationRule, bool NeedMinus = true, bool NeedPlus = true, class FieldBinding>
constexpr auto MakeGlobalFacetFiniteElementQuadData(
   const FieldBinding& field_binding)
{
   using Field = std::remove_cvref_t<FieldBinding>;

   if constexpr (is_boundary_face_field_binding_v<Field>)
   {
      using Mesh = std::remove_cvref_t<decltype(field_binding.GetFaceMesh())>;
      static_assert(
         global_face_mesh_has_static_face_family_v<Mesh>,
         "MakeFacetOperatorContext requires a boundary face field binding "
         "whose concrete face mesh type statically fixes its local face family.");
      constexpr size_t MinusI = global_face_mesh_minus_local_face_index_v<Mesh>;
      auto minus_qd = [&]() {
         if constexpr (NeedMinus)
            return MakeSelectedFiniteElementFacetQuadDataForSide<IntegrationRule, MinusI>(
               field_binding.GetMinusFiniteElementSpace());
         else return Empty{};
      }();
      using MinusQD = std::remove_cvref_t<decltype(minus_qd)>;
      return BoundaryFacetQData<MinusQD>{ static_cast<MinusQD>(minus_qd) };
   }
   else if constexpr (is_interior_face_field_binding_v<Field>)
   {
      using Mesh = std::remove_cvref_t<decltype(field_binding.GetFaceMesh())>;
      static_assert(
         global_face_mesh_has_static_face_family_v<Mesh>,
         "MakeFacetOperatorContext requires an interior face field binding "
         "whose concrete face mesh type statically fixes its minus/plus local "
         "face family.");
      constexpr size_t MinusI = global_face_mesh_minus_local_face_index_v<Mesh>;
      constexpr size_t PlusI = global_face_mesh_plus_local_face_index_v<Mesh>;
      auto minus_qd = [&]() {
         if constexpr (NeedMinus)
            return MakeSelectedFiniteElementFacetQuadDataForSide<IntegrationRule, MinusI>(
               field_binding.GetMinusFiniteElementSpace());
         else return Empty{};
      }();
      auto plus_qd = [&]() {
         if constexpr (NeedPlus)
            return MakeSelectedFiniteElementFacetQuadDataForSide<IntegrationRule, PlusI>(
               field_binding.GetPlusFiniteElementSpace());
         else return Empty{};
      }();
      using MinusQD = std::remove_cvref_t<decltype(minus_qd)>;
      using PlusQD = std::remove_cvref_t<decltype(plus_qd)>;
      return InteriorFacetQuadratureData<MinusQD, PlusQD>{
         static_cast<MinusQD>(minus_qd),
         static_cast<PlusQD>(plus_qd) };
   }
   else
   {
      static_assert(
         dependent_false_v<Field>,
         "MakeGlobalFacetFiniteElementQuadData requires a boundary or "
         "interior face field binding. Volume finite "
         "element spaces are valid only in the local/cell-owned facet path.");
   }
}

namespace operator_context_detail {
template<class Signature, class FaceDomain>
inline constexpr bool selected_facet_signature_v =
   !Signature::is_zero && Signature::domain_type::name == FaceDomain::domain_name &&
   (is_selected_boundary_face_execution_domain_v<FaceDomain>
      ? is_boundary_facets_domain<typename Signature::domain_type>::value
      : is_interior_facets_domain<typename Signature::domain_type>::value);

template<StaticString Name, class... Fields>
consteval bool HasFacetField(type_list<Fields...>)
{ return ((Name == Fields::name) || ...); }

template<StaticString Name, SemanticSide Side, class FaceDomain, class... Signatures>
consteval bool RequiredFacetField(type_list<Signatures...>)
{
   return ((selected_facet_signature_v<Signatures, FaceDomain> &&
      HasFacetField<Name>(concat_many_t<
         typename Signatures::template active_inputs_on<Side>,
         typename Signatures::template bound_inputs_on<Side>,
         typename Signatures::template test_outputs_on<Side>>{})) || ...);
}

template<class FaceDomain, class... Signatures>
consteval bool RequiresPlusGeometry(type_list<Signatures...>)
{
   return ((selected_facet_signature_v<Signatures, FaceDomain> &&
      Signatures::template geometry_on<SemanticSide::Plus> != GeometryRequirement::None) || ...);
}

template<class IR, class Signatures, class FaceDomain, class EntryType>
constexpr auto PrepareGlobalFacetField(const EntryType& entry)
{
   constexpr auto Name = EntryType::key_type::name;
   constexpr bool Minus = RequiredFacetField<Name, SemanticSide::Unqualified, FaceDomain>(Signatures{}) ||
                          RequiredFacetField<Name, SemanticSide::Minus, FaceDomain>(Signatures{});
   constexpr bool Plus = RequiredFacetField<Name, SemanticSide::Plus, FaceDomain>(Signatures{});
   if constexpr (Minus || Plus)
   {
      auto qd = MakeGlobalFacetFiniteElementQuadData<IR, Minus, Plus>(entry.value.space);
      return std::tuple{Entry<typename EntryType::key_type, decltype(qd)>{std::move(qd)}};
   }
   else return std::tuple{};
}

template<class IR, class Signatures, class FaceDomain, class EntryType>
constexpr auto PrepareGlobalFacetMesh(const EntryType& entry, const FaceDomain& face_domain)
{
   if constexpr (EntryType::key_type::name == FaceDomain::domain_name &&
      []<class... Terms>(type_list<Terms...>) {
         return (selected_facet_signature_v<Terms, FaceDomain> || ...);
      }(Signatures{}))
   {
      constexpr bool Plus = RequiresPlusGeometry<FaceDomain>(Signatures{});
      auto qd = MakeGlobalFacetMeshQuadData<IR, Plus>(face_domain);
      return std::tuple{Entry<typename EntryType::key_type, decltype(qd)>{std::move(qd)}};
   }
   else return std::tuple{};
}

template<class FormSignature, class Signatures, class WFContext, class IntegrationRule, class FaceDomain>
constexpr auto MakeFacetOperatorContextImpl(
   const WFContext& wf_ctx,
   const IntegrationRule& ir,
   const FaceDomain& face_domain)
{
   // Invariant: callers pass a restricted global facet context containing the
   // selected face-domain entry and FE fields already restricted to compatible
   // field face bindings. Mapping every remaining DomainKey to `face_domain`
   // qdata is safe only under that restricted-context invariant; unrestricted
   // weak-form contexts must not call this builder.
   using IR = std::remove_cvref_t<IntegrationRule>;
   using Mesh = std::remove_cvref_t<decltype(face_domain.GetFaceMesh())>;
   static_assert(
      global_face_mesh_has_static_face_family_v<Mesh>,
      "MakeFacetOperatorContext(form, context, integration_rule, face_domain) "
      "requires a global face execution domain whose concrete face mesh type "
      "statically fixes the minus/plus local face family.");

   auto face_ir = GetFaceIntegrationRules(IR{});

   auto mesh_facet_qd_t = std::apply(
      [&](auto const&... dom_entries)
      {
         return std::tuple_cat(
            PrepareGlobalFacetMesh<IR, Signatures>(dom_entries, face_domain)...);
      },
      wf_ctx.domains.entries
   );

   auto fe_facet_qd_t = std::apply(
      [&](auto const&... fe_entries)
      {
         return std::tuple_cat(
            PrepareGlobalFacetField<IR, Signatures, FaceDomain>(fe_entries)...);
      },
      wf_ctx.fe_fields.entries
   );

   auto mesh_facet_qd_map = ToStaticMap(std::move(mesh_facet_qd_t));
   auto fe_facet_qd_map = ToStaticMap(std::move(fe_facet_qd_t));

   return FacetOperatorContext<
      IR,
      decltype(mesh_facet_qd_map),
      decltype(fe_facet_qd_map),
      FormSignature>{
         ir,
         face_ir,
         std::move(mesh_facet_qd_map),
         std::move(fe_facet_qd_map)
      };
}

} // namespace operator_context_detail

template<FormExpr Form, class WFContext, class IntegrationRule, class FaceDomain>
constexpr auto MakeFacetOperatorContext(const Form&, const WFContext& context,
                                        const IntegrationRule& rule, const FaceDomain& domain)
{
   using Signature = form_signature_t<Form>;
   return operator_context_detail::MakeFacetOperatorContextImpl<
      Signature, typename Signature::integrands>(context, rule, domain);
}

template<class FormSignature, class... Integrands, class Context, class Rule, class FaceDomain>
auto MakeFacetOperatorContext(const std::tuple<Integrands...>&,
                              const Context& context, const Rule& rule,
                              const FaceDomain& domain)
{
   using Signatures = type_list<typename operator_context_detail::signature_lookup<
      std::remove_cvref_t<Integrands>, typename FormSignature::integrands>::type...>;
   return operator_context_detail::MakeFacetOperatorContextImpl<
      FormSignature, Signatures>(context, rule, domain);
}

} // namespace gendil

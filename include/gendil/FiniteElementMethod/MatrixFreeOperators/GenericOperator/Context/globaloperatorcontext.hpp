// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/domainfiniteelementspaceiteration.hpp"
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

// Compatibility adapter for lower-level interpolation/apply kernels that still
// consume the historical all-local-face qdata tuple shape. Global face context
// construction fills only the selected local-face entry and leaves the other
// entries Empty; this is not a general face-qdata abstraction.
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

template<class IntegrationRule, class MeshQDMap, class FEQDMap>
struct CellOnlyOperatorContext
{
   using integration_rule_type = IntegrationRule;

   IntegrationRule int_rule;
   MeshQDMap mesh_qd;
   FEQDMap fe_qd;

   GENDIL_HOST_DEVICE
   const IntegrationRule& integration_rule() const { return int_rule; }

   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) mesh_quad_data() const
   {
      return mesh_qd.template get<DomainKey<Name>>();
   }

   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) finite_element_quad_data() const
   {
      return fe_qd.template get<FiniteElementFieldKey<Name>>();
   }
};

template<class IntegrationRule, class MeshFacetQDMap, class FEFacetQDMap>
struct FacetOperatorContext
{
   using integration_rule_type = IntegrationRule;
   using face_integration_rules_type =
      decltype(GetFaceIntegrationRules(IntegrationRule{}));

   IntegrationRule int_rule;
   face_integration_rules_type face_int_rules;
   MeshFacetQDMap mesh_facet_qd;
   FEFacetQDMap fe_facet_qd;

   GENDIL_HOST_DEVICE
   const IntegrationRule& integration_rule() const { return int_rule; }

   GENDIL_HOST_DEVICE
   const face_integration_rules_type& facet_integration_rules() const
   {
      return face_int_rules;
   }

   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) mesh_facet_quad_data() const
   {
      return mesh_facet_qd.template get<DomainKey<Name>>();
   }

   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) finite_element_facet_quad_data() const
   {
      return fe_facet_qd.template get<FiniteElementFieldKey<Name>>();
   }
};

template<class IntegrationRule, size_t SelectedIndex, class VolumeSpace>
constexpr auto MakeSelectedMeshFacetQuadDataForSide(
   const VolumeSpace& /*space*/)
{
   using Space = std::remove_cvref_t<VolumeSpace>;
   using Mesh = typename Space::mesh_type;
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

template<class IntegrationRule, class FaceDomain>
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

   if constexpr (is_boundary_face_execution_batch_v<Domain>)
   {
      auto minus_qd =
         MakeSelectedMeshFacetQuadDataForSide<IntegrationRule, MinusI>(
            face_domain.GetCellFiniteElementSpace());
      using MinusQD = std::remove_cvref_t<decltype(minus_qd)>;
      return BoundaryFacetQData<MinusQD>{ static_cast<MinusQD>(minus_qd) };
   }
   else if constexpr (is_interior_face_execution_batch_v<Domain>)
   {
      auto minus_qd =
         MakeSelectedMeshFacetQuadDataForSide<IntegrationRule, MinusI>(
            face_domain.GetMinusCellFiniteElementSpace());
      auto plus_qd =
         MakeSelectedMeshFacetQuadDataForSide<IntegrationRule, PlusI>(
            face_domain.GetPlusCellFiniteElementSpace());
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

template<class IntegrationRule, class FieldBinding>
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
      auto minus_qd =
         MakeSelectedFiniteElementFacetQuadDataForSide<
            IntegrationRule,
            MinusI>(field_binding.GetMinusFiniteElementSpace());
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
      auto minus_qd =
         MakeSelectedFiniteElementFacetQuadDataForSide<
            IntegrationRule,
            MinusI>(field_binding.GetMinusFiniteElementSpace());
      auto plus_qd =
         MakeSelectedFiniteElementFacetQuadDataForSide<
            IntegrationRule,
            PlusI>(field_binding.GetPlusFiniteElementSpace());
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

template<class IR, class DomainEntry, class FaceDomain>
constexpr auto domain_entry_to_global_facet_mesh_qd_tuple(
   const DomainEntry& e,
   const FaceDomain& face_domain)
{
   using Key = typename DomainEntry::key_type;
   (void)e;
   auto qd = MakeGlobalFacetMeshQuadData<IR>(face_domain);
   using QD = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

template<class IR, class FEFieldEntry>
constexpr auto fe_field_entry_to_global_facet_qd_tuple(
   const FEFieldEntry& e)
{
   using Key = typename FEFieldEntry::key_type;
   const auto& fev = e.value;
   auto qd =
      MakeGlobalFacetFiniteElementQuadData<IR>(fev.space);
   using QD = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

template<class WFContext, class IntegrationRule>
constexpr auto MakeCellOnlyOperatorContext(
   const WFContext& wf_ctx,
   const IntegrationRule& ir)
{
   using IR = std::remove_cvref_t<IntegrationRule>;

   auto mesh_qd_t = std::apply(
      [&](auto const&... dom_entries)
      {
         return std::tuple_cat(domain_entry_to_mesh_qd_tuple<IR>(dom_entries)...);
      },
      wf_ctx.domains.entries
   );

   auto fe_qd_t = std::apply(
      [&](auto const&... fe_entries)
      {
         return std::tuple_cat(fe_field_entry_to_elem_qd_tuple<IR>(fe_entries)...);
      },
      wf_ctx.fe_fields.entries
   );

   auto mesh_qd_map = tuple_to_map(std::move(mesh_qd_t));
   auto fe_qd_map = tuple_to_map(std::move(fe_qd_t));

   return CellOnlyOperatorContext<
      IR,
      decltype(mesh_qd_map),
      decltype(fe_qd_map)>{
         ir,
         std::move(mesh_qd_map),
         std::move(fe_qd_map)
      };
}

template<class WFContext, class IntegrationRule, class FaceDomain>
constexpr auto MakeFacetOperatorContext(
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
      "MakeFacetOperatorContext(wf_ctx, integration_rule, face_domain) "
      "requires a global face execution domain whose concrete face mesh type "
      "statically fixes the minus/plus local face family.");

   auto face_ir = GetFaceIntegrationRules(IR{});

   auto mesh_facet_qd_t = std::apply(
      [&](auto const&... dom_entries)
      {
         return std::tuple_cat(
            domain_entry_to_global_facet_mesh_qd_tuple<IR>(
               dom_entries,
               face_domain)...);
      },
      wf_ctx.domains.entries
   );

   auto fe_facet_qd_t = std::apply(
      [&](auto const&... fe_entries)
      {
         return std::tuple_cat(
            fe_field_entry_to_global_facet_qd_tuple<IR>(
               fe_entries)...);
      },
      wf_ctx.fe_fields.entries
   );

   auto mesh_facet_qd_map = tuple_to_map(std::move(mesh_facet_qd_t));
   auto fe_facet_qd_map = tuple_to_map(std::move(fe_facet_qd_t));

   return FacetOperatorContext<
      IR,
      decltype(mesh_facet_qd_map),
      decltype(fe_facet_qd_map)>{
         ir,
         face_ir,
         std::move(mesh_facet_qd_map),
         std::move(fe_facet_qd_map)
      };
}

} // namespace gendil

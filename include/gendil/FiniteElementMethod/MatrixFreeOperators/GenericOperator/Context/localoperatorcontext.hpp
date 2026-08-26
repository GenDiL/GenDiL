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

   // Local/cell-owned facet qdata stores the all-local-face tuple used by the
   // fused CellIterator path. MinusSide/PlusSide are local-row compatibility
   // accessors over the same tuple-shaped data; this is not global side-selected qdata
   // and not the future two-space face model.
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
   class FEFacetQDMap>
struct OperatorContext
{
   using integration_rule_type = IntegrationRule;
   using face_integration_rules_type =
      decltype(GetFaceIntegrationRules(IntegrationRule{}));

   IntegrationRule int_rule;
   face_integration_rules_type face_int_rules;
   MeshQDMap mesh_qd;
   FEQDMap fe_qd;
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

template<class WFContext, class IntegrationRule>
constexpr auto MakeOperatorContext(const WFContext& wf_ctx, const IntegrationRule& ir)
{
   using IR = std::remove_cvref_t<IntegrationRule>;

   auto face_ir = GetFaceIntegrationRules(IR{});

   auto mesh_qd_t = std::apply(
      [&](auto const&... dom_entries)
      {
         return std::tuple_cat(
            MakeMeshQuadDataEntryTuple<IR>(dom_entries)...);
      },
      wf_ctx.domains.entries
   );

   auto fe_qd_t = std::apply(
      [&](auto const&... fe_entries)
      {
         return std::tuple_cat(
            MakeFiniteElementQuadDataEntryTuple<IR>(fe_entries)...);
      },
      wf_ctx.fe_fields.entries
   );

   auto mesh_facet_qd_t = std::apply(
      [&](auto const&... dom_entries)
      {
         return std::tuple_cat(
            MakeLocalFacetMeshQuadDataEntryTuple<IR>(dom_entries)...);
      },
      wf_ctx.domains.entries
   );

   auto fe_facet_qd_t = std::apply(
      [&](auto const&... fe_entries)
      {
         return std::tuple_cat(
            MakeLocalFacetFiniteElementQuadDataEntryTuple<IR>(
               fe_entries)...);
      },
      wf_ctx.fe_fields.entries
   );

   auto mesh_qd_map       = tuple_to_map(std::move(mesh_qd_t));
   auto fe_qd_map         = tuple_to_map(std::move(fe_qd_t));
   auto mesh_facet_qd_map = tuple_to_map(std::move(mesh_facet_qd_t));
   auto fe_facet_qd_map   = tuple_to_map(std::move(fe_facet_qd_t));

   return OperatorContext<
      IR,
      decltype(mesh_qd_map),
      decltype(fe_qd_map),
      decltype(mesh_facet_qd_map),
      decltype(fe_facet_qd_map)>{
         ir,
         face_ir,
         std::move(mesh_qd_map),
         std::move(fe_qd_map),
         std::move(mesh_facet_qd_map),
         std::move(fe_facet_qd_map)
      };
}

} // namespace gendil

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

template<class IntegrationRule, class Space>
constexpr auto MakeMeshFacetQuadData(const CellIntegrationDomain<Space>& /*dom*/)
{
   using SpaceType = std::remove_cvref_t<Space>;
   using FaceIRs = decltype(GetFaceIntegrationRules(IntegrationRule{}));

   if constexpr (is_mixed_finite_element_space_v<SpaceType>)
   {
      static_assert(
         dependent_false_v<SpaceType>,
         "MakeMeshFacetQuadData builds local/cell-owned facet qdata for one "
         "selected homogeneous CellIntegrationDomain<Space>. Mixed domains "
         "must be iterated and restricted to a homogeneous cell batch before "
         "MakeOperatorContext is built.");
   }
   else if constexpr (is_cell_finite_element_space_v<SpaceType>)
   {
      using Mesh = typename SpaceType::mesh_type;
      auto qd = MakeMeshFaceQuadData<Mesh>(FaceIRs{});
      using QD = std::remove_cvref_t<decltype(qd)>;
      // Local/cell-owned facet compatibility: local traversal may visit any
      // local face of the active Cell, so this stores the full all-face tuple.
      // Global face-domain traversal uses side-selected face qdata instead.
      return LocalFacetQuadratureData<QD>{ static_cast<QD>(qd) };
   }
   else
   {
      static_assert(
         dependent_false_v<SpaceType>,
         "MakeMeshFacetQuadData requires CellIntegrationDomain<Space> to wrap "
         "a selected homogeneous cell finite element space.");
   }
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
   static_assert(
      is_cell_finite_element_space_v<Space>,
      "MakeFiniteElementFacetQuadData builds local/cell-owned facet qdata "
      "from a volume finite element space. Face finite element spaces are "
      "valid only in side-selected global facet contexts.");

   // Local/cell-owned facet compatibility case. The binding is a volume finite
   // element space, and the local traversal may select any local face. Global
   // face-domain contexts use side-selected qdata so they can select exactly
   // the minus/plus face family of the face mesh.
   auto qd =
      MakeVolumeFiniteElementFacetQuadData<IntegrationRule>(space);
   using QD = std::remove_cvref_t<decltype(qd)>;
   return LocalFacetQuadratureData<QD>{ static_cast<QD>(qd) };
}

template<class IR, class DomainEntry>
constexpr auto domain_entry_to_mesh_facet_qd_tuple(const DomainEntry& e)
{
   using Key = typename DomainEntry::key_type;
   auto qd   = MakeMeshFacetQuadData<IR>(e.value);
   using QD  = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

template<class IR, class FEFieldEntry>
constexpr auto fe_field_entry_to_facet_qd_tuple(const FEFieldEntry& e)
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

   auto mesh_facet_qd_t = std::apply(
      [&](auto const&... dom_entries)
      {
         return std::tuple_cat(domain_entry_to_mesh_facet_qd_tuple<IR>(dom_entries)...);
      },
      wf_ctx.domains.entries
   );

   auto fe_facet_qd_t = std::apply(
      [&](auto const&... fe_entries)
      {
         return std::tuple_cat(fe_field_entry_to_facet_qd_tuple<IR>(fe_entries)...);
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

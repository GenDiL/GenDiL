// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/staticmap.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globalfacefieldbinding.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/doftoquad.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/signatures.hpp"

namespace gendil
{

namespace operator_context_detail {

// Lookup only: signatures were resolved against complete-form declarations
// before any domain selection or numerical preparation.
template<class Integrand, class Signatures> struct signature_lookup;
template<class Integrand, class Head, class... Tail>
struct signature_lookup<Integrand, type_list<Head, Tail...>>
   : std::conditional_t<std::is_same_v<std::remove_cvref_t<Integrand>, typename Head::integrand_type>,
        std::type_identity<Head>, signature_lookup<Integrand, type_list<Tail...>>> {};
template<class Integrand>
struct signature_lookup<Integrand, type_list<>>
{
   static_assert(dependent_false_v<Integrand>,
      "The integrand is absent from the operator context's analyzed form.");
};
template<class FormSignature, class Integrand>
struct signature_for : signature_lookup<Integrand, typename FormSignature::integrands> {};
template<class IntegrationRule, class Mesh>
consteval bool IntegrationRuleGeometryMatchesMesh()
{
   if constexpr (
      mesh::MeshWithCellGeometry<Mesh> &&
      requires
      {
         typename IntegrationRule::geometry;
      })
   {
      return std::is_same_v<
         typename IntegrationRule::geometry,
         mesh::mesh_geometry_t<Mesh>>;
   }
   else
   {
      return false;
   }
}

template<class IntegrationRule, class Mesh>
constexpr void ValidateIntegrationRuleGeometry()
{
   static_assert(
      IntegrationRuleGeometryMatchesMesh<IntegrationRule, Mesh>(),
      "IntegrationRule: volume reference geometry must exactly match the "
      "selected domain mesh cell geometry.");
}

} // namespace operator_context_detail

template<class IntegrationRule, class MeshQDMap, class FEQDMap, class FormSignature>
struct CellOnlyOperatorContext
{
   using integration_rule_type = IntegrationRule;
   using form_signature = FormSignature;
   template<class Integrand> using signature_for =
      typename operator_context_detail::signature_for<FormSignature, Integrand>::type;

   IntegrationRule int_rule;
   MeshQDMap mesh_qd;
   FEQDMap fe_qd;

   GENDIL_HOST_DEVICE
   const IntegrationRule& GetIntegrationRule() const { return int_rule; }

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
};

template<class IntegrationRule, class DomainView>
constexpr auto MakeMeshQuadData(const DomainView& domain)
{
   const auto& mesh = GetCellIntegrationDomainMesh(domain);
   using Mesh = std::remove_cvref_t<decltype(mesh)>;
   operator_context_detail::ValidateIntegrationRuleGeometry<
      IntegrationRule,
      Mesh>();
   using QD = typename Mesh::cell_type::template QuadData<IntegrationRule>;
   return QD{};
}

template<class IntegrationRule, class SpaceView>
constexpr auto MakeFiniteElementQuadData(const SpaceView& space)
{
   using Space = std::remove_cvref_t<SpaceView>;
   if constexpr (
      is_boundary_face_field_binding_v<Space> ||
      is_interior_face_field_binding_v<Space>)
   {
      static_assert(
         dependent_false_v<Space>,
         "MakeFiniteElementQuadData builds cell/volume finite element qdata. "
         "Face field bindings must use the global facet qdata builder "
         "so minus/plus sides are represented explicitly.");
   }
   else if constexpr (is_cell_finite_element_space_v<Space>)
   {
      (void)space;
      using FE    = typename Space::finite_element_type;
      using Shape = typename FE::shape_functions;
      return MakeDofToQuad<Shape, IntegrationRule>();
   }
   else
   {
      static_assert(
         dependent_false_v<Space>,
         "MakeFiniteElementQuadData requires a selected homogeneous finite "
         "element space. Mixed finite element spaces must be iterated and "
         "restricted to a selected cell batch before qdata construction.");
   }
}

template<class IR, class DomainEntry>
constexpr auto MakeMeshQuadDataEntryTuple(const DomainEntry& e)
{
   using Key = typename DomainEntry::key_type;
   auto qd   = MakeMeshQuadData<IR>(e.value);
   using QD  = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

template<class IR, class FEFieldEntry>
constexpr auto MakeFiniteElementQuadDataEntryTuple(const FEFieldEntry& e)
{
   using Key       = typename FEFieldEntry::key_type;
   const auto& fev = e.value;
   auto qd         = MakeFiniteElementQuadData<IR>(fev.space);
   using QD        = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

} // namespace gendil

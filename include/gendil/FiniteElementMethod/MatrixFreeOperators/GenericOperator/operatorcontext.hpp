// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticmap.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/doftoquad.hpp"

namespace gendil
{

/**
 * @file operatorcontext.hpp
 * @brief Matrix-free operator context storing quadrature-related data for cell
 *        and facet integration.
 *
 * This file defines:
 *
 * - the `OperatorContext` aggregate,
 * - default customization points used to build cell and facet quadrature data,
 * - helper functions used by `MakeOperatorContext`.
 *
 * The operator context stores two families of data:
 *
 * - **cell data**
 *   - one cell integration rule,
 *   - one mesh quadrature-data object per domain,
 *   - one finite-element quadrature-data object per finite-element field;
 *
 * - **facet data**
 *   - one collection of face integration rules derived from the cell rule,
 *   - one mesh facet quadrature-data container per domain,
 *   - one finite-element facet quadrature-data container per finite-element
 *     field.
 *
 * For facets, the stored quadrature data is intentionally the **full per-face
 * container**. Selection of the local-face entry and any conforming /
 * non-conforming handling is deferred to the existing face-aware interpolation
 * layer.
 *
 * @note This design assumes the existence of:
 * - `DomainKey<Name>`
 * - `FiniteElementFieldKey<Name>`
 * - `Entry<Key, Value>`
 * - `tuple_to_map(...)`
 * - `GetFaceIntegrationRules(IntegrationRule{})`
 * - `MakeMeshFaceQuadData<Mesh>(face_integration_rules)`
 * - `MakeFaceDofToQuad<ShapeFunctions, FaceIntegrationRules>()`
 * - `MakeDofToQuad<ShapeFunctions, IntegrationRule>()`
 */


/* -------------------------------------------------------------------------- */
/*                              OperatorContext                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief Matrix-free operator context storing quadrature-related data for cell
 *        and facet integration.
 *
 * @tparam IntegrationRule  Cell integration-rule type.
 * @tparam MeshQDMap        Map type storing cell mesh quadrature data keyed by
 *                          `DomainKey<Name>`.
 * @tparam FEQDMap          Map type storing cell FE quadrature data keyed by
 *                          `FiniteElementFieldKey<Name>`.
 * @tparam MeshFacetQDMap   Map type storing facet mesh quadrature data keyed by
 *                          `DomainKey<Name>`.
 * @tparam FEFacetQDMap     Map type storing facet FE quadrature data keyed by
 *                          `FiniteElementFieldKey<Name>`.
 *
 * The facet maps store the full per-face quadrature-data containers. They are
 * not restricted to a local face at this level.
 */
template<
   class IntegrationRule,
   class MeshQDMap,
   class FEQDMap,
   class MeshFacetQDMap,
   class FEFacetQDMap>
struct OperatorContext
{
   /// Type of the cell integration rule.
   using integration_rule_type = IntegrationRule;

   /**
    * @brief Type of the face integration-rule container associated with the
    *        cell integration rule.
    */
   using face_integration_rules_type =
      decltype(GetFaceIntegrationRules(IntegrationRule{}));

   /// Cell integration rule used for cell/domain integration.
   IntegrationRule int_rule;

   /// Face integration rules derived from the cell integration rule.
   face_integration_rules_type face_int_rules;

   /// Cell mesh quadrature data: `DomainKey<Name> -> MeshQuadData<IntegrationRule>`.
   MeshQDMap mesh_qd;

   /// Cell FE quadrature data: `FiniteElementFieldKey<Name> -> DofToQuad`.
   FEQDMap fe_qd;

   /**
    * @brief Facet mesh quadrature data:
    * `DomainKey<Name> -> MeshFaceQuadData<face_integration_rules_type>`.
    */
   MeshFacetQDMap mesh_facet_qd;

   /**
    * @brief Facet FE quadrature data:
    * `FiniteElementFieldKey<Name> -> FaceDofToQuad`.
    */
   FEFacetQDMap fe_facet_qd;

   /**
    * @brief Return the stored cell integration rule.
    */
   GENDIL_HOST_DEVICE
   const IntegrationRule& integration_rule() const { return int_rule; }

   /**
    * @brief Return the stored face integration rules.
    */
   GENDIL_HOST_DEVICE
   const face_integration_rules_type& facet_integration_rules() const
   {
      return face_int_rules;
   }

   /**
    * @brief Return the cell mesh quadrature data associated with domain `Name`.
    *
    * @tparam Name  Compile-time domain name.
    */
   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) mesh_quad_data() const
   {
      return mesh_qd.template get<DomainKey<Name>>();
   }

   /**
    * @brief Return the cell finite-element quadrature data associated with FE
    *        field `Name`.
    *
    * @tparam Name  Compile-time finite-element field name.
    */
   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) finite_element_quad_data() const
   {
      return fe_qd.template get<FiniteElementFieldKey<Name>>();
   }

   /**
    * @brief Return the facet mesh quadrature data associated with domain `Name`.
    *
    * @tparam Name  Compile-time domain name.
    *
    * @return The full per-face mesh quadrature-data container.
    */
   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) mesh_facet_quad_data() const
   {
      return mesh_facet_qd.template get<DomainKey<Name>>();
   }

   /**
    * @brief Return the facet finite-element quadrature data associated with FE
    *        field `Name`.
    *
    * @tparam Name  Compile-time finite-element field name.
    *
    * @return The full per-face FE quadrature-data container.
    */
   template<StaticString Name>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) finite_element_facet_quad_data() const
   {
      return fe_facet_qd.template get<FiniteElementFieldKey<Name>>();
   }
};


/* -------------------------------------------------------------------------- */
/*                        Default quadrature-data builders                     */
/* -------------------------------------------------------------------------- */

/**
 * @brief Construct the default cell mesh quadrature data for a domain view.
 *
 * The default implementation assumes:
 * @code
 * typename Domain::cell_type::template QuadData<IntegrationRule>
 * @endcode
 *
 * @tparam IntegrationRule  Cell integration-rule type.
 * @tparam DomainView       Domain-view type.
 *
 * @param[in] dom  Domain view.
 *
 * @return A default-constructed cell mesh quadrature-data object.
 */
template<class IntegrationRule, class DomainView>
constexpr auto MakeMeshQuadData(const DomainView& /*dom*/)
{
   using Domain = std::remove_cvref_t<DomainView>;
   using QD = typename Domain::cell_type::template QuadData<IntegrationRule>;
   return QD{};
}

/**
 * @brief Construct the default cell FE quadrature data for a space view.
 *
 * The default implementation assumes the space view exposes:
 * @code
 * typename Space::finite_element_type::shape_functions
 * @endcode
 *
 * @tparam IntegrationRule  Cell integration-rule type.
 * @tparam SpaceView        FE-space view type.
 *
 * @param[in] space  FE-space view.
 *
 * @return A default-constructed cell FE quadrature-data object.
 */
template<class IntegrationRule, class SpaceView>
constexpr auto MakeFiniteElementQuadData(const SpaceView& /*space*/)
{
   using Space = std::remove_cvref_t<SpaceView>;
   using FE    = typename Space::finite_element_type;
   using Shape = typename FE::shape_functions;
   return MakeDofToQuad<Shape, IntegrationRule>();
}

/**
 * @brief Construct the default facet mesh quadrature data for a domain view.
 *
 * The default implementation assumes:
 * @code
 * auto face_ir = GetFaceIntegrationRules(IntegrationRule{});
 * auto qd      = MakeMeshFaceQuadData<Domain>(face_ir);
 * @endcode
 *
 * @tparam IntegrationRule  Cell integration-rule type from which facet rules
 *                          are derived.
 * @tparam DomainView       Domain-view type.
 *
 * @param[in] dom  Domain view.
 *
 * @return A default-constructed facet mesh quadrature-data container.
 */
template<class IntegrationRule, class DomainView>
constexpr auto MakeMeshFacetQuadData(const DomainView& /*dom*/)
{
   using Domain  = std::remove_cvref_t<DomainView>;
   using FaceIRs = decltype(GetFaceIntegrationRules(IntegrationRule{}));
   return MakeMeshFaceQuadData<Domain>(FaceIRs{});
}

/**
 * @brief Construct the default facet FE quadrature data for a space view.
 *
 * The default implementation assumes the space view exposes:
 * @code
 * typename Space::finite_element_type::shape_functions
 * @endcode
 * and that facet interpolation data can be built through
 * `MakeFaceDofToQuad<Shape, FaceIRs>()`.
 *
 * @tparam IntegrationRule  Cell integration-rule type from which facet rules
 *                          are derived.
 * @tparam SpaceView        FE-space view type.
 *
 * @param[in] space  FE-space view.
 *
 * @return A default-constructed facet FE quadrature-data container.
 */
template<class IntegrationRule, class SpaceView>
constexpr auto MakeFiniteElementFacetQuadData(const SpaceView& /*space*/)
{
   using Space   = std::remove_cvref_t<SpaceView>;
   using FE      = typename Space::finite_element_type;
   using Shape   = typename FE::shape_functions;
   using FaceIRs = decltype(GetFaceIntegrationRules(IntegrationRule{}));
   return MakeFaceDofToQuad<Shape, FaceIRs>();
}


/* -------------------------------------------------------------------------- */
/*                     Internal helpers: entry -> tuple                        */
/* -------------------------------------------------------------------------- */

/**
 * @brief Convert one domain entry into a one-element tuple containing the
 *        corresponding cell mesh quadrature-data entry.
 *
 * @tparam IR          Cell integration-rule type.
 * @tparam DomainEntry Domain-map entry type.
 *
 * @param[in] e  Domain entry.
 */
template<class IR, class DomainEntry>
constexpr auto domain_entry_to_mesh_qd_tuple(const DomainEntry& e)
{
   using Key = typename DomainEntry::key_type;
   auto qd   = MakeMeshQuadData<IR>(e.value);
   using QD  = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

/**
 * @brief Convert one FE-field entry into a one-element tuple containing the
 *        corresponding cell FE quadrature-data entry.
 *
 * @tparam IR           Cell integration-rule type.
 * @tparam FEFieldEntry FE-field-map entry type.
 *
 * @param[in] e  FE-field entry.
 */
template<class IR, class FEFieldEntry>
constexpr auto fe_field_entry_to_elem_qd_tuple(const FEFieldEntry& e)
{
   using Key       = typename FEFieldEntry::key_type;
   const auto& fev = e.value;
   auto qd         = MakeFiniteElementQuadData<IR>(fev.space);
   using QD        = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

/**
 * @brief Convert one domain entry into a one-element tuple containing the
 *        corresponding facet mesh quadrature-data entry.
 *
 * @tparam IR          Cell integration-rule type.
 * @tparam DomainEntry Domain-map entry type.
 *
 * @param[in] e  Domain entry.
 */
template<class IR, class DomainEntry>
constexpr auto domain_entry_to_mesh_facet_qd_tuple(const DomainEntry& e)
{
   using Key = typename DomainEntry::key_type;
   auto qd   = MakeMeshFacetQuadData<IR>(e.value);
   using QD  = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

/**
 * @brief Convert one FE-field entry into a one-element tuple containing the
 *        corresponding facet FE quadrature-data entry.
 *
 * @tparam IR           Cell integration-rule type.
 * @tparam FEFieldEntry FE-field-map entry type.
 *
 * @param[in] e  FE-field entry.
 */
template<class IR, class FEFieldEntry>
constexpr auto fe_field_entry_to_facet_qd_tuple(const FEFieldEntry& e)
{
   using Key       = typename FEFieldEntry::key_type;
   const auto& fev = e.value;
   auto qd         = MakeFiniteElementFacetQuadData<IR>(fev.space);
   using QD        = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}


/* -------------------------------------------------------------------------- */
/*                           MakeOperatorContext                              */
/* -------------------------------------------------------------------------- */

/**
 * @brief Build an `OperatorContext` from a weak-form context and a cell
 *        integration rule.
 *
 * The resulting context stores:
 *
 * - the cell integration rule,
 * - the face integration rules derived from it,
 * - one cell mesh quadrature-data object per domain,
 * - one cell FE quadrature-data object per FE field,
 * - one facet mesh quadrature-data container per domain,
 * - one facet FE quadrature-data container per FE field.
 *
 * @tparam WFContext        Weak-form-context type.
 * @tparam IntegrationRule  Cell integration-rule type.
 *
 * @param[in] wf_ctx  Weak-form context containing the domain and FE-field maps.
 * @param[in] ir      Cell integration rule.
 *
 * @return A fully constructed `OperatorContext`.
 */
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

// // Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// //
// // SPDX-License-Identifier: (BSD-3-Clause)

// #pragma once

// #include "gendil/prelude.hpp"
// #include "gendil/Utilities/staticmap.hpp"
// #include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/doftoquad.hpp"

// namespace gendil
// {

// //------------------------------------------------------------------------------
// // OperatorContext
// //------------------------------------------------------------------------------

// template<class IntegrationRule, class MeshQDMap, class FEQDMap>
// struct OperatorContext
// {
//    // Cell context
//    IntegrationRule int_rule;
//    MeshQDMap mesh_qd;  // DomainKey<Name> -> MeshQuadData<IR>
//    FEQDMap fe_qd;  // FiniteElementFieldKey<Name> -> FiniteElementQuadData<IR>

//    const IntegrationRule & integration_rule() const { return int_rule; }
//    template<StaticString Name>
//    constexpr decltype(auto) mesh_quad_data() const { return mesh_qd.template get<DomainKey<Name>>(); }

//    template<StaticString Name>
//    constexpr decltype(auto) finite_element_quad_data() const { return fe_qd.template get<FiniteElementFieldKey<Name>>(); }
// };

// //------------------------------------------------------------------------------
// // Customization points (defaults)
// //------------------------------------------------------------------------------

// // Default: Domain is (a view of) a Mesh type that provides cell_type::QuadData<IR>.
// template<class IntegrationRule, class DomainView>
// constexpr auto MakeMeshQuadData(const DomainView& /*dom*/)
// {
//    using Domain = std::remove_cvref_t<DomainView>;
//    using QD = typename Domain::cell_type::template QuadData<IntegrationRule>;
//    return QD{};
// }

// // Default: FE-space view provides finite_element_type::shape_functions.
// // You can adapt this if your FE-space view exposes shape functions differently.
// template<class IntegrationRule, class SpaceView>
// constexpr auto MakeFiniteElementQuadData(const SpaceView& /*space*/)
// {
//    using Space = std::remove_cvref_t<SpaceView>;
//    using FE    = typename Space::finite_element_type;
//    using Shape = typename FE::shape_functions;
//    // If MakeDofToQuad expects (Shape, IR) as template params:
//    return MakeDofToQuad<Shape, IntegrationRule>();
// }

// //------------------------------------------------------------------------------
// // Internal helpers: build tuple of entries -> StaticMap
// //------------------------------------------------------------------------------

// template<class IR, class DomainEntry>
// constexpr auto domain_entry_to_mesh_qd_tuple(const DomainEntry& e)
// {
//    using Key = typename DomainEntry::key_type;        // DomainKey<Name>
//    // Key is guaranteed domain key because we apply on wf_ctx.domains only.
//    auto qd = MakeMeshQuadData<IR>(e.value);
//    using QD = std::remove_cvref_t<decltype(qd)>;
//    return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
// }

// template<class IR, class FEFieldEntry>
// constexpr auto fe_field_entry_to_elem_qd_tuple(const FEFieldEntry& e)
// {
//    using Key = typename FEFieldEntry::key_type;       // FiniteElementFieldKey<Name>
//    const auto& fev = e.value;                         // FiniteElementFieldView<SpaceView, DofsView>
//    auto qd = MakeFiniteElementQuadData<IR>(fev.space);
//    using QD = std::remove_cvref_t<decltype(qd)>;
//    return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
// }

// //------------------------------------------------------------------------------
// // MakeOperatorContext
// //------------------------------------------------------------------------------

// // Pass IR either as a type parameter or as an object; we only use the type.
// // Example usage:
// //   auto op_ctx = gendil::op::MakeOperatorContext<IntegrationRule>(wf_ctx);
// // or
// //   auto op_ctx = gendil::op::MakeOperatorContext(wf_ctx, IntegrationRule{});
// template<class WFContext, class IntegrationRule>
// constexpr auto MakeOperatorContext(const WFContext& wf_ctx, const IntegrationRule& ir)
// {
//    using IR = std::remove_cvref_t<IntegrationRule>;

//    // Build mesh_qd map from wf_ctx.domains
//    auto mesh_qd_t = std::apply(
//       [&](auto const&... dom_entries)
//       {
//          return std::tuple_cat(domain_entry_to_mesh_qd_tuple<IR>(dom_entries)...);
//       },
//       wf_ctx.domains.entries
//    );

//    // Build elem_qd map from wf_ctx.fe_fields
//    auto elem_qd_t = std::apply(
//       [&](auto const&... fe_entries)
//       {
//          return std::tuple_cat(fe_field_entry_to_elem_qd_tuple<IR>(fe_entries)...);
//       },
//       wf_ctx.fe_fields.entries
//    );

//    auto mesh_qd_map = tuple_to_map(std::move(mesh_qd_t));
//    auto fe_qd_map   = tuple_to_map(std::move(elem_qd_t));

//    return OperatorContext<IntegrationRule, decltype(mesh_qd_map), decltype(fe_qd_map)>{
//       ir,
//       std::move(mesh_qd_map),
//       std::move(fe_qd_map)
//    };
// }

// } // namespace gendil

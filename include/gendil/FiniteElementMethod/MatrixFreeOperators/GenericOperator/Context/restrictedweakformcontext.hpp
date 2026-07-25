// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globalfacefieldbinding.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/domainiteration.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"
#include "gendil/Utilities/debug.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <tuple>
#include <type_traits>

namespace gendil {

enum class restricted_execution_domain_kind
{
   cell,
   interior_face,
   boundary_face
};

template<class Domain>
struct restricted_execution_domain_traits;

template<
   StaticString Name,
   size_t CellI,
   class CellMesh,
   bool Partitioned>
struct restricted_execution_domain_traits<
   SelectedCellExecutionDomain<Name, CellI, CellMesh, Partitioned>>
{
   static constexpr auto kind = restricted_execution_domain_kind::cell;
   static constexpr size_t index = CellI;
};

template<
   StaticString Name,
   size_t FaceI,
   class FacePart,
   class CellMesh>
struct restricted_execution_domain_traits<
   SelectedBoundaryFaceExecutionDomain<
      Name,
      FaceI,
      FacePart,
      CellMesh>>
{
   static constexpr auto kind =
      restricted_execution_domain_kind::boundary_face;
   static constexpr size_t index = FaceI;
};

template<
   StaticString Name,
   size_t FaceI,
   class FacePart,
   class MinusCellMesh,
   class PlusCellMesh>
struct restricted_execution_domain_traits<
   SelectedInteriorFaceExecutionDomain<
      Name,
      FaceI,
      FacePart,
      MinusCellMesh,
      PlusCellMesh>>
{
   static constexpr auto kind =
      restricted_execution_domain_kind::interior_face;
   static constexpr size_t index = FaceI;
};

template<class Domain>
inline constexpr auto restricted_execution_domain_kind_v =
   restricted_execution_domain_traits<
      std::remove_cvref_t<Domain>>::kind;

template<class Domain>
inline constexpr size_t restricted_execution_domain_index_v =
   restricted_execution_domain_traits<
      std::remove_cvref_t<Domain>>::index;

template<
   StaticString Name,
   size_t CellI,
   class CellMesh,
   bool Partitioned>
constexpr auto MakeRestrictedDomainEntries(
   Cells<Name>,
   const SelectedCellExecutionDomain<
      Name,
      CellI,
      CellMesh,
      Partitioned>& selected)
{
   using Selected = std::remove_cvref_t<decltype(selected)>;
   using Domain = CellIntegrationDomain<Selected>;
   return std::tuple{
      Entry<DomainKey<Name>, Domain>{
         Domain{ Selected{ selected } } } };
}

template<
   StaticString Name,
   size_t CellI,
   class CellMesh,
   bool Partitioned>
constexpr auto MakeRestrictedDomainEntries(
   InteriorFacets<Name>,
   const SelectedCellExecutionDomain<
      Name,
      CellI,
      CellMesh,
      Partitioned>& selected)
{
   using Selected = std::remove_cvref_t<decltype(selected)>;
   using Domain = CellIntegrationDomain<Selected>;
   return std::tuple{
      Entry<DomainKey<Name>, Domain>{
         Domain{ Selected{ selected } } } };
}

template<
   StaticString Name,
   size_t FaceI,
   class FacePart,
   class MinusCellMesh,
   class PlusCellMesh>
constexpr auto MakeRestrictedDomainEntries(
   InteriorFacets<Name>,
   const SelectedInteriorFaceExecutionDomain<
      Name,
      FaceI,
      FacePart,
      MinusCellMesh,
      PlusCellMesh>& selected)
{
   using Selected = std::remove_cvref_t<decltype(selected)>;
   using CellSelection =
      SelectedCellExecutionDomain<Name, FacePart::minus_cell_index, MinusCellMesh, true>;
   using CellDomain = CellIntegrationDomain<CellSelection>;
   using FaceDomain = InteriorFaceIntegrationDomain<Selected>;
   return std::tuple{
      Entry<DomainKey<Name>, CellDomain>{
         CellDomain{
            CellSelection{ MinusCellMesh{ selected.GetMinusCellMesh() } } } },
      Entry<InteriorFaceDomainKey<Name>, FaceDomain>{
         FaceDomain{ Selected{ selected } } } };
}

template<
   StaticString Name,
   size_t CellI,
   class CellMesh,
   bool Partitioned>
constexpr auto MakeRestrictedDomainEntries(
   BoundaryFacets<Name>,
   const SelectedCellExecutionDomain<
      Name,
      CellI,
      CellMesh,
      Partitioned>& selected)
{
   using Selected = std::remove_cvref_t<decltype(selected)>;
   using Domain = CellIntegrationDomain<Selected>;
   return std::tuple{
      Entry<DomainKey<Name>, Domain>{
         Domain{ Selected{ selected } } } };
}

template<
   StaticString Name,
   size_t FaceI,
   class FacePart,
   class CellMesh>
constexpr auto MakeRestrictedDomainEntries(
   BoundaryFacets<Name>,
   const SelectedBoundaryFaceExecutionDomain<
      Name,
      FaceI,
      FacePart,
      CellMesh>& selected)
{
   using Selected = std::remove_cvref_t<decltype(selected)>;
   using CellSelection =
      SelectedCellExecutionDomain<Name, FacePart::cell_index, CellMesh, true>;
   using CellDomain = CellIntegrationDomain<CellSelection>;
   using FaceDomain = BoundaryFaceIntegrationDomain<Selected>;
   return std::tuple{
      Entry<DomainKey<Name>, CellDomain>{
         CellDomain{
            CellSelection{ CellMesh{ selected.GetCellMesh() } } } },
      Entry<BoundaryFaceDomainKey<Name>, FaceDomain>{
         FaceDomain{ Selected{ selected } } } };
}

template<class DomainMesh, class FieldSpace>
void ValidateSelectedCellMeshCompatibility(
   const DomainMesh& domain_mesh,
   const FieldSpace& field_space)
{
   using Field = std::remove_cvref_t<FieldSpace>;
   using FieldMesh = typename Field::mesh_type;
   using Compatibility =
      IsCompatibleMeshDomain<
         std::remove_cvref_t<DomainMesh>,
         FieldMesh>;

   static_assert(
      Compatibility::value,
      "Active finite-element field mesh type is incompatible with the "
      "selected integration-domain cell mesh type.");
   if constexpr (Compatibility::value)
   {
      GENDIL_VERIFY(
         Compatibility::Check(
            domain_mesh,
            static_cast<const FieldMesh&>(field_space)),
         "Active finite-element field is not compatible with the selected "
         "integration-domain cell topology.");
   }
}

template<class Requirement, class IntegrationDomain, class WFContext>
void ValidateActiveFiniteElementFieldDomainCompatibility(
   const IntegrationDomain& integration_domain,
   const WFContext& wf_ctx)
{
   constexpr auto Name = Requirement::name;
   using Context = std::remove_cvref_t<WFContext>;
   static_assert(
      Context::template has_fe_field<Name>(),
      "WeakFormContext is missing an active trial, test, or coefficient "
      "finite-element field required by the weak form.");

   if constexpr (Context::template has_fe_field<Name>())
   {
      const auto& field = wf_ctx.template fe_field<Name>().space;
      using Field = std::remove_cvref_t<decltype(field)>;
      using Domain = std::remove_cvref_t<IntegrationDomain>;

      if constexpr (is_mesh_integration_domain_v<Domain>)
      {
         static_assert(
            is_cell_finite_element_space_v<Field>,
            "An active field on MeshIntegrationDomain must use a homogeneous "
            "FiniteElementSpace.");
         if constexpr (is_cell_finite_element_space_v<Field>)
         {
            ValidateSelectedCellMeshCompatibility(
               integration_domain.mesh,
               field);
         }
      }
      else if constexpr (is_partition_integration_domain_v<Domain>)
      {
         static_assert(
            is_mixed_finite_element_space_v<Field>,
            "An active field on PartitionIntegrationDomain must use a "
            "MixedFiniteElementSpace.");
         if constexpr (is_mixed_finite_element_space_v<Field>)
         {
            using DomainPartition =
               typename Domain::partition_type;
            using FieldPartition =
               typename Field::partition_type;
            using Compatibility =
               IsCompatiblePartitionTopology<
                  DomainPartition,
                  FieldPartition>;
            static_assert(
               Compatibility::value,
               "Active MixedFiniteElementSpace has an incompatible cell-part "
               "layout or an integration-domain face relation references a "
               "missing field cell part.");
            if constexpr (Compatibility::value)
            {
               GENDIL_VERIFY(
                  Compatibility::Check(
                     integration_domain.partition,
                     field.GetPartition()),
                  "Active MixedFiniteElementSpace cell topology is "
                  "incompatible with PartitionIntegrationDomain.");
            }
         }
      }
      else
      {
         static_assert(
            dependent_false_v<Domain>,
            "Active finite-element fields require a mesh or partition "
            "integration domain.");
      }
   }
}

template<
   class IntegrationDomain,
   class WFContext,
   class... Requirements>
void ValidateActiveFiniteElementFieldDomainCompatibility(
   const IntegrationDomain& integration_domain,
   const WFContext& wf_ctx,
   type_list<Requirements...>)
{
   (
      ValidateActiveFiniteElementFieldDomainCompatibility<Requirements>(
         integration_domain,
         wf_ctx),
      ...);
}

template<class Integrand, class WFContext>
void ValidateIntegrandWeakFormContext(
   const Integrand&,
   const WFContext& wf_ctx)
{
   using Form = std::remove_cvref_t<Integrand>;
   constexpr auto DomainName = Form::domain_type::name;
   using Context = std::remove_cvref_t<WFContext>;
   static_assert(
      Context::template has_domain<DomainName>(),
      "Weak form integrand requires an integration domain registered under "
      "its domain name.");

   if constexpr (Context::template has_domain<DomainName>())
   {
      const auto& normalized_domain =
         wf_ctx.template domain<DomainName>();
      const auto& integration_domain = normalized_domain.domain;
      using Domain = std::remove_cvref_t<decltype(integration_domain)>;

      if constexpr (
         InteriorFacetIntegrand<Form> &&
         is_partition_integration_domain_v<Domain>)
      {
         static_assert(
            Context::template has_interior_face_domain<DomainName>(),
            "InteriorFacets<Name>: PartitionIntegrationDomain has no interior "
            "face parts; partition domains do not fall back to local-facet "
            "traversal.");
      }
      if constexpr (
         BoundaryFacetIntegrand<Form> &&
         is_partition_integration_domain_v<Domain>)
      {
         static_assert(
            Context::template has_boundary_face_domain<DomainName>(),
            "BoundaryFacets<Name>: PartitionIntegrationDomain has no boundary "
            "face parts; partition domains do not fall back to local-facet "
            "traversal.");
      }

      using ActiveFields =
         active_finite_element_field_requirements_t<Form>;
      ValidateActiveFiniteElementFieldDomainCompatibility(
         integration_domain,
         wf_ctx,
         ActiveFields{});
   }
}

template<class Map, class WFContext>
void ValidateWeakFormContextEntries(
   const SumFormExpr<Map>& weak_form,
   const WFContext& wf_ctx)
{
   std::apply(
      [&] (const auto&... entries)
      {
         (
            ValidateIntegrandWeakFormContext(entries.value, wf_ctx),
            ...);
      },
      weak_form.map.entries);
}

template<class Domain, FieldExpr Expr, class WFContext>
void ValidateWeakFormContextEntries(
   const Integrand<Domain, Expr>& integrand,
   const WFContext& wf_ctx)
{
   ValidateIntegrandWeakFormContext(integrand, wf_ctx);
}

template<class WeakForm, class WFContext>
void ValidateWeakFormContext(
   const WeakForm& weak_form,
   const WFContext& wf_ctx)
{
   using namespace generic_operator_detail;
   constexpr auto DomainKind =
      generic_operator_domain_kind_v<WeakForm, WFContext>;

   if constexpr (DomainKind == GenericOperatorDomainKind::Invalid)
   {
      static_assert(
         DomainKind != GenericOperatorDomainKind::Invalid,
         "Weak form integrand requires an integration domain registered "
         "under its domain name.");
   }
   else if constexpr (DomainKind == GenericOperatorDomainKind::Mixed)
   {
      static_assert(
         DomainKind != GenericOperatorDomainKind::Mixed,
         "GenericOperator does not support mixing MeshIntegrationDomain and "
         "PartitionIntegrationDomain entries in one weak form.");
   }
   else
   {
      ValidateWeakFormContextEntries(weak_form, wf_ctx);
   }
}

template<
   class FieldKey,
   restricted_execution_domain_kind Kind,
   size_t Index,
   class FieldSpace,
   class SelectedDomain>
constexpr decltype(auto) RestrictFiniteElementFieldSpace(
   const FieldSpace& field_space,
   const SelectedDomain& selected)
{
   using Field = std::remove_cvref_t<FieldSpace>;
   using Selection = std::remove_cvref_t<SelectedDomain>;
   constexpr bool IsPartitioned =
      is_partitioned_selected_execution_domain_v<Selection>;

   if constexpr (is_mixed_finite_element_space_v<Field>)
   {
      static_assert(
         IsPartitioned,
         "Active MixedFiniteElementSpace field requires a "
         "PartitionIntegrationDomain; mesh domains require homogeneous "
         "finite-element fields.");

      if constexpr (Kind == restricted_execution_domain_kind::cell)
      {
         static_assert(
            Index < Field::num_cell_spaces,
            "Active MixedFiniteElementSpace field does not contain the "
            "selected integration-domain cell-part index.");
         if constexpr (Index < Field::num_cell_spaces)
         {
            const auto& cell_space =
               field_space.template GetCellFiniteElementSpace<Index>();
            ValidateSelectedCellMeshCompatibility(
               selected.GetCellMesh(),
               cell_space);
            return cell_space;
         }
      }
      else if constexpr (
         Kind == restricted_execution_domain_kind::interior_face)
      {
         constexpr size_t MinusI = Selection::minus_cell_part_index;
         constexpr size_t PlusI = Selection::plus_cell_part_index;
         static_assert(
            MinusI < Field::num_cell_spaces,
            "Active MixedFiniteElementSpace field does not contain the "
            "minus cell-part index referenced by the integration-domain "
            "interior face part.");
         static_assert(
            PlusI < Field::num_cell_spaces,
            "Active MixedFiniteElementSpace field does not contain the "
            "plus cell-part index referenced by the integration-domain "
            "interior face part.");

         if constexpr (
            MinusI < Field::num_cell_spaces &&
            PlusI < Field::num_cell_spaces)
         {
            const auto& minus_space =
               field_space.template GetCellFiniteElementSpace<MinusI>();
            const auto& plus_space =
               field_space.template GetCellFiniteElementSpace<PlusI>();
            ValidateSelectedCellMeshCompatibility(
               selected.GetMinusCellMesh(),
               minus_space);
            ValidateSelectedCellMeshCompatibility(
               selected.GetPlusCellMesh(),
               plus_space);

            using FacePart = typename Selection::face_part_type;
            using MinusSpace = std::remove_cvref_t<decltype(minus_space)>;
            using PlusSpace = std::remove_cvref_t<decltype(plus_space)>;
            return InteriorFaceFieldBinding<
               FacePart,
               MinusSpace,
               PlusSpace>{
                  FacePart{ selected.GetInteriorFacePart() },
                  MinusSpace{ minus_space },
                  PlusSpace{ plus_space } };
         }
      }
      else
      {
         constexpr size_t CellI = Selection::cell_part_index;
         static_assert(
            CellI < Field::num_cell_spaces,
            "Active MixedFiniteElementSpace field does not contain the "
            "cell-part index referenced by the integration-domain boundary "
            "face part.");

         if constexpr (CellI < Field::num_cell_spaces)
         {
            const auto& cell_space =
               field_space.template GetCellFiniteElementSpace<CellI>();
            ValidateSelectedCellMeshCompatibility(
               selected.GetCellMesh(),
               cell_space);

            using FacePart = typename Selection::face_part_type;
            using CellSpace = std::remove_cvref_t<decltype(cell_space)>;
            return BoundaryFaceFieldBinding<FacePart, CellSpace>{
               FacePart{ selected.GetBoundaryFacePart() },
               CellSpace{ cell_space } };
         }
      }
   }
   else if constexpr (is_cell_finite_element_space_v<Field>)
   {
      static_assert(
         Kind == restricted_execution_domain_kind::cell &&
         !IsPartitioned,
         "Active homogeneous FiniteElementSpace field requires a "
         "MeshIntegrationDomain and cannot be used for partition/global-face "
         "execution.");

      if constexpr (
         Kind == restricted_execution_domain_kind::cell &&
         !IsPartitioned)
      {
         ValidateSelectedCellMeshCompatibility(
            selected.GetCellMesh(),
            field_space);
         return field_space;
      }
   }
   else
   {
      static_assert(
         dependent_false_v<Field>,
         "Active finite-element field must use a homogeneous "
         "FiniteElementSpace or MixedFiniteElementSpace.");
   }
}

template<class Requirement, class WFContext, class SelectedDomain>
constexpr auto MakeRestrictedFiniteElementFieldEntry(
   const WFContext& wf_ctx,
   const SelectedDomain& selected)
{
   constexpr auto Name = Requirement::name;
   using Context = std::remove_cvref_t<WFContext>;
   static_assert(
      Context::template has_fe_field<Name>(),
      "WeakFormContext is missing an active trial, test, or coefficient "
      "finite-element field required by the weak form.");

   if constexpr (Context::template has_fe_field<Name>())
   {
      constexpr auto Kind =
         restricted_execution_domain_kind_v<SelectedDomain>;
      constexpr size_t Index =
         restricted_execution_domain_index_v<SelectedDomain>;
      const auto& fev = wf_ctx.template fe_field<Name>();
      decltype(auto) restricted_space =
         RestrictFiniteElementFieldSpace<
            FiniteElementFieldKey<Name>,
            Kind,
            Index>(
               fev.space,
               selected);
      return MakeFiniteElementField<Name>(restricted_space, fev.dofs);
   }
}

template<class WFContext, class SelectedDomain, class... Requirements>
constexpr auto MakeRestrictedFiniteElementFieldEntries(
   const WFContext& wf_ctx,
   const SelectedDomain& selected,
   type_list<Requirements...>)
{
   return std::tuple{
      MakeRestrictedFiniteElementFieldEntry<Requirements>(
         wf_ctx,
         selected)... };
}

template<
   class WeakForm,
   class WFContext,
   class DomainTag,
   class SelectedDomain>
constexpr auto MakeRestrictedWeakFormContext(
   const WFContext& wf_ctx,
   DomainTag domain_tag,
   const SelectedDomain& selected)
{
   using ActiveFields =
      active_finite_element_field_requirements_t<
         std::remove_cvref_t<WeakForm>>;

   auto domain_entries =
      MakeRestrictedDomainEntries(domain_tag, selected);
   auto fe_field_entries =
      MakeRestrictedFiniteElementFieldEntries(
         wf_ctx,
         selected,
         ActiveFields{});
   auto entries = std::tuple_cat(
      std::move(domain_entries),
      std::move(fe_field_entries),
      wf_ctx.fields.entries);

   return std::apply(
      [] (auto&&... entries_)
      {
         return MakeWeakFormContext(
            std::forward<decltype(entries_)>(entries_)...);
      },
      std::move(entries));
}

} // namespace gendil

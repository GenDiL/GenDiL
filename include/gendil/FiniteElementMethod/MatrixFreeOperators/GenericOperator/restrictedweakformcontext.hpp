// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/globalfacefiniteelementspace.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/domainfiniteelementspaceiteration.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <tuple>
#include <type_traits>

namespace gendil {

enum class restricted_execution_batch_kind
{
   cell,
   interior_face,
   boundary_face
};

template<class Batch>
struct restricted_execution_batch_traits;

template<StaticString Name, size_t CellI, class CellSpace>
struct restricted_execution_batch_traits<
   CellExecutionBatch<Name, CellI, CellSpace>>
{
   static constexpr auto kind = restricted_execution_batch_kind::cell;
   static constexpr size_t index = CellI;
};

template<
   StaticString Name,
   size_t FaceI,
   class BoundaryFaceSpace>
struct restricted_execution_batch_traits<
   BoundaryFaceExecutionBatch<
      Name,
      FaceI,
      BoundaryFaceSpace>>
{
   static constexpr auto kind = restricted_execution_batch_kind::boundary_face;
   static constexpr size_t index = FaceI;
};

template<
   StaticString Name,
   size_t FaceI,
   class InteriorFaceSpace>
struct restricted_execution_batch_traits<
   InteriorFaceExecutionBatch<
      Name,
      FaceI,
      InteriorFaceSpace>>
{
   static constexpr auto kind = restricted_execution_batch_kind::interior_face;
   static constexpr size_t index = FaceI;
};

template<class Batch>
inline constexpr auto restricted_execution_batch_kind_v =
   restricted_execution_batch_traits<std::remove_cvref_t<Batch>>::kind;

template<class Batch>
inline constexpr size_t restricted_execution_batch_index_v =
   restricted_execution_batch_traits<std::remove_cvref_t<Batch>>::index;

template<class Space>
constexpr auto MakeRestrictedDomainEntryFromCellSpace(const Space& space)
{
   using SpaceType = std::remove_cvref_t<Space>;
   return CellIntegrationDomain<SpaceType>{ SpaceType{ space } };
}

template<StaticString Name, size_t CellI, class CellSpace>
constexpr auto MakeRestrictedDomainEntries(
   Cells<Name>,
   const CellExecutionBatch<Name, CellI, CellSpace>& batch)
{
   using Space = std::remove_cvref_t<CellSpace>;
   using Domain = CellIntegrationDomain<Space>;
   return std::tuple{
      Entry<DomainKey<Name>, Domain>{
         MakeRestrictedDomainEntryFromCellSpace(
            batch.GetCellFiniteElementSpace()) } };
}

template<StaticString Name, size_t CellI, class CellSpace>
constexpr auto MakeRestrictedDomainEntries(
   InteriorFacets<Name>,
   const CellExecutionBatch<Name, CellI, CellSpace>& batch)
{
   using Space = std::remove_cvref_t<CellSpace>;
   using Domain = CellIntegrationDomain<Space>;
   return std::tuple{
      Entry<DomainKey<Name>, Domain>{
         MakeRestrictedDomainEntryFromCellSpace(
            batch.GetCellFiniteElementSpace()) } };
}

template<
   StaticString Name,
   size_t FaceI,
   class InteriorFaceSpace>
constexpr auto MakeRestrictedDomainEntries(
   InteriorFacets<Name>,
   const InteriorFaceExecutionBatch<
      Name,
      FaceI,
      InteriorFaceSpace>& batch)
{
   using CellSpace =
      std::remove_cvref_t<decltype(batch.GetMinusCellFiniteElementSpace())>;
   using CellDomain = CellIntegrationDomain<CellSpace>;
   using FaceSpace = std::remove_cvref_t<InteriorFaceSpace>;
   using FaceDomain = InteriorFaceIntegrationDomain<FaceSpace>;
   return std::tuple{
      Entry<DomainKey<Name>, CellDomain>{
         MakeRestrictedDomainEntryFromCellSpace(
            batch.GetMinusCellFiniteElementSpace()) },
      Entry<InteriorFaceDomainKey<Name>, FaceDomain>{
         FaceDomain{
            FaceSpace{ batch.GetInteriorFaceFiniteElementSpace() } } } };
}

template<StaticString Name, size_t CellI, class CellSpace>
constexpr auto MakeRestrictedDomainEntries(
   BoundaryFacets<Name>,
   const CellExecutionBatch<Name, CellI, CellSpace>& batch)
{
   using Space = std::remove_cvref_t<CellSpace>;
   using Domain = CellIntegrationDomain<Space>;
   return std::tuple{
      Entry<DomainKey<Name>, Domain>{
         MakeRestrictedDomainEntryFromCellSpace(
            batch.GetCellFiniteElementSpace()) } };
}

template<
   StaticString Name,
   size_t FaceI,
   class BoundaryFaceSpace>
constexpr auto MakeRestrictedDomainEntries(
   BoundaryFacets<Name>,
   const BoundaryFaceExecutionBatch<
      Name,
      FaceI,
      BoundaryFaceSpace>& batch)
{
   using CellSpace =
      std::remove_cvref_t<decltype(batch.GetCellFiniteElementSpace())>;
   using CellDomain = CellIntegrationDomain<CellSpace>;
   using FaceSpace = std::remove_cvref_t<BoundaryFaceSpace>;
   using FaceDomain = BoundaryFaceIntegrationDomain<FaceSpace>;
   return std::tuple{
      Entry<DomainKey<Name>, CellDomain>{
         MakeRestrictedDomainEntryFromCellSpace(
            batch.GetCellFiniteElementSpace()) },
      Entry<BoundaryFaceDomainKey<Name>, FaceDomain>{
         FaceDomain{
            FaceSpace{ batch.GetBoundaryFaceFiniteElementSpace() } } } };
}

template<class FieldKey, class FieldFaceSpace, class ExecutionFaceSpace>
constexpr void ValidateRestrictedGlobalFacetFieldBinding(
   const FieldFaceSpace& /*field_face_space*/,
   const ExecutionFaceSpace& /*execution_face_space*/)
{
   using Field = std::remove_cvref_t<FieldFaceSpace>;
   using Execution = std::remove_cvref_t<ExecutionFaceSpace>;

   if constexpr (is_boundary_face_finite_element_space_v<Execution>)
   {
      if constexpr (!is_boundary_face_finite_element_space_v<Field>)
      {
         static_assert(
            dependent_false_v<Field>,
            "FiniteElementField<Name> cannot be restricted to the selected "
            "BoundaryFacets<Name> face batch. Global facet execution requires "
            "fields to bind to the corresponding boundary face finite element "
            "space entry; volume finite element spaces are valid only in the "
            "local/cell-owned facet path.");
      }
      else
      {
         static_assert(
            std::is_same_v<
               typename Field::face_mesh_type,
               typename Execution::face_mesh_type>,
            "FiniteElementField<Name> boundary face finite element space is "
            "not compatible with the selected BoundaryFacets<Name> execution "
            "face batch. Stage 1 checks boundary/interior kind and "
            "face_mesh_type as a static face kind/family compatibility check, "
            "and relies on shared tuple ordering for batch identity; full "
            "runtime topology identity is deferred to Partition metadata.");
      }
   }
   else if constexpr (is_interior_face_finite_element_space_v<Execution>)
   {
      if constexpr (!is_interior_face_finite_element_space_v<Field>)
      {
         static_assert(
            dependent_false_v<Field>,
            "FiniteElementField<Name> cannot be restricted to the selected "
            "InteriorFacets<Name> face batch. Global facet execution requires "
            "fields to bind to the corresponding interior face finite element "
            "space entry; volume finite element spaces are valid only in the "
            "local/cell-owned facet path.");
      }
      else
      {
         static_assert(
            std::is_same_v<
               typename Field::face_mesh_type,
               typename Execution::face_mesh_type>,
            "FiniteElementField<Name> interior face finite element space is "
            "not compatible with the selected InteriorFacets<Name> execution "
            "face batch. Stage 1 checks boundary/interior kind and "
            "face_mesh_type as a static face kind/family compatibility check, "
            "and relies on shared tuple ordering for batch identity; full "
            "runtime topology identity is deferred to Partition metadata.");
      }
   }
   else
   {
      static_assert(
         dependent_false_v<Execution>,
         "ValidateRestrictedGlobalFacetFieldBinding requires a boundary or "
         "interior face finite element space execution batch.");
   }
}

template<
   class FieldKey,
   restricted_execution_batch_kind Kind,
   size_t Index,
   class FieldSpace,
   class Batch>
constexpr decltype(auto) RestrictFiniteElementFieldSpace(
   const FieldSpace& field_space,
   const Batch& batch)
{
   using FieldSpaceType = std::remove_cvref_t<FieldSpace>;

   if constexpr (is_mixed_finite_element_space_v<FieldSpaceType>)
   {
      if constexpr (Kind == restricted_execution_batch_kind::cell)
      {
         static_assert(
            Index < FieldSpaceType::num_cell_spaces,
            "FiniteElementField<Name> cannot be restricted to Cells<Name> "
            "batch Index. Stage 1 requires every registered finite element "
            "field to be restrictable to every selected execution batch.");
         return field_space.template GetCellFiniteElementSpace<Index>();
      }
      else if constexpr (Kind == restricted_execution_batch_kind::interior_face)
      {
         static_assert(
            Index < FieldSpaceType::num_interior_face_spaces,
            "FiniteElementField<Name> cannot be restricted to "
            "InteriorFacets<Name> face batch Index. Stage 1 requires every "
            "registered finite element field to be restrictable to every "
            "selected execution batch.");
         decltype(auto) selected_space =
            field_space.template GetInteriorFaceFiniteElementSpace<Index>();
         ValidateRestrictedGlobalFacetFieldBinding<FieldKey>(
            selected_space,
            batch.GetInteriorFaceFiniteElementSpace());
         return selected_space;
      }
      else
      {
         static_assert(
            Index < FieldSpaceType::num_boundary_face_spaces,
            "FiniteElementField<Name> cannot be restricted to "
            "BoundaryFacets<Name> face batch Index. Stage 1 requires every "
            "registered finite element field to be restrictable to every "
            "selected execution batch.");
         decltype(auto) selected_space =
            field_space.template GetBoundaryFaceFiniteElementSpace<Index>();
         ValidateRestrictedGlobalFacetFieldBinding<FieldKey>(
            selected_space,
            batch.GetBoundaryFaceFiniteElementSpace());
         return selected_space;
      }
   }
   else if constexpr (is_cell_finite_element_space_v<FieldSpaceType>)
   {
      if constexpr (Kind == restricted_execution_batch_kind::cell)
      {
         using SelectedCellSpaceType =
            std::remove_cvref_t<
               decltype(batch.GetCellFiniteElementSpace())>;
         static_assert(
            std::is_same_v<
               typename FieldSpaceType::mesh_type,
               typename SelectedCellSpaceType::mesh_type>,
            "FiniteElementField<Name> cannot be restricted to Cells<Name> "
            "batch Index. Homogeneous Stage 1 finite element fields must be "
            "defined on the same mesh topology type as the selected execution "
            "batch.");
         return field_space;
      }
      else if constexpr (Kind == restricted_execution_batch_kind::interior_face)
      {
         static_assert(
            dependent_false_v<FieldSpaceType>,
            "FiniteElementField<Name> is bound to a volume finite element "
            "space in global facet execution for InteriorFacets<Name> face "
            "batch Index. Global facet execution requires fields to be "
            "restricted to the corresponding face finite element space entry. "
            "Volume finite element spaces are valid only in the "
            "local/cell-owned facet path.");
      }
      else
      {
         static_assert(
            dependent_false_v<FieldSpaceType>,
            "FiniteElementField<Name> is bound to a volume finite element "
            "space in global facet execution for BoundaryFacets<Name> face "
            "batch Index. Global facet execution requires fields to be "
            "restricted to the corresponding face finite element space entry. "
            "Volume finite element spaces are valid only in the "
            "local/cell-owned facet path.");
      }
   }
   else
   {
      static_assert(
         dependent_false_v<FieldSpaceType>,
         "FiniteElementField<Name> must be registered on a homogeneous cell "
         "finite element space or a MixedFiniteElementSpace for Stage 1 "
         "restricted-context execution.");
   }
}

template<class Batch, class FEFieldEntry>
constexpr auto MakeRestrictedFiniteElementFieldEntry(
   const FEFieldEntry& entry,
   const Batch& batch)
{
   using Key = entry_key_t<FEFieldEntry>;
   constexpr auto Kind = restricted_execution_batch_kind_v<Batch>;
   constexpr size_t Index = restricted_execution_batch_index_v<Batch>;
   const auto& fev = entry.value;
   decltype(auto) restricted_space =
      RestrictFiniteElementFieldSpace<Key, Kind, Index>(
         fev.space,
         batch);
   return MakeFiniteElementField<Key::name>(restricted_space, fev.dofs);
}

template<class WFContext, class Batch>
constexpr auto MakeRestrictedFiniteElementFieldEntries(
   const WFContext& wf_ctx,
   const Batch& batch)
{
   return std::apply(
      [&] (auto const&... entries)
      {
         return std::tuple{
            MakeRestrictedFiniteElementFieldEntry(entries, batch)... };
      },
      wf_ctx.fe_fields.entries);
}

template<
   StaticString TrialName,
   StaticString TestName,
   class WFContext,
   class DomainTag,
   class Batch>
constexpr auto MakeRestrictedWeakFormContext(
   const WFContext& wf_ctx,
   DomainTag domain_tag,
   const Batch& batch)
{
   (void)TrialName;
   (void)TestName;

   auto domain_entries =
      MakeRestrictedDomainEntries(domain_tag, batch);
   auto fe_field_entries =
      MakeRestrictedFiniteElementFieldEntries(wf_ctx, batch);
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

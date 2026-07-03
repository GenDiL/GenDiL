// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globalfacefieldbinding.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/domainfiniteelementspaceiteration.hpp"
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
   class FacePart,
   class CellSpace>
struct restricted_execution_batch_traits<
   BoundaryFaceExecutionBatch<
      Name,
      FaceI,
      FacePart,
      CellSpace>>
{
   static constexpr auto kind = restricted_execution_batch_kind::boundary_face;
   static constexpr size_t index = FaceI;
};

template<
   StaticString Name,
   size_t FaceI,
   class FacePart,
   class MinusCellSpace,
   class PlusCellSpace>
struct restricted_execution_batch_traits<
   InteriorFaceExecutionBatch<
      Name,
      FaceI,
      FacePart,
      MinusCellSpace,
      PlusCellSpace>>
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
   class FacePart,
   class MinusCellSpace,
   class PlusCellSpace>
constexpr auto MakeRestrictedDomainEntries(
   InteriorFacets<Name>,
   const InteriorFaceExecutionBatch<
      Name,
      FaceI,
      FacePart,
      MinusCellSpace,
      PlusCellSpace>& batch)
{
   using CellSpace =
      std::remove_cvref_t<decltype(batch.GetMinusCellFiniteElementSpace())>;
   using CellDomain = CellIntegrationDomain<CellSpace>;
   using FaceDomain =
      InteriorFaceIntegrationDomain<
         std::remove_cvref_t<decltype(batch)>>;
   return std::tuple{
      Entry<DomainKey<Name>, CellDomain>{
         MakeRestrictedDomainEntryFromCellSpace(
            batch.GetMinusCellFiniteElementSpace()) },
      Entry<InteriorFaceDomainKey<Name>, FaceDomain>{
         FaceDomain{ std::remove_cvref_t<decltype(batch)>{ batch } } } };
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
   class FacePart,
   class CellSpace>
constexpr auto MakeRestrictedDomainEntries(
   BoundaryFacets<Name>,
   const BoundaryFaceExecutionBatch<
      Name,
      FaceI,
      FacePart,
      CellSpace>& batch)
{
   using SelectedCellSpace =
      std::remove_cvref_t<decltype(batch.GetCellFiniteElementSpace())>;
   using CellDomain = CellIntegrationDomain<SelectedCellSpace>;
   using FaceDomain =
      BoundaryFaceIntegrationDomain<
         std::remove_cvref_t<decltype(batch)>>;
   return std::tuple{
      Entry<DomainKey<Name>, CellDomain>{
         MakeRestrictedDomainEntryFromCellSpace(
            batch.GetCellFiniteElementSpace()) },
      Entry<BoundaryFaceDomainKey<Name>, FaceDomain>{
         FaceDomain{ std::remove_cvref_t<decltype(batch)>{ batch } } } };
}

template<class FieldKey, size_t FaceI, class FieldSpace, class Batch>
constexpr void ValidateInteriorFieldBindingCompatibility(
   const FieldSpace&,
   const Batch&)
{
   using Field = std::remove_cvref_t<FieldSpace>;
   using ExecutionFacePart = typename std::remove_cvref_t<Batch>::face_part_type;
   static_assert(
      FaceI < Field::num_interior_face_parts,
      "FiniteElementField<Name> cannot be restricted to the selected "
      "InteriorFacets<Name> face batch. The field's MixedFiniteElementSpace "
      "does not contain the corresponding partition interior face part.");
   if constexpr (FaceI < Field::num_interior_face_parts)
   {
      using FieldFacePart =
         std::tuple_element_t<FaceI, typename Field::interior_face_parts_type>;
      static_assert(
         std::is_same_v<FieldFacePart, ExecutionFacePart>,
         "FiniteElementField<Name> partition interior face part is not "
         "compatible with the selected InteriorFacets<Name> execution batch. "
         "Global facet field binding requires the field and execution domain "
         "to share the same partition face topology at the selected index.");
   }
   static_assert(
      std::remove_cvref_t<Batch>::minus_cell_part_index < Field::num_cell_spaces,
      "FiniteElementField<Name> interior-face minus cell-part index is out "
      "of bounds for the field's MixedFiniteElementSpace.");
   static_assert(
      std::remove_cvref_t<Batch>::plus_cell_part_index < Field::num_cell_spaces,
      "FiniteElementField<Name> interior-face plus cell-part index is out "
      "of bounds for the field's MixedFiniteElementSpace.");
}

template<class FieldKey, size_t FaceI, class FieldSpace, class Batch>
constexpr void ValidateBoundaryFieldBindingCompatibility(
   const FieldSpace&,
   const Batch&)
{
   using Field = std::remove_cvref_t<FieldSpace>;
   using ExecutionFacePart = typename std::remove_cvref_t<Batch>::face_part_type;
   static_assert(
      FaceI < Field::num_boundary_face_parts,
      "FiniteElementField<Name> cannot be restricted to the selected "
      "BoundaryFacets<Name> face batch. The field's MixedFiniteElementSpace "
      "does not contain the corresponding partition boundary face part.");
   if constexpr (FaceI < Field::num_boundary_face_parts)
   {
      using FieldFacePart =
         std::tuple_element_t<FaceI, typename Field::boundary_face_parts_type>;
      static_assert(
         std::is_same_v<FieldFacePart, ExecutionFacePart>,
         "FiniteElementField<Name> partition boundary face part is not "
         "compatible with the selected BoundaryFacets<Name> execution batch. "
         "Global facet field binding requires the field and execution domain "
         "to share the same partition face topology at the selected index.");
   }
   static_assert(
      std::remove_cvref_t<Batch>::cell_part_index < Field::num_cell_spaces,
      "FiniteElementField<Name> boundary-face cell-part index is out of "
      "bounds for the field's MixedFiniteElementSpace.");
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
         ValidateInteriorFieldBindingCompatibility<FieldKey, Index>(
            field_space,
            batch);
         constexpr size_t MinusCellI =
            std::remove_cvref_t<Batch>::minus_cell_part_index;
         constexpr size_t PlusCellI =
            std::remove_cvref_t<Batch>::plus_cell_part_index;
         const auto& minus_space =
            field_space.template GetCellFiniteElementSpace<MinusCellI>();
         const auto& plus_space =
            field_space.template GetCellFiniteElementSpace<PlusCellI>();
         using FacePart = typename std::remove_cvref_t<Batch>::face_part_type;
         using MinusSpace = std::remove_cvref_t<decltype(minus_space)>;
         using PlusSpace = std::remove_cvref_t<decltype(plus_space)>;
         return InteriorFaceFieldBinding<
            FacePart,
            MinusSpace,
            PlusSpace>{
               FacePart{ batch.GetInteriorFacePart() },
               MinusSpace{ minus_space },
               PlusSpace{ plus_space } };
      }
      else
      {
         ValidateBoundaryFieldBindingCompatibility<FieldKey, Index>(
            field_space,
            batch);
         constexpr size_t CellI =
            std::remove_cvref_t<Batch>::cell_part_index;
         const auto& cell_space =
            field_space.template GetCellFiniteElementSpace<CellI>();
         using FacePart = typename std::remove_cvref_t<Batch>::face_part_type;
         using CellSpace = std::remove_cvref_t<decltype(cell_space)>;
         return BoundaryFaceFieldBinding<
            FacePart,
            CellSpace>{
               FacePart{ batch.GetBoundaryFacePart() },
               CellSpace{ cell_space } };
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
            "batch Index. Global facet execution requires fields to be bound "
            "to a MixedFiniteElementSpace sharing the selected partition. "
            "Volume finite element spaces are valid only in the "
            "local/cell-owned facet path.");
      }
      else
      {
         static_assert(
            dependent_false_v<FieldSpaceType>,
            "FiniteElementField<Name> is bound to a volume finite element "
            "space in global facet execution for BoundaryFacets<Name> face "
            "batch Index. Global facet execution requires fields to be bound "
            "to a MixedFiniteElementSpace sharing the selected partition. "
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

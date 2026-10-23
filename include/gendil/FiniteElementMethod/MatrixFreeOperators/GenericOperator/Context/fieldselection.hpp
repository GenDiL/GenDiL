// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Select each field's finite-element space for the cells or faces being integrated.
 *
 * A registered field may use one space on a mesh or different spaces on the cell
 * parts of a partition. Before evaluating a batch of cells or faces, execution
 * needs the corresponding field spaces and their DOF restrictions. The helpers
 * here select those spaces and check that their cell numbering matches the
 * integration domain.
 *
 * Callers normally select all required fields through MakeRestrictedWeakFormContext
 * (available through Context/context.hpp). Given a registered context, a selected
 * batch, and Requirements that include the field "u":
 * @code
 * auto selected = MakeRestrictedWeakFormContext<Requirements>(
 *    registered_context, selected_domain);
 * const auto& u_space = selected.GetSpace<"u">();
 * @endcode
 * Keep the registered context alive when selected fields borrow persistent data.
 *
 * A cell batch selects one volume space. A partition interior-face batch pairs
 * its connectivity with the minus and plus volume spaces; a boundary-face batch
 * pairs it with the adjacent volume space. The integration domain supplies the
 * face connectivity, so the field's own partition need not store the same face
 * parts. Mesh-local facets continue to use the volume space selected for their
 * cell traversal.
 *
 * Required field names and input/output roles are decided by the caller. This
 * layer selects spaces without binding vectors, gathering DOFs, or preparing
 * basis data. BindExecutionContext and MakeOperatorContext handle those later
 * storage-binding and preparation stages.
 */

#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/selectedexecutiondomain.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globalfacefieldbinding.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"
#include "gendil/Utilities/debug.hpp"

namespace gendil {

/// Batch topology used to choose a volume space or adjacent face spaces.
enum class restricted_execution_domain_kind
{
   cell,
   interior_face,
   boundary_face
};

/**
 * @brief Identify the kind and tuple position of a selected execution batch.
 *
 * `index` is the cell-part or face-part position for a partition, and zero for
 * whole-mesh cell traversal. It is not an individual cell or face number. For
 * face batches, the selected domain separately identifies the adjacent cell
 * parts. Only the selected-domain types below are supported.
 */
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

/// Batch kind, accepting a selected-domain type with const/reference qualifiers.
template<class Domain>
inline constexpr auto restricted_execution_domain_kind_v =
   restricted_execution_domain_traits<
      std::remove_cvref_t<Domain>>::kind;

/// Cell-part or face-part position carried by the selected-domain type.
template<class Domain>
inline constexpr size_t restricted_execution_domain_index_v =
   restricted_execution_domain_traits<
      std::remove_cvref_t<Domain>>::index;

/**
 * @brief Check that domain cell indices address the same cells in a field space.
 *
 * Matching mesh types and an available MeshIdentityTraits definition are required
 * at compile time. The host-side check then compares cell counts and indexed
 * topology identity through IsCompatibleMeshDomain; equal counts alone do not
 * establish compatibility. A mismatch fails GENDIL_VERIFY, including in Release
 * builds. This check reads the meshes without modifying them or accessing DOFs.
 */
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
   static_assert(
      !Compatibility::value || Compatibility::has_indexed_identity,
      "Active finite-element field mesh compatibility cannot establish "
      "indexed-topology identity. Provide MeshIdentityTraits for this mesh "
      "type.");
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

/**
 * @brief Obtain the field space or adjacent spaces needed by one selected batch.
 *
 * For a homogeneous field on a mesh, return that volume space. For a mixed field
 * on a partition, a cell batch selects its cell-part space; a face batch selects
 * the spaces named by the integration domain's face-to-cell relations. The
 * selected spaces retain their existing bases and DOF restrictions.
 *
 * Cell selections return const references into field_space. Keep field_space
 * alive while using such a reference; passing a temporary does not extend its
 * lifetime. Interior and boundary selections return InteriorFaceFieldBinding
 * and BoundaryFaceFieldBinding values containing copies of the selected face
 * part and spaces. Any storage borrowed by those objects must remain valid.
 *
 * Mixed spaces require partition selection; homogeneous spaces require a
 * non-partitioned cell selection. Invalid combinations or missing cell parts
 * fail at compile time. Every selected volume space is checked by
 * ValidateSelectedCellMeshCompatibility before it is returned or copied.
 *
 * @tparam FieldKey Caller-supplied field key; currently unused by this helper.
 * @tparam Kind Use restricted_execution_domain_kind_v<SelectedDomain>.
 * @tparam Index Use restricted_execution_domain_index_v<SelectedDomain>. Face
 * batches use their adjacent cell-part indices to select spaces, not this index.
 */
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

} // namespace gendil

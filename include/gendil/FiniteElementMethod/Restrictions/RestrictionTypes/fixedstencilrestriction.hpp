// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Scalar restrictions with compile-time row weights and runtime indices.
 *
 * A fixed-stencil definition describes the number of algebraic entries in
 * every native element row and the weight of every entry. The definition is
 * type-level metadata: a completed restriction stores only its borrowed
 * algebraic index map and its dimensions. This representation is useful for
 * manually supplied p-nonconforming constraints whose topology and
 * coefficients repeat on every element.
 *
 * Indices are ordered element-major, then native-row-major, then entry-major.
 * They are final coordinates in the field-wide algebraic vector; this
 * restriction performs no rebasing or ownership inference. Stencil weights
 * follow the native-row-major entry order for one element and are materialized
 * as compile-time constants during row visitation.
 *
 * A stencil definition is an empty type supplying two static `std::array`
 * values:
 *
 * @code{.cpp}
 * struct Q2Constraint
 * {
 *    inline static constexpr std::array<std::size_t, 3>
 *       row_entry_counts{ 1, 2, 1 };
 *    inline static constexpr std::array<Real, 4>
 *       entry_weights{ 1.0, 0.5, 0.5, 1.0 };
 * };
 * @endcode
 */

#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/fixedstencilrestrictiondetails.hpp"
#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionvalidation.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>

namespace gendil {

/** @brief Borrowed construction data for a fixed-stencil restriction. */
template <
   details::FixedStencilDefinition Stencil,
   class Index = GlobalIndex >
struct FixedStencilRestrictionSpecification
{
   static_assert(
      std::integral< Index > &&
         !std::same_as< std::remove_cv_t< Index >, bool >,
      "Fixed-stencil algebraic indices must use a non-Boolean integral type." );

   HostDevicePointer< const Index > indices;
   GlobalIndex index_count;
   GlobalIndex num_global_dofs;
   std::optional< GlobalIndex > algebraic_dof_extent{};

   constexpr FixedStencilRestrictionSpecification(
      const HostDevicePointer< const Index > indices_,
      const GlobalIndex index_count_,
      const GlobalIndex num_global_dofs_ )
      : indices( indices_ ),
        index_count( index_count_ ),
        num_global_dofs( num_global_dofs_ )
   { }

   constexpr FixedStencilRestrictionSpecification(
      const HostDevicePointer< const Index > indices_,
      const GlobalIndex index_count_,
      const GlobalIndex num_global_dofs_,
      const GlobalIndex algebraic_dof_extent_ )
      : indices( indices_ ),
        index_count( index_count_ ),
        num_global_dofs( num_global_dofs_ ),
        algebraic_dof_extent( algebraic_dof_extent_ )
   { }
};

/**
 * @brief Completed scalar restriction with static weights and runtime indices.
 *
 * The stencil definition is not a data member. Consequently restriction and
 * kernel captures contain no row-count, prefix, or weight arrays.
 */
template <
   StaticDofShape DofShape,
   details::FixedStencilDefinition Stencil,
   class Index = GlobalIndex >
struct FixedStencilRestriction
{
   static_assert(
      details::ValidateFixedStencilForDofShape< Stencil, DofShape >() );

   using dof_shape_type = DofShape;
   using stencil_type = Stencil;
   using index_type = Index;

   static constexpr Integer tensor_dim =
      static_cast< Integer >( dof_shape_type::size() );

   HostDevicePointer< const Index > indices;
   GlobalIndex index_count;
   GlobalIndex num_local_dofs;
   GlobalIndex num_global_dofs;
   GlobalIndex algebraic_dof_extent;
};

/** @brief Visit one native row with statically unrolled constexpr weights. */
template <
   StaticDofShape DofShape,
   details::FixedStencilDefinition Stencil,
   class Index,
   class Visitor >
GENDIL_HOST_DEVICE
constexpr void ForEachRestrictionEntry(
   const FixedStencilRestriction< DofShape, Stencil, Index > & restriction,
   const GlobalIndex element,
   const std::array< GlobalIndex, DofShape::size() > & local_dof,
   Visitor && visitor )
{
   GENDIL_ASSERT(
      details::DofIndexIsInBounds< DofShape >( local_dof ),
      "FixedStencilRestriction local tensor index is out of bounds." );
   const GlobalIndex native_row = FlattenMultiIndex< DofShape >( local_dof );
   GENDIL_ASSERT(
      element * Product( DofShape{} ) + native_row <
         restriction.num_local_dofs,
      "FixedStencilRestriction element-local row is out of bounds." );
   details::DispatchFixedStencilRow< 0 >(
      restriction,
      element,
      native_row,
      visitor );
}

template < StaticDofShape DofShape, class Stencil, class Index >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfLocalDofs(
   const FixedStencilRestriction< DofShape, Stencil, Index > & restriction )
{
   return restriction.num_local_dofs;
}

template < StaticDofShape DofShape, class Stencil, class Index >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfGlobalDofs(
   const FixedStencilRestriction< DofShape, Stencil, Index > & restriction )
{
   return restriction.num_global_dofs;
}

template < StaticDofShape DofShape, class Stencil, class Index >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetAlgebraicDofExtent(
   const FixedStencilRestriction< DofShape, Stencil, Index > & restriction )
{
   return restriction.algebraic_dof_extent;
}

/**
 * @brief Restriction-wide count when every stencil row is uniform.
 *
 * A nonuniform stencil reports the dynamic sentinel here while retaining an
 * exact compile-time count for every `NativeRestrictionRowTag` below.
 */
template < StaticDofShape DofShape, class Stencil, class Index >
inline constexpr size_t static_restriction_entry_count_v<
   FixedStencilRestriction< DofShape, Stencil, Index > > =
      details::FixedStencilUniformEntryCount< Stencil >();

template <
   StaticDofShape DofShape,
   class Stencil,
   class Index,
   size_t Component,
   GlobalIndex LocalDofIndex >
inline constexpr size_t static_restriction_entry_count_for_v<
   FixedStencilRestriction< DofShape, Stencil, Index >,
   NativeRestrictionRowTag< Component, LocalDofIndex > > = []
   {
      static_assert(
         Component == 0,
         "Scalar fixed-stencil restrictions only have component zero." );
      static_assert(
         LocalDofIndex < Product( DofShape{} ),
         "Fixed-stencil native local DoF index is out of range." );
      return details::FixedStencilRowEntryCount<
         Stencil,
         static_cast< size_t >( LocalDofIndex ) >();
   }();

template < StaticDofShape DofShape, class Stencil, class Index >
inline constexpr bool restriction_may_share_global_dofs_v<
   FixedStencilRestriction< DofShape, Stencil, Index > > = true;

template < StaticDofShape DofShape, class Stencil, class Index >
inline constexpr bool restriction_rows_have_unique_coordinates_v<
   FixedStencilRestriction< DofShape, Stencil, Index > > =
      details::FixedStencilRowsHaveAtMostOneEntry< Stencil >();

template < StaticDofShape DofShape, class Stencil, class Index >
struct restriction_supports_element_reference_view<
   FixedStencilRestriction< DofShape, Stencil, Index > > : std::false_type { };

/** @brief Validate that the borrowed index map is resident for execution. */
template <
   bool OnDevice,
   StaticDofShape DofShape,
   class Stencil,
   class Index >
void ValidateRestrictionMemoryAccess(
   const FixedStencilRestriction< DofShape, Stencil, Index > & restriction,
   const GlobalIndex active_row_count )
{
   constexpr bool has_entries =
      details::FixedStencilEntriesPerElement< Stencil >() != 0;
   if constexpr ( OnDevice )
   {
#ifdef GENDIL_USE_DEVICE
      GENDIL_VERIFY(
         active_row_count == 0 || !has_entries ||
            restriction.indices.device_pointer != nullptr,
         "Device fixed-stencil restriction access requires a device-resident index map." );
#else
      static_assert(
         !OnDevice,
         "Device fixed-stencil restriction access requires GenDiL device support." );
#endif
   }
   else
   {
      GENDIL_VERIFY(
         active_row_count == 0 || !has_entries ||
            restriction.indices.host_pointer != nullptr,
         "Host fixed-stencil restriction access requires a host-resident index map." );
   }
}

/** @brief Validate dimensions, exact map length, and host-visible indices. */
template < StaticDofShape DofShape, class Stencil, class Index >
void ValidateRestrictionRepresentation(
   const FixedStencilRestriction< DofShape, Stencil, Index > & restriction )
{
   constexpr GlobalIndex rows_per_element = Product( DofShape{} );
   constexpr GlobalIndex entries_per_element = static_cast< GlobalIndex >(
      details::FixedStencilEntriesPerElement< Stencil >() );
   GENDIL_VERIFY(
      rows_per_element != 0 &&
         restriction.num_local_dofs % rows_per_element == 0,
      "FixedStencilRestriction local-row count is incompatible with its DoF shape." );
   const GlobalIndex num_elements =
      restriction.num_local_dofs / rows_per_element;
   GENDIL_VERIFY(
      restriction.index_count == CheckedMultiply(
         num_elements,
         entries_per_element,
         "Fixed-stencil index count overflows GlobalIndex." ),
      "Fixed-stencil index count does not match the mesh and stencil definition." );

   bool has_resident_map = restriction.indices.host_pointer != nullptr;
#ifdef GENDIL_USE_DEVICE
   has_resident_map = has_resident_map ||
      restriction.indices.device_pointer != nullptr;
#endif
   GENDIL_VERIFY(
      restriction.index_count == 0 || has_resident_map,
      "A nonempty fixed-stencil restriction requires a resident index map." );

   if ( restriction.indices.host_pointer != nullptr )
   {
      for ( GlobalIndex entry = 0; entry < restriction.index_count; ++entry )
      {
         const Index mapped_dof = restriction.indices.host_pointer[entry];
         if constexpr ( std::signed_integral< Index > )
         {
            GENDIL_VERIFY(
               mapped_dof >= 0,
               "Fixed-stencil index map contains a negative algebraic index." );
         }
         if constexpr ( sizeof( Index ) > sizeof( GlobalIndex ) )
         {
            using UnsignedIndex = std::make_unsigned_t< Index >;
            GENDIL_VERIFY(
               static_cast< UnsignedIndex >( mapped_dof ) <=
                  std::numeric_limits< GlobalIndex >::max(),
               "Fixed-stencil algebraic index exceeds GlobalIndex." );
         }
         const GlobalIndex algebraic_dof =
            static_cast< GlobalIndex >( mapped_dof );
         GENDIL_VERIFY(
            algebraic_dof < restriction.algebraic_dof_extent,
            "Fixed-stencil index exceeds the algebraic DoF extent." );
      }
   }
}

/** @brief Complete a fixed-stencil specification for a scalar finite element. */
template <
   typename Mesh,
   typename FiniteElement,
   details::FixedStencilDefinition Stencil,
   class Index >
auto MakeElementDoFRestriction(
   const Mesh & mesh,
   const FiniteElement &,
   const FixedStencilRestrictionSpecification< Stencil, Index > & specification )
{
   using ShapeFunctions = typename FiniteElement::shape_functions;
   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "FixedStencilRestrictionSpecification is scalar; compose completed scalar restrictions with VectorRestriction for vector finite elements." );
   using DofShape = finite_element_dof_shape_t< ShapeFunctions >;
   static_assert(
      details::ValidateFixedStencilForDofShape< Stencil, DofShape >() );

   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( mesh.GetNumberOfCells() );
   const GlobalIndex num_local_dofs =
      CheckedElementLocalDofCount< ShapeFunctions >( num_elements );
   constexpr GlobalIndex entries_per_element = static_cast< GlobalIndex >(
      details::FixedStencilEntriesPerElement< Stencil >() );
   GENDIL_VERIFY(
      specification.index_count == CheckedMultiply(
         num_elements,
         entries_per_element,
         "Fixed-stencil index count overflows GlobalIndex." ),
      "Fixed-stencil index count does not match the mesh and stencil definition." );

   FixedStencilRestriction< DofShape, Stencil, Index > result{
      specification.indices,
      specification.index_count,
      num_local_dofs,
      specification.num_global_dofs,
      specification.algebraic_dof_extent.value_or(
         specification.num_global_dofs ) };
   ValidateElementDoFRestriction( result );
   return result;
}

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Contiguous one-entry L2 restriction specifications and maps.
 */

#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionunitweight.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionvalidation.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/vectorrestriction.hpp"

#include <optional>
#include <tuple>
#include <utility>

namespace gendil {

/**
 * @brief Construction input for a direct contiguous L2 restriction.
 *
 * An omitted shift selects zero for a standalone space. An omitted algebraic
 * extent selects the smallest extent containing the shifted DoF interval.
 */
struct ContiguousL2RestrictionSpecification
{
   std::optional< GlobalIndex > shift{};
   std::optional< GlobalIndex > algebraic_dof_extent{};

   constexpr ContiguousL2RestrictionSpecification() = default;

   constexpr explicit ContiguousL2RestrictionSpecification(
      const GlobalIndex shift_ )
      : shift( shift_ )
   { }

   constexpr ContiguousL2RestrictionSpecification(
      const GlobalIndex shift_,
      const GlobalIndex algebraic_dof_extent_ )
      : shift( shift_ ),
        algebraic_dof_extent( algebraic_dof_extent_ )
   { }
};

/**
 * @brief Shifted, direct, one-entry restriction for a scalar L2 DoF shape.
 *
 * Element-local rows are flattened in the order defined by @c DofShape and
 * mapped to the contiguous interval beginning at @c shift.
 */
template < StaticDofShape DofShape >
struct ContiguousL2Restriction
{
   using dof_shape_type = DofShape;

   static constexpr Integer tensor_dim =
      static_cast< Integer >( dof_shape_type::size() );

   const GlobalIndex shift;
   const GlobalIndex num_local_dofs;
   const GlobalIndex algebraic_dof_extent;
};

/** @brief Visit the single unit-weight entry of one contiguous L2 row. */
template < StaticDofShape DofShape, typename Visitor >
GENDIL_HOST_DEVICE
constexpr void ForEachRestrictionEntry(
   const ContiguousL2Restriction< DofShape > & restriction,
   const GlobalIndex element,
   const std::array<
      GlobalIndex,
      ContiguousL2Restriction< DofShape >::tensor_dim > &
      local_dof,
   Visitor && visitor )
{
   constexpr GlobalIndex local_dofs = Product( DofShape{} );
   GENDIL_ASSERT(
      details::DofIndexIsInBounds< DofShape >( local_dof ),
      "ContiguousL2Restriction local tensor index is out of bounds." );
   const GlobalIndex row =
      element * local_dofs + FlattenMultiIndex< DofShape >( local_dof );
   GENDIL_ASSERT(
      row < restriction.num_local_dofs,
      "ContiguousL2Restriction element-local row is out of bounds." );
   const GlobalIndex algebraic_dof = restriction.shift + row;
   GENDIL_ASSERT(
      algebraic_dof < restriction.algebraic_dof_extent,
      "ContiguousL2Restriction global DoF is out of bounds." );
   std::forward< Visitor >( visitor )(
      algebraic_dof,
      RestrictionUnitWeight{} );
}

/** @brief Return the number of element-local restriction rows. */
template < StaticDofShape DofShape >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfLocalDofs(
   const ContiguousL2Restriction< DofShape > & restriction )
{
   return restriction.num_local_dofs;
}

/** @brief Return the number of logical global L2 DoFs. */
template < StaticDofShape DofShape >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfGlobalDofs(
   const ContiguousL2Restriction< DofShape > & restriction )
{
   return restriction.num_local_dofs;
}

/** @brief Return the size of the containing algebraic coordinate space. */
template < StaticDofShape DofShape >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetAlgebraicDofExtent(
   const ContiguousL2Restriction< DofShape > & restriction )
{
   return restriction.algebraic_dof_extent;
}

template < StaticDofShape DofShape >
inline constexpr size_t static_restriction_entry_count_v<
   ContiguousL2Restriction< DofShape > > = 1;

template < StaticDofShape DofShape >
inline constexpr bool restriction_may_share_global_dofs_v<
   ContiguousL2Restriction< DofShape > > = false;

template < StaticDofShape... ComponentDofShapes >
inline constexpr bool restriction_may_share_global_dofs_v<
   VectorRestriction<
      ContiguousL2Restriction< ComponentDofShapes >... > > = false;

template < StaticDofShape DofShape >
struct restriction_supports_element_reference_view<
   ContiguousL2Restriction< DofShape > > : std::true_type { };

/** @brief Contiguous restrictions carry no external mapping storage. */
template < bool OnDevice, StaticDofShape DofShape >
void ValidateRestrictionMemoryAccess(
   const ContiguousL2Restriction< DofShape > &,
   const GlobalIndex )
{
   static_cast< void >( OnDevice );
}

/** @brief Build the component restrictions for a vector L2 direct sum. */
template < typename ShapeFunctions, size_t... Component >
auto MakeVectorL2Restriction(
   const GlobalIndex num_elements,
   const GlobalIndex base_shift,
   const GlobalIndex algebraic_dof_extent,
   std::index_sequence< Component... > )
{
   using Restriction = VectorL2Restriction<
      ContiguousL2Restriction<
         component_dof_shape_t< ShapeFunctions, Component > >... >;

   const GlobalIndex num_local_dofs =
      CheckedElementLocalDofCount< ShapeFunctions >( num_elements );

   const Restriction restriction{
      std::tuple{
         ContiguousL2Restriction<
            component_dof_shape_t< ShapeFunctions, Component > >{
               CheckedAdd(
                  base_shift,
                  CheckedVectorComponentOffset<
                     ShapeFunctions,
                     Component >( num_elements ),
                  "Vector restriction global component shift overflow." ),
               CheckedMultiply(
                  num_elements,
                  static_cast< GlobalIndex >(
                     Product(
                        component_dof_shape_t<
                           ShapeFunctions,
                           Component >{} ) ),
                  "Vector restriction component local DoF count overflow." ),
               algebraic_dof_extent }... },
      num_local_dofs,
      num_local_dofs,
      algebraic_dof_extent };

   GlobalIndex next_shift = base_shift;
   auto validate_component = [&]< size_t C >(
      std::integral_constant< size_t, C > )
   {
      const auto & component =
         GetComponentRestriction< C >( restriction );
      GENDIL_VERIFY(
         component.shift == next_shift,
         "Vector L2 component ranges must form a disjoint direct sum." );
      next_shift = CheckedAdd(
         next_shift,
         GetNumberOfLocalDofs( component ),
         "Vector L2 component range overflow." );
   };
   ( validate_component(
        std::integral_constant< size_t, Component >{} ),
     ... );
   return restriction;
}

/**
 * @brief Build a scalar or vector contiguous L2 restriction with final
 * placement.
 */
template < typename Mesh, typename FiniteElement >
auto MakeContiguousL2Restriction(
   const Mesh & mesh,
   const FiniteElement &,
   const GlobalIndex shift,
   const GlobalIndex algebraic_dof_extent )
{
   using ShapeFunctions = typename FiniteElement::shape_functions;
   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( mesh.GetNumberOfCells() );
   const GlobalIndex num_local_dofs =
      CheckedElementLocalDofCount< ShapeFunctions >( num_elements );
   GENDIL_VERIFY(
      CheckedAdd(
         shift,
         num_local_dofs,
         "Contiguous L2 restriction addressed interval overflow." ) <=
         algebraic_dof_extent,
      "Contiguous L2 restriction addressed interval exceeds its algebraic extent." );

   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      return MakeVectorL2Restriction< ShapeFunctions >(
         num_elements,
         shift,
         algebraic_dof_extent,
         std::make_index_sequence< ShapeFunctions::vector_dim >{} );
   }
   else
   {
      using DofShape = finite_element_dof_shape_t< ShapeFunctions >;
      return ContiguousL2Restriction< DofShape >{
         shift,
         num_local_dofs,
         algebraic_dof_extent };
   }
}

/** @brief Complete a contiguous L2 specification for a mesh and finite element. */
template < typename Mesh, typename FiniteElement >
auto MakeElementDoFRestriction(
   const Mesh & mesh,
   const FiniteElement & finite_element,
   const ContiguousL2RestrictionSpecification & specification )
{
   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( mesh.GetNumberOfCells() );
   using ShapeFunctions = typename FiniteElement::shape_functions;
   const GlobalIndex num_local_dofs =
      CheckedElementLocalDofCount< ShapeFunctions >( num_elements );
   const GlobalIndex shift = specification.shift.value_or( 0 );
   const GlobalIndex minimum_algebraic_dof_extent = CheckedAdd(
      shift,
      num_local_dofs,
      "Contiguous L2 algebraic extent overflow." );
   const GlobalIndex algebraic_dof_extent =
      specification.algebraic_dof_extent.value_or(
         minimum_algebraic_dof_extent );
   GENDIL_VERIFY(
      algebraic_dof_extent >= minimum_algebraic_dof_extent,
      "Contiguous L2 restriction algebraic extent is smaller than its addressed interval." );
   return MakeContiguousL2Restriction(
      mesh,
      finite_element,
      shift,
      algebraic_dof_extent );
}

/** @brief Validate that the addressed contiguous interval fits its extent. */
template < StaticDofShape DofShape >
void ValidateRestrictionRepresentation(
   const ContiguousL2Restriction< DofShape > & restriction )
{
   GENDIL_VERIFY(
      CheckedAdd(
         restriction.shift,
         restriction.num_local_dofs,
         "Contiguous L2 restriction addressed interval overflow." ) <=
         restriction.algebraic_dof_extent,
      "Contiguous L2 restriction addressed interval exceeds its algebraic extent." );
}

} // namespace gendil

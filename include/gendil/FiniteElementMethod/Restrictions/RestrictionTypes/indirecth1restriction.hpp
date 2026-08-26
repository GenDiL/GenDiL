// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief One-entry indirect H1 restriction specifications and maps.
 */

#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionunitweight.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionvalidation.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/vectorrestriction.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

#include <optional>
#include <tuple>
#include <utility>

namespace gendil {

/**
 * @brief Construction input for a scalar one-entry indirect H1 restriction.
 *
 * The element-to-global index map is borrowed. Supplying @c map_entry_count
 * enables validation against the mesh and finite-element local DoF count.
 */
struct IndirectH1RestrictionSpecification
{
   // Borrowed element-to-L-vector map. The caller retains ownership.
   HostDevicePointer< const int > indices;
   GlobalIndex num_global_dofs;
   std::optional< GlobalIndex > map_entry_count{};

   constexpr IndirectH1RestrictionSpecification(
      const HostDevicePointer< const int > indices_,
      const GlobalIndex num_global_dofs_ )
      : indices( indices_ ),
        num_global_dofs( num_global_dofs_ )
   { }

   constexpr IndirectH1RestrictionSpecification(
      const HostDevicePointer< const int > indices_,
      const GlobalIndex num_global_dofs_,
      const GlobalIndex map_entry_count_ )
      : indices( indices_ ),
        num_global_dofs( num_global_dofs_ ),
        map_entry_count( map_entry_count_ )
   { }
};

/**
 * @brief One-entry indirect restriction for a scalar H1 DoF shape.
 *
 * Each flattened element-local row reads one index from @c indices and adds
 * @c global_offset to obtain its final algebraic coordinate.
 */
template < StaticDofShape DofShape >
struct IndirectH1Restriction
{
   using dof_shape_type = DofShape;

   static constexpr Integer tensor_dim =
      static_cast< Integer >( dof_shape_type::size() );

   HostDevicePointer< const int > indices;
   const GlobalIndex global_offset;
   const GlobalIndex num_local_dofs;
   const GlobalIndex num_global_dofs;
   const GlobalIndex algebraic_dof_extent;
};

/**
 * @brief Construction input for a component-major vector H1 restriction
 * sharing one scalar index map.
 */
template < size_t NumComponents >
struct VectorIndirectH1RestrictionSpecification
{
   static constexpr size_t num_components = NumComponents;

   IndirectH1RestrictionSpecification scalar_specification;
};

/** @brief Visit the single unit-weight entry of one indirect H1 row. */
template < StaticDofShape DofShape, typename Visitor >
GENDIL_HOST_DEVICE
constexpr void ForEachRestrictionEntry(
   const IndirectH1Restriction< DofShape > & restriction,
   const GlobalIndex element,
   const std::array<
      GlobalIndex,
      IndirectH1Restriction< DofShape >::tensor_dim > &
      local_dof,
   Visitor && visitor )
{
   constexpr GlobalIndex local_dofs = Product( DofShape{} );
   GENDIL_ASSERT(
      details::DofIndexIsInBounds< DofShape >( local_dof ),
      "IndirectH1Restriction local tensor index is out of bounds." );
   const GlobalIndex row =
      element * local_dofs + FlattenMultiIndex< DofShape >( local_dof );
   GENDIL_ASSERT(
      row < restriction.num_local_dofs,
      "IndirectH1Restriction element-local row is out of bounds." );
   const int mapped_dof = restriction.indices[row];
   GENDIL_ASSERT(
      mapped_dof >= 0,
      "IndirectH1Restriction contains a negative element-to-L-vector index." );
   GENDIL_ASSERT(
      static_cast< GlobalIndex >( mapped_dof ) <
         restriction.num_global_dofs,
      "IndirectH1Restriction element-to-L-vector index exceeds its logical global DoF count." );
   const GlobalIndex algebraic_dof =
      restriction.global_offset + static_cast< GlobalIndex >( mapped_dof );
   GENDIL_ASSERT(
      algebraic_dof < restriction.algebraic_dof_extent,
      "IndirectH1Restriction global DoF is out of bounds." );
   std::forward< Visitor >( visitor )(
      algebraic_dof,
      RestrictionUnitWeight{} );
}

/** @brief Return the number of element-local restriction rows. */
template < StaticDofShape DofShape >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfLocalDofs(
   const IndirectH1Restriction< DofShape > & restriction )
{
   return restriction.num_local_dofs;
}

/** @brief Return the number of logical global H1 DoFs in this map. */
template < StaticDofShape DofShape >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfGlobalDofs(
   const IndirectH1Restriction< DofShape > & restriction )
{
   return restriction.num_global_dofs;
}

/** @brief Return the size of the containing algebraic coordinate space. */
template < StaticDofShape DofShape >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetAlgebraicDofExtent(
   const IndirectH1Restriction< DofShape > & restriction )
{
   return restriction.algebraic_dof_extent;
}

template < StaticDofShape DofShape >
inline constexpr size_t static_restriction_entry_count_v<
   IndirectH1Restriction< DofShape > > = 1;

template < StaticDofShape DofShape >
inline constexpr bool restriction_may_share_global_dofs_v<
   IndirectH1Restriction< DofShape > > = true;

template < StaticDofShape DofShape >
struct restriction_supports_element_reference_view<
   IndirectH1Restriction< DofShape > > : std::true_type { };

/** @brief Validate that an indirect map is resident in the selected memory. */
template < bool OnDevice, StaticDofShape DofShape >
void ValidateRestrictionMemoryAccess(
   const IndirectH1Restriction< DofShape > & restriction,
   const GlobalIndex active_row_count )
{
   if constexpr ( OnDevice )
   {
#ifdef GENDIL_USE_DEVICE
      GENDIL_VERIFY(
         active_row_count == 0 ||
            restriction.indices.device_pointer != nullptr,
         "Device restriction access requires a device-resident indirect map." );
#else
      static_assert(
         !OnDevice,
         "Device restriction access requires GenDiL device support." );
#endif
   }
   else
   {
      GENDIL_VERIFY(
         active_row_count == 0 ||
            restriction.indices.host_pointer != nullptr,
         "Host restriction access requires a host-resident indirect map." );
   }
}

/** @brief Complete a scalar indirect H1 specification. */
template < typename Mesh, typename FiniteElement >
auto MakeElementDoFRestriction(
   const Mesh & mesh,
   const FiniteElement &,
   const IndirectH1RestrictionSpecification & specification )
{
   using ShapeFunctions = typename FiniteElement::shape_functions;
   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "IndirectH1RestrictionSpecification is scalar; use the vector indirect H1 specification for vector finite elements." );
   using DofShape = finite_element_dof_shape_t< ShapeFunctions >;
   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( mesh.GetNumberOfCells() );
   const GlobalIndex num_local_dofs =
      CheckedElementLocalDofCount< ShapeFunctions >( num_elements );
   GENDIL_VERIFY(
      !specification.map_entry_count.has_value() ||
         *specification.map_entry_count == num_local_dofs,
      "Indirect H1 map length does not match the mesh/finite-element local DoF count." );
   return IndirectH1Restriction< DofShape >{
      specification.indices,
      0,
      num_local_dofs,
      specification.num_global_dofs,
      specification.num_global_dofs };
}

/** @brief Build a vector H1 restriction whose components share one map. */
template <
   size_t NumComponents,
   typename ShapeFunctions,
   size_t... Component >
auto MakeVectorIndirectH1Restriction(
   const VectorIndirectH1RestrictionSpecification< NumComponents > &
      specification,
   const GlobalIndex num_elements,
   std::index_sequence< Component... > )
{
   const auto & scalar = specification.scalar_specification;
   const GlobalIndex vector_global_dofs = CheckedMultiply(
      static_cast< GlobalIndex >( NumComponents ),
      scalar.num_global_dofs,
      "Vector indirect H1 global extent overflow." );
   const GlobalIndex component_local_dofs = CheckedMultiply(
      num_elements,
      static_cast< GlobalIndex >(
         Product( component_dof_shape_t< ShapeFunctions, 0 >{} ) ),
      "Vector indirect H1 component local DoF count overflow." );
   const GlobalIndex num_local_dofs = CheckedMultiply(
      static_cast< GlobalIndex >( NumComponents ),
      component_local_dofs,
      "Vector indirect H1 local DoF count overflow." );

   using Restriction = VectorRestriction<
      IndirectH1Restriction<
         component_dof_shape_t< ShapeFunctions, Component > >... >;
   return Restriction{
      std::tuple{
         IndirectH1Restriction<
            component_dof_shape_t< ShapeFunctions, Component > >{
               scalar.indices,
               CheckedMultiply(
                  static_cast< GlobalIndex >( Component ),
                  scalar.num_global_dofs,
                  "Vector indirect H1 component offset overflow." ),
               component_local_dofs,
               scalar.num_global_dofs,
               vector_global_dofs }... },
      num_local_dofs,
      vector_global_dofs,
      vector_global_dofs };
}

/** @brief Complete a vector indirect H1 specification. */
template < size_t NumComponents, typename Mesh, typename FiniteElement >
auto MakeElementDoFRestriction(
   const Mesh & mesh,
   const FiniteElement &,
   const VectorIndirectH1RestrictionSpecification< NumComponents > &
      specification )
{
   using ShapeFunctions = typename FiniteElement::shape_functions;
   static_assert(
      is_vector_shape_functions_v< ShapeFunctions >,
      "Vector indirect H1 restriction requires vector shape functions." );
   static_assert(
      NumComponents == ShapeFunctions::vector_dim,
      "Vector indirect H1 component count must match the finite element." );
   static_assert(
      VectorComponentDofShapesMatchFirst< ShapeFunctions >(),
      "The shared-map vector indirect H1 representation requires identical component DoF shapes." );
   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( mesh.GetNumberOfCells() );
   const GlobalIndex component_local_dofs = CheckedMultiply(
      num_elements,
      static_cast< GlobalIndex >(
         Product( component_dof_shape_t< ShapeFunctions, 0 >{} ) ),
      "Vector indirect H1 component local DoF count overflow." );
   GENDIL_VERIFY(
      !specification.scalar_specification.map_entry_count.has_value() ||
         *specification.scalar_specification.map_entry_count ==
            component_local_dofs,
      "Vector indirect H1 map length does not match one component's local DoF count." );
   return MakeVectorIndirectH1Restriction<
      NumComponents,
      ShapeFunctions >(
         specification,
         num_elements,
         std::make_index_sequence< NumComponents >{} );
}

/** @brief Validate the mapped interval and borrowed-map residency. */
template < StaticDofShape DofShape >
void ValidateRestrictionRepresentation(
   const IndirectH1Restriction< DofShape > & restriction )
{
   GENDIL_VERIFY(
      CheckedAdd(
         restriction.global_offset,
         restriction.num_global_dofs,
         "Indirect H1 addressed interval overflow." ) <=
         restriction.algebraic_dof_extent,
      "Indirect H1 addressed interval exceeds its algebraic extent." );
   bool has_resident_map =
      restriction.indices.host_pointer != nullptr;
#ifdef GENDIL_USE_DEVICE
   has_resident_map = has_resident_map ||
      restriction.indices.device_pointer != nullptr;
#endif
   GENDIL_VERIFY(
      restriction.num_local_dofs == 0 || has_resident_map,
      "A nonempty indirect H1 restriction requires a resident mapping." );
}

/** @brief Wrap a scalar H1 specification for a fixed component count. */
template < size_t NumComponents >
constexpr auto MakeVectorIndirectH1RestrictionSpecification(
   const IndirectH1RestrictionSpecification & scalar_specification )
{
   return VectorIndirectH1RestrictionSpecification< NumComponents >{
      scalar_specification };
}

} // namespace gendil

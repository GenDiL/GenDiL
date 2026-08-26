// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/contiguousl2restriction.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionvalidation.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/vectorshapefunctions.hpp"
#include "gendil/Meshes/mesh.hpp"
#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <type_traits>

namespace gendil {

// TODO: typename... FiniteElements?
/**
 * @brief A simple class representing a finite element space.
 * 
 * @tparam Mesh The type of mesh used by the finite element space.
 * @tparam FiniteElement The type of finite element used by the finite element space.
 */
template < typename Mesh, typename FiniteElement, typename Restriction >
class FiniteElementSpace : public Mesh
{
   static_assert(
      ElementDoFRestriction< Restriction >,
      "FiniteElementSpace stores a completed ElementDoFRestriction, not a restriction specification." );
   static_assert(
      CompatibleElementDoFRestrictionFor<
         Restriction,
         FiniteElement >,
      "FiniteElementSpace restriction local DoF shapes are incompatible with the finite element." );
   static_assert(
      std::is_same_v<
         mesh::mesh_geometry_t<Mesh>,
         typename FiniteElement::geometry>,
      "FiniteElementSpace: finite-element reference geometry must exactly "
      "match Mesh::cell_type::geometry.");

public:
   using mesh_type = Mesh;
   using finite_element_type = FiniteElement;
   using restriction_type = Restriction;

   const FiniteElement finite_element;
   const Restriction restriction;

   GENDIL_HOST_DEVICE
   FiniteElementSpace( const Mesh & mesh,
                       const FiniteElement & finite_element,
                       const Restriction & restriction ) :
      Mesh( mesh ),
      finite_element( finite_element ),
      restriction( restriction )
   { }

   GENDIL_HOST_DEVICE
   Integer GetNumberOfFiniteElements() const
   {
      return this->GetNumberOfCells();
   }

   /** @brief Return the restriction's logical global DoF count.
    *
    * For an ordinary zero-based space this is also its algebraic vector
    * extent.  Use `GetAlgebraicDofExtent` when placement offsets are
    * semantically relevant.
    */
   GENDIL_HOST_DEVICE
   Integer GetNumberOfFiniteElementDofs() const
   {
      return GetNumberOfGlobalDofs( restriction );
   }
};

/**
 * @brief Shape-function type associated with a homogeneous finite-element space.
 *
 * Top-level cv-qualifiers and references on `FESpace` are ignored.
 *
 * @tparam FESpace A homogeneous finite-element-space type.
 */
template < typename FESpace >
using finite_element_space_shape_functions_t =
   typename std::remove_cvref_t<
      FESpace >::finite_element_type::shape_functions;

/**
 * @brief Whether a homogeneous finite-element space has vector shape functions.
 *
 * @tparam FESpace A homogeneous finite-element-space type.
 */
template < typename FESpace >
inline constexpr bool is_vector_finite_element_space_v =
   is_vector_shape_functions_v<
      finite_element_space_shape_functions_t< FESpace > >;

/**
 * @brief Return the element-DoF restriction stored by a space.
 *
 * @param space The homogeneous finite-element space.
 * @return A const reference to the stored restriction.
 */
template < typename Mesh, typename FiniteElement, typename Restriction >
GENDIL_HOST_DEVICE
constexpr const Restriction & GetRestriction(
   const FiniteElementSpace< Mesh, FiniteElement, Restriction > & space )
{
   return space.restriction;
}

/**
 * @brief Return the number of element-local restriction rows represented by a space.
 *
 * @param space The homogeneous finite-element space.
 */
template < typename Mesh, typename FiniteElement, typename Restriction >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfLocalDofs(
   const FiniteElementSpace< Mesh, FiniteElement, Restriction > & space )
{
   return GetNumberOfLocalDofs( GetRestriction( space ) );
}

/**
 * @brief Return the logical number of global DoFs represented by a space.
 *
 * This count need not equal the algebraic storage extent when the restriction
 * is shifted or embedded in a larger coordinate space.
 *
 * @param space The homogeneous finite-element space.
 */
template < typename Mesh, typename FiniteElement, typename Restriction >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfGlobalDofs(
   const FiniteElementSpace< Mesh, FiniteElement, Restriction > & space )
{
   return GetNumberOfGlobalDofs( GetRestriction( space ) );
}

/**
 * @brief Return the algebraic storage-coordinate extent required by a space.
 *
 * @param space The homogeneous finite-element space.
 */
template < typename Mesh, typename FiniteElement, typename Restriction >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetAlgebraicDofExtent(
   const FiniteElementSpace< Mesh, FiniteElement, Restriction > & space )
{
   return GetAlgebraicDofExtent( GetRestriction( space ) );
}

/**
 * @brief Factory to construct finite element spaces. Useful to hide explicit type.
 * 
 * @tparam Mesh The type of the mesh used by the finite element space.
 * @tparam FiniteElement The type of finite element used by the finite element space.
 * @tparam Restriction The completed element-DoF restriction type.
 * @param mesh The mesh used by the finite element space.
 * @param finite_element The reference finite element used by the finite element space.
 * @param restriction The completed restriction mapping global DoFs to element-local DoFs.
 * @return auto The resulting finite element space.
 */
template < typename Mesh, typename FiniteElement, typename Restriction >
   requires CompatibleElementDoFRestrictionFor<
      Restriction,
      FiniteElement >
auto MakeFiniteElementSpace(
   const Mesh & mesh,
   const FiniteElement & finite_element,
   const Restriction & restriction )
{
   ValidateElementDoFRestrictionFor(
      mesh,
      finite_element,
      restriction );
   return FiniteElementSpace< Mesh, FiniteElement, Restriction >( mesh, finite_element, restriction );
}

/**
 * @brief Construct a finite-element space from a contextual restriction specification.
 *
 * The specification is completed through an ADL `MakeElementDoFRestriction`
 * customization, then validated and stored as a completed restriction.
 *
 * @param mesh The mesh used by the finite-element space.
 * @param finite_element The reference finite element used by the space.
 * @param specification Construction data for the completed restriction.
 * @return A finite-element space storing the completed restriction type.
 */
template < typename Mesh, typename FiniteElement, typename Specification >
   requires RestrictionSpecificationFor<
      Specification,
      Mesh,
      FiniteElement >
auto MakeFiniteElementSpace(
   const Mesh & mesh,
   const FiniteElement & finite_element,
   const Specification & specification )
{
   auto restriction = MakeElementDoFRestriction(
      mesh,
      finite_element,
      specification );
   return MakeFiniteElementSpace(
      mesh,
      finite_element,
      restriction );
}

/**
 * @brief Diagnose an unsupported third argument to `MakeFiniteElementSpace`.
 *
 * This overload participates only when the argument is neither a compatible
 * completed restriction nor a contextual restriction specification.
 */
template < typename Mesh, typename FiniteElement, typename ThirdArgument >
   requires (
      !CompatibleElementDoFRestrictionFor<
         ThirdArgument,
         FiniteElement > &&
      !RestrictionSpecificationFor<
         ThirdArgument,
         Mesh,
         FiniteElement > )
auto MakeFiniteElementSpace(
   const Mesh &,
   const FiniteElement &,
   const ThirdArgument & )
{
   static_assert(
      dependent_false_v< ThirdArgument >,
      "MakeFiniteElementSpace third argument must be either a completed restriction compatible with the mesh/finite element or a contextual restriction specification with an ADL MakeElementDoFRestriction customization." );
}

/**
 * @brief Factory to construct DG finite element spaces. Useful to hide explicit type.
 * 
 * @tparam Mesh The type of the mesh used by the finite element space.
 * @tparam FiniteElement The type of finite element used by the finite element space.
 * @param mesh The mesh used by the finite element space.
 * @param finite_element The reference finite element used by the finite element space.
 * @return auto The resulting finite element space.
 */
template < typename Mesh, typename FiniteElement >
auto MakeFiniteElementSpace( const Mesh & mesh, const FiniteElement & finite_element )
{
   return MakeFiniteElementSpace(
      mesh,
      finite_element,
      ContiguousL2RestrictionSpecification{} );
}

/**
 * @brief Utility struct to access the dimension of a finite element space.
 * 
 * @tparam FESpace 
 */
template <typename FESpace>
struct get_dim;

template <typename Mesh, typename FiniteElement, typename Restriction >
struct get_dim< FiniteElementSpace< Mesh, FiniteElement, Restriction > >
{
   static constexpr Integer value = FiniteElement::space_dim;
};

template < typename FESpace >
inline constexpr Integer get_dim_v = get_dim< FESpace >::value;

template < typename FESpace >
constexpr Integer GetDim( FESpace const & fe_space )
{
   return get_dim_v< FESpace >;
}

/** @brief Number of shape-function components in a homogeneous finite-element space. */
template<class SpaceView>
inline constexpr size_t num_comp_v =
   finite_element_space_shape_functions_t< SpaceView >::num_comp;

}

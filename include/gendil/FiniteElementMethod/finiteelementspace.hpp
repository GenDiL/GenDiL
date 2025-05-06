// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/restriction.hpp"

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

   GENDIL_HOST_DEVICE
   Integer GetNumberOfFiniteElementDofs() const
   {
      if constexpr ( std::is_same_v< restriction_type, L2Restriction > )
      {
         return this->GetNumberOfCells() * FiniteElement::GetNumDofs();
      }
      else // H1Restriction
      {
         return restriction.num_dofs;
      }
   }
};

/**
 * @brief Factory to construct finite element spaces. Useful to hide explicit type.
 * 
 * @tparam Mesh The type of the mesh used by the finite element space.
 * @tparam FiniteElement The type of finite element used by the finite element space.
 * @tparam Restriction The type of the finite element restriction.
 * @param mesh The mesh used by the finite element space.
 * @param finite_element The reference finite element used by the finite element space.
 * @param restriction The restriction mapping from global degrees-of-freedom to element local degrees-of-freedom.
 * @return auto The resulting finite element space.
 */
template < typename Mesh, typename FiniteElement, typename Restriction >
auto MakeFiniteElementSpace( const Mesh & mesh, const FiniteElement & finite_element, const Restriction & restriction )
{
   return FiniteElementSpace< Mesh, FiniteElement, Restriction >( mesh, finite_element, restriction );
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
   return MakeFiniteElementSpace( mesh, finite_element, L2Restriction{} );
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

}

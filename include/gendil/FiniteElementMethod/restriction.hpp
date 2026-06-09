// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

#include <type_traits>

namespace gendil {

struct H1Restriction
{
   // Non-owning scalar element-to-true-DoF map. The caller owns the pointed-to
   // storage and must keep it alive for the finite element space lifetime.
   HostDevicePointer< const int > indices;
   const Integer num_dofs;
};

template < size_t NComp >
struct VectorH1Restriction
{
   static constexpr size_t num_comp = NComp;

   // Reuses the scalar H1 element-to-true-DoF map for each component. Vector
   // true DoFs are numbered component-major:
   // global = component * scalar_num_dofs + scalar_true_dof.
   // The indices pointer is borrowed with the same lifetime contract as
   // H1Restriction::indices.
   HostDevicePointer< const int > indices;
   const Integer scalar_num_dofs;
};

struct L2Restriction
{
   const GlobalIndex shift{0};
};

template < typename Restriction >
struct is_vector_h1_restriction : std::false_type {};

template < size_t NComp >
struct is_vector_h1_restriction< VectorH1Restriction< NComp > >
   : std::true_type {};

template < typename Restriction >
inline constexpr bool is_vector_h1_restriction_v =
   is_vector_h1_restriction< std::remove_cvref_t< Restriction > >::value;

template < typename Restriction >
inline constexpr bool is_h1_restriction_v =
   std::is_same_v< std::remove_cvref_t< Restriction >, H1Restriction > ||
   is_vector_h1_restriction_v< Restriction >;

template < size_t NComp >
VectorH1Restriction< NComp > MakeVectorH1Restriction(
   const H1Restriction & scalar_restriction )
{
   return VectorH1Restriction< NComp >{
      scalar_restriction.indices,
      scalar_restriction.num_dofs
   };
}

}

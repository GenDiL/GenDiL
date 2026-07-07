// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

#include <array>
#include <tuple>
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

template < typename Restriction, typename DofShape >
struct TensorProductRestrictionFactor
{
   using restriction_type = Restriction;
   using dof_shape = DofShape;
};

template < typename ... Factors >
struct TensorProductRestriction
{
   static constexpr size_t num_factors = sizeof...( Factors );

   using factors_type = std::tuple< Factors... >;
   using restrictions_type =
      std::tuple< typename Factors::restriction_type... >;
   using dof_shapes_type = std::tuple< typename Factors::dof_shape... >;

   restrictions_type restrictions;
   std::array< GlobalIndex, num_factors > element_strides;
   std::array< GlobalIndex, num_factors > global_dof_strides;
   GlobalIndex num_dofs;
};

template < typename Restriction >
struct restriction_traits;

template <>
struct restriction_traits< L2Restriction >
{
   static constexpr bool is_direct_index_map = true;
   static constexpr bool is_injective = true;
};

template <>
struct restriction_traits< H1Restriction >
{
   static constexpr bool is_direct_index_map = true;
   static constexpr bool is_injective = false;
};

template < size_t NComp >
struct restriction_traits< VectorH1Restriction< NComp > >
{
   // Component-aware legacy vector H1 behavior. Tensor-product v1 factors are
   // scalar topology restrictions and should not treat this as a scalar factor.
   static constexpr bool is_direct_index_map = true;
   static constexpr bool is_injective = false;
};

template < typename ... Factors >
struct restriction_traits< TensorProductRestriction< Factors... > >
{
   static constexpr bool is_direct_index_map =
      ( restriction_traits<
           typename Factors::restriction_type >::is_direct_index_map && ... );
   static constexpr bool is_injective =
      ( restriction_traits<
           typename Factors::restriction_type >::is_injective && ... );
};

template < typename Restriction >
inline constexpr bool restriction_is_direct_index_map_v =
   restriction_traits< std::remove_cvref_t< Restriction > >::is_direct_index_map;

template < typename Restriction >
inline constexpr bool restriction_is_injective_v =
   restriction_traits< std::remove_cvref_t< Restriction > >::is_injective;

template < typename Restriction >
struct restriction_supports_contiguous_element_view : std::false_type {};

template <>
struct restriction_supports_contiguous_element_view< L2Restriction >
   : std::true_type {};

template < typename Restriction >
inline constexpr bool restriction_supports_contiguous_element_view_v =
   restriction_supports_contiguous_element_view<
      std::remove_cvref_t< Restriction > >::value;

template < typename Restriction >
struct is_vector_h1_restriction : std::false_type {};

template < size_t NComp >
struct is_vector_h1_restriction< VectorH1Restriction< NComp > >
   : std::true_type {};

template < typename Restriction >
inline constexpr bool is_vector_h1_restriction_v =
   is_vector_h1_restriction< std::remove_cvref_t< Restriction > >::value;

template < typename Restriction >
struct is_tensor_product_restriction : std::false_type {};

template < typename ... Factors >
struct is_tensor_product_restriction<
   TensorProductRestriction< Factors... > >
   : std::true_type {};

template < typename Restriction >
inline constexpr bool is_tensor_product_restriction_v =
   is_tensor_product_restriction< std::remove_cvref_t< Restriction > >::value;

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

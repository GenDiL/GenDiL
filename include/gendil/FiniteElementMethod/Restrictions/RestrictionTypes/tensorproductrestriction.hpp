// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Completed restrictions assembled from scalar tensor-product factors.
 */

#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/FiniteElementMethod/Restrictions/globaldofindex.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionunitweight.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionvalidation.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/vectorrestriction.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/cat.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"
#include "gendil/Utilities/multiindex.hpp"

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

/** @brief Require a unit-entry tensor restriction usable as a product factor. */
template < typename Restriction >
concept TensorProductFactorRestriction =
   TensorElementDoFRestriction< Restriction > &&
   static_restriction_entry_count_v<
      std::remove_cvref_t< Restriction > > == 1 &&
   restriction_supports_element_reference_view_v< Restriction >;

/**
 * @brief One-entry restriction for a Cartesian product of scalar factors.
 *
 * Each factor is a completed tensor restriction whose emitted coordinate is
 * already final in that factor's algebraic space. The product combines those
 * coordinates with first-factor-fastest strides formed from factor algebraic
 * extents. The concatenated factor shapes form the native local DoF
 * coordinate.
 */
template < typename... FactorRestrictions >
   requires (
      sizeof...( FactorRestrictions ) > 0 &&
      ( TensorProductFactorRestriction< FactorRestrictions > && ... ) )
struct TensorProductRestriction
{
   static constexpr size_t num_factors = sizeof...( FactorRestrictions );

   using restrictions_type = std::tuple< FactorRestrictions... >;
   using dof_shape_type = cat_t<
      typename FactorRestrictions::dof_shape_type... >;

   static constexpr Integer tensor_dim =
      static_cast< Integer >( dof_shape_type::size() );

   restrictions_type restrictions;
   std::array< GlobalIndex, num_factors > element_strides;
   std::array< GlobalIndex, num_factors > algebraic_dof_strides;
   GlobalIndex num_local_dofs;
   GlobalIndex num_global_dofs;
   GlobalIndex algebraic_dof_extent;
};

/** @brief Return the total number of tensor-product local rows. */
template < typename... FactorRestrictions >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfLocalDofs(
   const TensorProductRestriction< FactorRestrictions... > & restriction )
{
   return restriction.num_local_dofs;
}

/** @brief Return the number of tensor-product logical global DoFs. */
template < typename... FactorRestrictions >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfGlobalDofs(
   const TensorProductRestriction< FactorRestrictions... > & restriction )
{
   return restriction.num_global_dofs;
}

/** @brief Return the tensor-product algebraic coordinate extent. */
template < typename... FactorRestrictions >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetAlgebraicDofExtent(
   const TensorProductRestriction< FactorRestrictions... > & restriction )
{
   return restriction.algebraic_dof_extent;
}

template < typename... FactorRestrictions >
inline constexpr size_t static_restriction_entry_count_v<
   TensorProductRestriction< FactorRestrictions... > > = 1;

template < typename... FactorRestrictions >
inline constexpr bool restriction_may_share_global_dofs_v<
   TensorProductRestriction< FactorRestrictions... > > =
      ( restriction_may_share_global_dofs_v<
           FactorRestrictions > || ... );

template < typename... FactorRestrictions >
struct restriction_supports_element_reference_view<
   TensorProductRestriction< FactorRestrictions... > > : std::true_type { };

/** @brief Validate memory access recursively for every algebraic factor. */
template < bool OnDevice, typename... FactorRestrictions >
void ValidateRestrictionMemoryAccess(
   const TensorProductRestriction< FactorRestrictions... > & restriction,
   const GlobalIndex active_row_count )
{
   std::apply(
      [&] ( const auto & ... factor )
      {
         auto validate_factor = [&] ( const auto & current_factor )
         {
            using Factor = std::remove_cvref_t<
               decltype( current_factor ) >;
            constexpr GlobalIndex factor_element_dofs =
               Product( typename Factor::dof_shape_type{} );
            static_assert(
               factor_element_dofs > 0,
               "Tensor-product factors require a nonempty DoF shape." );
            const GlobalIndex factor_elements =
               GetNumberOfLocalDofs( current_factor ) /
               factor_element_dofs;
            ValidateRestrictionMemoryAccess< OnDevice >(
               current_factor,
               active_row_count == 0 ? 0 : factor_elements );
         };
         ( validate_factor( factor ), ... );
      },
      restriction.restrictions );
}

template < typename TensorRestriction, size_t I >
using tensor_product_factor_type_t =
   std::tuple_element_t< I, typename TensorRestriction::restrictions_type >;

template < typename TensorRestriction, size_t I >
using tensor_product_factor_dof_shape_t =
   typename tensor_product_factor_type_t<
      TensorRestriction,
      I >::dof_shape_type;

template < typename TensorRestriction, size_t I >
inline constexpr size_t tensor_product_factor_dof_rank_v =
   tensor_product_factor_dof_shape_t< TensorRestriction, I >::size();

template < typename TensorRestriction, size_t I, size_t ... J >
GENDIL_HOST_DEVICE
constexpr size_t TensorProductLocalDofRankOffset( std::index_sequence< J... > )
{
   return ( size_t{0} + ... +
      tensor_product_factor_dof_rank_v< TensorRestriction, J > );
}

template < typename TensorRestriction, size_t I >
inline constexpr size_t tensor_product_local_dof_rank_offset_v =
   TensorProductLocalDofRankOffset< TensorRestriction, I >(
      std::make_index_sequence< I >{} );

/** @brief Form first-coordinate-fastest strides for a product of counts. */
template < size_t N >
std::array< GlobalIndex, N > MakePrefixStrides(
   const std::array< GlobalIndex, N > & counts,
   const char * overflow_message )
{
   std::array< GlobalIndex, N > strides{};
   GlobalIndex stride = 1;
   for ( size_t i = 0; i < N; ++i )
   {
      strides[i] = stride;
      stride = CheckedMultiply(
         stride,
         counts[i],
         overflow_message );
   }
   return strides;
}

/** @brief Decompose a product element ordinal into factor element indices. */
template < typename TensorRestriction >
GENDIL_HOST_DEVICE
void TensorProductElementIndices(
   const TensorRestriction & restriction,
   const GlobalIndex element_index,
   std::array< GlobalIndex, TensorRestriction::num_factors > & element_indices )
{
   static_assert(
      TensorRestriction::num_factors > 0,
      "TensorProductRestriction must have at least one factor." );

   GlobalIndex remaining = element_index;
   for ( size_t i = TensorRestriction::num_factors - 1; i > 0; --i )
   {
      element_indices[i] = remaining / restriction.element_strides[i];
      remaining -= restriction.element_strides[i] * element_indices[i];
   }
   element_indices[0] = remaining;
}

template <
   typename TensorRestriction,
   size_t I,
   size_t... LocalIndex >
GENDIL_HOST_DEVICE
GlobalIndex TensorProductFactorAlgebraicDof(
   const TensorRestriction & restriction,
   const std::array< GlobalIndex, TensorRestriction::num_factors > & element_indices,
   const std::array< GlobalIndex, TensorRestriction::tensor_dim > & local_dof,
   std::index_sequence< LocalIndex... > )
{
   constexpr size_t rank_offset =
      tensor_product_local_dof_rank_offset_v< TensorRestriction, I >;
   const std::array< GlobalIndex, sizeof...( LocalIndex ) > factor_local_dof{
      local_dof[rank_offset + LocalIndex]... };
   return GetGlobalDofIndex(
      std::get< I >( restriction.restrictions ),
      element_indices[I],
      factor_local_dof );
}

template <
   typename TensorRestriction,
   size_t ... I >
GENDIL_HOST_DEVICE
GlobalIndex TensorProductElementToGlobalDofIndex(
   const TensorRestriction & restriction,
   const GlobalIndex element_index,
   const std::array< GlobalIndex, TensorRestriction::tensor_dim > & local_dof,
   std::index_sequence< I... > )
{
   std::array< GlobalIndex, TensorRestriction::num_factors > element_indices{};
   TensorProductElementIndices(
      restriction,
      element_index,
      element_indices );

   GlobalIndex global_index = 0;
   (
      ( global_index +=
           TensorProductFactorAlgebraicDof< TensorRestriction, I >(
              restriction,
              element_indices,
              local_dof,
              std::make_index_sequence<
                 tensor_product_factor_dof_rank_v<
                    TensorRestriction,
                    I > >{} ) * restriction.algebraic_dof_strides[I] ),
      ... );
   return global_index;
}

/** @brief Visit the single unit-weight entry of one tensor-product row. */
template < typename... FactorRestrictions, typename Visitor >
GENDIL_HOST_DEVICE
constexpr void ForEachRestrictionEntry(
   const TensorProductRestriction< FactorRestrictions... > & restriction,
   const GlobalIndex element,
   const std::array<
      GlobalIndex,
      TensorProductRestriction< FactorRestrictions... >::tensor_dim > & local_dof,
   Visitor && visitor )
{
   using Restriction = TensorProductRestriction< FactorRestrictions... >;
   using DofShape = typename Restriction::dof_shape_type;
   GENDIL_ASSERT(
      details::DofIndexIsInBounds< DofShape >( local_dof ),
      "TensorProductRestriction local tensor index is out of bounds." );
   [[maybe_unused]] constexpr GlobalIndex local_dofs = Product( DofShape{} );
   GENDIL_ASSERT(
      element * local_dofs + FlattenMultiIndex< DofShape >( local_dof ) <
         restriction.num_local_dofs,
      "TensorProductRestriction element-local row is out of bounds." );
   const GlobalIndex algebraic_dof =
      TensorProductElementToGlobalDofIndex(
         restriction,
         element,
         local_dof,
         std::make_index_sequence< Restriction::num_factors >{} );
   GENDIL_ASSERT(
      algebraic_dof < restriction.algebraic_dof_extent,
      "TensorProductRestriction global DoF is out of bounds." );
   std::forward< Visitor >( visitor )(
      algebraic_dof,
      RestrictionUnitWeight{} );
}

namespace details {

template < typename FactorSpace >
using tensor_product_space_restriction_t =
   typename std::remove_cvref_t< FactorSpace >::restriction_type;

template < typename FactorSpace >
concept ScalarTensorProductFactorSpace =
   TensorProductFactorRestriction<
      tensor_product_space_restriction_t< FactorSpace > >;

template < typename Restriction, size_t... Component >
consteval bool VectorTensorProductComponentsAreFactors(
   std::index_sequence< Component... > )
{
   return (
      TensorProductFactorRestriction< std::remove_cvref_t< decltype(
         GetComponentRestriction< Component >(
            std::declval< const Restriction & >() ) ) > > && ... );
}

template < typename FactorSpace >
concept VectorTensorProductFactorSpace =
   VectorElementDoFRestriction<
      tensor_product_space_restriction_t< FactorSpace > > &&
   VectorTensorProductComponentsAreFactors<
      tensor_product_space_restriction_t< FactorSpace > >(
         std::make_index_sequence<
            tensor_product_space_restriction_t<
               FactorSpace >::num_components >{} );

template < typename FirstFactorSpace, typename... RemainingFactorSpaces >
consteval size_t TensorProductVectorComponentCount()
{
   if constexpr (
      VectorTensorProductFactorSpace< FirstFactorSpace > )
   {
      return tensor_product_space_restriction_t<
         FirstFactorSpace >::num_components;
   }
   else
   {
      static_assert(
         sizeof...( RemainingFactorSpaces ) > 0,
         "A vector tensor-product factory requires at least one vector factor." );
      return TensorProductVectorComponentCount<
         RemainingFactorSpaces... >();
   }
}

template < size_t NumComponents, typename FactorSpace >
consteval bool TensorProductVectorComponentCountMatches()
{
   if constexpr (
      VectorTensorProductFactorSpace< FactorSpace > )
   {
      return tensor_product_space_restriction_t<
         FactorSpace >::num_components == NumComponents;
   }
   else
   {
      return true;
   }
}

template <
   TensorProductFactorRestriction... FactorRestrictions >
auto MakeScalarTensorProductRestriction(
   const std::array<
      GlobalIndex,
      sizeof...( FactorRestrictions ) > & element_counts,
   const FactorRestrictions & ... factor_restrictions )
{
   using Restriction = TensorProductRestriction<
      FactorRestrictions... >;
   const std::array< GlobalIndex, sizeof...( FactorRestrictions ) >
      global_dof_counts{
         GetNumberOfGlobalDofs( factor_restrictions )... };
   const std::array< GlobalIndex, sizeof...( FactorRestrictions ) >
      algebraic_dof_extents{
         GetAlgebraicDofExtent( factor_restrictions )... };

   GlobalIndex num_local_dofs = 1;
   ( ( num_local_dofs = CheckedMultiply(
          num_local_dofs,
          GetNumberOfLocalDofs( factor_restrictions ),
          "Tensor-product restriction local DoF extent overflow." ) ),
     ... );
   GlobalIndex num_global_dofs = 1;
   for ( const GlobalIndex count : global_dof_counts )
   {
      num_global_dofs = CheckedMultiply(
         num_global_dofs,
         count,
         "Tensor-product restriction global DoF extent overflow." );
   }
   GlobalIndex algebraic_dof_extent = 1;
   for ( const GlobalIndex extent : algebraic_dof_extents )
   {
      algebraic_dof_extent = CheckedMultiply(
         algebraic_dof_extent,
         extent,
         "Tensor-product restriction algebraic DoF extent overflow." );
   }

   return Restriction{
      std::make_tuple( factor_restrictions... ),
      MakePrefixStrides(
         element_counts,
         "Tensor-product restriction element-stride overflow." ),
      MakePrefixStrides(
         algebraic_dof_extents,
         "Tensor-product restriction algebraic-stride overflow." ),
      num_local_dofs,
      num_global_dofs,
      algebraic_dof_extent
   };
}

template < size_t Component, typename FactorSpace >
constexpr decltype(auto) GetTensorProductComponentFactorRestriction(
   const FactorSpace & factor_space )
{
   if constexpr (
      VectorTensorProductFactorSpace< FactorSpace > )
   {
      return GetComponentRestriction< Component >(
         factor_space.restriction );
   }
   else
   {
      return ( factor_space.restriction );
   }
}

template < size_t Component, typename... FactorSpaces >
auto MakeTensorProductComponentRestriction(
   const std::array< GlobalIndex, sizeof...( FactorSpaces ) > & element_counts,
   const FactorSpaces & ... factor_spaces )
{
   return MakeScalarTensorProductRestriction(
      element_counts,
      GetTensorProductComponentFactorRestriction< Component >(
         factor_spaces )... );
}

template < typename... ComponentRestrictions >
auto MakeVectorTensorProductRestriction(
   std::tuple< ComponentRestrictions... > component_restrictions )
{
   static_assert(
      sizeof...( ComponentRestrictions ) > 0,
      "A vector tensor-product restriction requires at least one component." );

   GlobalIndex num_local_dofs = 0;
   GlobalIndex num_global_dofs = 0;
   const GlobalIndex algebraic_dof_extent =
      GetAlgebraicDofExtent( std::get< 0 >( component_restrictions ) );
   std::apply(
      [&] ( const auto & ... component )
      {
         auto accumulate_component = [&] ( const auto & current_component )
         {
            GENDIL_VERIFY(
               GetAlgebraicDofExtent( current_component ) ==
                  algebraic_dof_extent,
               "Every component tensor-product restriction must report the "
               "same final algebraic extent." );
            num_local_dofs = CheckedAdd(
               num_local_dofs,
               GetNumberOfLocalDofs( current_component ),
               "Vector tensor-product restriction local extent overflow." );
            num_global_dofs = CheckedAdd(
               num_global_dofs,
               GetNumberOfGlobalDofs( current_component ),
               "Vector tensor-product restriction global count overflow." );
         };
         ( accumulate_component( component ), ... );
      },
      component_restrictions );

   using Restriction = VectorRestriction< ComponentRestrictions... >;
   Restriction restriction{
      std::move( component_restrictions ),
      num_local_dofs,
      num_global_dofs,
      algebraic_dof_extent };
   ValidateElementDoFRestriction( restriction );
   return restriction;
}

template <
   size_t NumComponents,
   typename... FactorSpaces,
   size_t... Component >
auto MakeVectorTensorProductRestriction(
   const std::array< GlobalIndex, sizeof...( FactorSpaces ) > & element_counts,
   std::index_sequence< Component... >,
   const FactorSpaces & ... factor_spaces )
{
   static_assert(
      sizeof...( Component ) == NumComponents,
      "Vector tensor-product component sequence has the wrong size." );
   return MakeVectorTensorProductRestriction(
      std::tuple{
         MakeTensorProductComponentRestriction< Component >(
            element_counts,
            factor_spaces... )... } );
}

} // namespace details

/** @brief Build a completed tensor-product restriction from scalar spaces. */
template < typename ... FactorSpaces >
   requires (
      details::ScalarTensorProductFactorSpace< FactorSpaces > && ... )
auto MakeTensorProductRestriction( const FactorSpaces & ... factor_spaces )
{
   const std::array< GlobalIndex, sizeof...( FactorSpaces ) > element_counts{
      static_cast< GlobalIndex >(
         factor_spaces.GetNumberOfFiniteElements() )... };
   return details::MakeScalarTensorProductRestriction(
      element_counts,
      factor_spaces.restriction... );
}

/**
 * @brief Build a componentwise tensor product from scalar and vector spaces.
 *
 * Scalar factors are broadcast to every output component. Vector factors are
 * zipped by component and must have the same compile-time component count.
 * Selected component restrictions retain their final factor-wide algebraic
 * coordinates. Each scalar component product uses the product of the factor
 * algebraic extents, so all output components share one final extent even
 * when their occupied coordinate subsets are sparse or disjoint. The result
 * is not implicitly rebased or compacted and may therefore contain unused
 * algebraic coordinates.
 */
template < typename ... FactorSpaces >
   requires (
      ( ( details::ScalarTensorProductFactorSpace< FactorSpaces > ||
          details::VectorTensorProductFactorSpace< FactorSpaces > ) && ... ) &&
      ( details::VectorTensorProductFactorSpace< FactorSpaces > || ... ) )
auto MakeTensorProductRestriction( const FactorSpaces & ... factor_spaces )
{
   constexpr size_t num_components =
      details::TensorProductVectorComponentCount< FactorSpaces... >();
   constexpr bool matching_component_counts =
      ( details::TensorProductVectorComponentCountMatches<
           num_components,
           FactorSpaces >() && ... );
   if constexpr ( !matching_component_counts )
   {
      static_assert(
         matching_component_counts,
         "Every vector tensor-product factor must have the same number of "
         "components." );
   }
   else
   {
      const std::array< GlobalIndex, sizeof...( FactorSpaces ) > element_counts{
         static_cast< GlobalIndex >(
            factor_spaces.GetNumberOfFiniteElements() )... };
      return details::MakeVectorTensorProductRestriction< num_components >(
         element_counts,
         std::make_index_sequence< num_components >{},
         factor_spaces... );
   }
}

/** @brief Validate factor maps, algebraic strides, and product dimensions. */
template < typename... FactorRestrictions >
void ValidateRestrictionRepresentation(
   const TensorProductRestriction< FactorRestrictions... > & restriction )
{
   GlobalIndex expected_element_stride = 1;
   GlobalIndex expected_algebraic_stride = 1;
   GlobalIndex expected_num_local_dofs = 1;
   GlobalIndex expected_num_global_dofs = 1;
   GlobalIndex expected_algebraic_dof_extent = 1;
   size_t factor_index = 0;

   std::apply(
      [&] ( const auto & ... factor )
      {
         auto validate_factor = [&] ( const auto & current_factor )
         {
            using Factor = std::remove_cvref_t<
               decltype( current_factor ) >;
            constexpr GlobalIndex factor_element_dofs =
               Product( typename Factor::dof_shape_type{} );
            ValidateElementDoFRestriction( current_factor );
            GENDIL_VERIFY(
               factor_element_dofs > 0 &&
                  GetNumberOfLocalDofs( current_factor ) %
                     factor_element_dofs == 0,
               "Tensor-product factor local count is inconsistent with its DoF shape." );
            GENDIL_VERIFY(
               restriction.element_strides[factor_index] ==
                  expected_element_stride,
               "Tensor-product restriction element strides are inconsistent with its factors." );
            GENDIL_VERIFY(
               restriction.algebraic_dof_strides[factor_index] ==
                  expected_algebraic_stride,
               "Tensor-product restriction algebraic strides are inconsistent with its factors." );

            const GlobalIndex factor_elements =
               GetNumberOfLocalDofs( current_factor ) /
               factor_element_dofs;
            expected_element_stride = CheckedMultiply(
               expected_element_stride,
               factor_elements,
               "Tensor-product restriction element-stride overflow." );
            expected_algebraic_stride = CheckedMultiply(
               expected_algebraic_stride,
               GetAlgebraicDofExtent( current_factor ),
               "Tensor-product restriction algebraic-stride overflow." );
            expected_num_local_dofs = CheckedMultiply(
               expected_num_local_dofs,
               GetNumberOfLocalDofs( current_factor ),
               "Tensor-product restriction local DoF extent overflow." );
            expected_num_global_dofs = CheckedMultiply(
               expected_num_global_dofs,
               GetNumberOfGlobalDofs( current_factor ),
               "Tensor-product restriction global DoF extent overflow." );
            expected_algebraic_dof_extent = CheckedMultiply(
               expected_algebraic_dof_extent,
               GetAlgebraicDofExtent( current_factor ),
               "Tensor-product restriction algebraic DoF extent overflow." );
            ++factor_index;
         };
         ( validate_factor( factor ), ... );
      },
      restriction.restrictions );

   GENDIL_VERIFY(
      restriction.num_local_dofs == expected_num_local_dofs,
      "Tensor-product restriction local DoF count disagrees with its factors." );
   GENDIL_VERIFY(
      restriction.num_global_dofs == expected_num_global_dofs,
      "Tensor-product restriction global DoF count disagrees with its factors." );
   GENDIL_VERIFY(
      restriction.algebraic_dof_extent ==
         expected_algebraic_dof_extent,
      "Tensor-product restriction algebraic extent disagrees with its factors." );
}

} // namespace gendil

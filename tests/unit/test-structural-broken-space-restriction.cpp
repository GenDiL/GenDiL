// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>

namespace structural_restriction_test {

struct PermutedRestriction
{
   using dof_shape_type = std::index_sequence< 2 >;
   static constexpr gendil::Integer tensor_dim = 1;
};

inline gendil::GlobalIndex GetNumberOfLocalDofs(
   const PermutedRestriction & )
{
   return 4;
}

inline gendil::GlobalIndex GetNumberOfGlobalDofs(
   const PermutedRestriction & )
{
   return 4;
}

inline gendil::GlobalIndex GetAlgebraicDofExtent(
   const PermutedRestriction & )
{
   return 4;
}

template < typename Visitor >
GENDIL_HOST_DEVICE
void ForEachRestrictionEntry(
   const PermutedRestriction &,
   const gendil::GlobalIndex element,
   const std::array< gendil::GlobalIndex, 1 > & local_dof,
   Visitor && visitor )
{
   std::forward< Visitor >( visitor )(
      element * 2 + ( 1 - local_dof[0] ),
      gendil::RestrictionUnitWeight{} );
}

} // namespace structural_restriction_test

namespace gendil {

template <>
inline constexpr size_t static_restriction_entry_count_v<
   structural_restriction_test::PermutedRestriction > = 1;

template <>
inline constexpr bool restriction_may_share_global_dofs_v<
   structural_restriction_test::PermutedRestriction > = false;

template <>
struct restriction_supports_element_reference_view<
   structural_restriction_test::PermutedRestriction > : std::true_type { };

} // namespace gendil

int main()
{
   using namespace gendil;
   using structural_restriction_test::PermutedRestriction;
   static_assert(
      ElementwiseIndependentRestriction< PermutedRestriction > );

   Cartesian1DMesh mesh( 0.5, 2 );
   const auto finite_element =
      MakeLegendreFiniteElement( FiniteElementOrders< 1 >{} );
   const auto direct_space =
      MakeFiniteElementSpace( mesh, finite_element );
   const auto permuted_space =
      MakeFiniteElementSpace(
         mesh,
         finite_element,
         PermutedRestriction{} );
   const auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 3 >{} );
   auto coefficient =
      [] GENDIL_HOST_DEVICE ( const std::array< Real, 1 > & )
      {
         return Real( 1 );
      };
   const auto direct_operator =
      MakeMassFiniteElementOperator< SerialKernelConfiguration >(
         direct_space,
         integration_rule,
         coefficient );
   const auto permuted_operator =
      MakeMassFiniteElementOperator< SerialKernelConfiguration >(
         permuted_space,
         integration_rule,
         coefficient );

   Vector direct_input( 4 );
   Vector permuted_input( 4 );
   Real * direct_input_data = direct_input.WriteHostData();
   Real * permuted_input_data = permuted_input.WriteHostData();
   for ( GlobalIndex element = 0; element < 2; ++element )
   {
      for ( GlobalIndex local = 0; local < 2; ++local )
      {
         const Real value =
            Real( 1 ) + Real( 2 * element + local );
         direct_input_data[element * 2 + local] = value;
         permuted_input_data[element * 2 + ( 1 - local )] = value;
      }
   }

   Vector direct_output( 4 );
   Vector permuted_output( 4 );
   direct_operator( direct_input, direct_output );
   permuted_operator( permuted_input, permuted_output );
   const Real * direct_output_data = direct_output.ReadHostData();
   const Real * permuted_output_data = permuted_output.ReadHostData();

   constexpr Real tolerance = 1.0e-12;
   for ( GlobalIndex element = 0; element < 2; ++element )
   {
      for ( GlobalIndex local = 0; local < 2; ++local )
      {
         if ( std::abs(
                 direct_output_data[element * 2 + local] -
                 permuted_output_data[element * 2 + ( 1 - local )] ) >
              tolerance )
         {
            std::cout
               << "Structural broken-space restriction changed the mass "
                  "operator action.\n";
            return 1;
         }
      }
   }
   return 0;
}

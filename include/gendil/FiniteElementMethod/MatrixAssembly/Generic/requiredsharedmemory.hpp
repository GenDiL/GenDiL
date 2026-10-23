// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Compute shared scratch requirements for sparse-assembly probes.
 *
 * The calculation uses the same local field bindings as numerical assembly
 * and reuses the matrix-free per-integrand resource requirements.
 */

#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/localdofoperations.hpp"
#include "gendil/Utilities/KernelContext/kernelcontext.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil
{
namespace details
{

// Probe rows and bindings are shared by a phase; numerical storage is scoped
// to each original integrand, so scratch is the maximum of its requirements.
// The result counts Real entries, matching the existing shared-memory traits.
template < class Policy, class Integrands, class Context, class OperatorContext >
consteval size_t SparseAssemblyRequiredSharedMemory()
{
   if constexpr ( std::tuple_size_v< Integrands > == 0 )
   {
      return size_t{ 0 };
   }
   else
   {
      using TermTuple = Integrands;
      using First = std::remove_cvref_t< std::tuple_element_t< 0, TermTuple > >;
      using Signature = typename OperatorContext::template signature_for< First >;
      constexpr auto TrialName = weak_form_traits::only_trial_field_name_v< Integrands >;
      constexpr auto TestName = weak_form_traits::only_test_field_name_v< Integrands >;
      using InputSpace = std::remove_cvref_t< decltype( GetMinusFacetFieldVolumeSpace(
         std::declval< const Context & >().template GetSpace< TrialName >() ) ) >;
      using OutputSpace = std::remove_cvref_t< decltype( GetMinusFacetFieldVolumeSpace(
         std::declval< const Context & >().template GetSpace< TestName >() ) ) >;
      using Kernel = KernelContext< Policy, 0 >;
      using InputRow = decltype( MakeZeroElementVector(
         std::declval< const Kernel & >(), std::declval< const InputSpace & >() ) );
      using OutputRow = decltype( MakeZeroElementVector(
         std::declval< const Kernel & >(), std::declval< const OutputSpace & >() ) );
      if constexpr ( Signature::entity == SemanticEntity::InteriorFacet )
      {
         using OutputBinding = std::remove_cvref_t< decltype(
            std::declval< const Context & >().template GetSpace< TestName >() ) >;
         if constexpr ( is_interior_face_field_binding_v< OutputBinding > )
         {
            using PlusInputSpace = std::remove_cvref_t< decltype( GetPlusFacetFieldVolumeSpace(
               std::declval< const Context & >().template GetSpace< TrialName >() ) ) >;
            using PlusOutputSpace = std::remove_cvref_t< decltype( GetPlusFacetFieldVolumeSpace(
               std::declval< const Context & >().template GetSpace< TestName >() ) ) >;
            using PlusInputRow = decltype( MakeZeroElementVector(
               std::declval< const Kernel & >(), std::declval< const PlusInputSpace & >() ) );
            using PlusOutputRow = decltype( MakeZeroElementVector(
               std::declval< const Kernel & >(), std::declval< const PlusOutputSpace & >() ) );
            using Minus = decltype( BindElementDofs< SemanticSide::Minus >(
               std::declval< const Integrands & >(), std::declval< const Context & >(),
               std::declval< const OperatorContext & >(),
               ReadLocal< TrialName >( std::declval< const InputRow & >() ),
               AddLocal< TestName >( std::declval< OutputRow & >() ) ) );
            using Plus = decltype( BindElementDofs< SemanticSide::Plus >(
               std::declval< const Integrands & >(), std::declval< const Context & >(),
               std::declval< const OperatorContext & >(),
               ReadLocal< TrialName >( std::declval< const PlusInputRow & >() ),
               AddLocal< TestName >( std::declval< PlusOutputRow & >() ) ) );
            return detail::TermSharedSlots< Policy, Minus, OperatorContext, Plus >(
               static_cast< TermTuple * >( nullptr ) );
         }
         else
         {
            using Minus = decltype( BindElementDofs< SemanticSide::Minus >(
               std::declval< const Integrands & >(), std::declval< const Context & >(),
               std::declval< const OperatorContext & >(),
               ReadLocal< TrialName >( std::declval< const InputRow & >() ),
               AddLocal< TestName >( std::declval< OutputRow & >() ) ) );
            using Plus = decltype( BindElementDofs< SemanticSide::Plus >(
               std::declval< const Integrands & >(), std::declval< const Context & >(),
               std::declval< const OperatorContext & >(),
               ReadLocal< TrialName >( std::declval< const InputRow & >() ) ) );
            return detail::TermSharedSlots< Policy, Minus, OperatorContext, Plus >(
               static_cast< TermTuple * >( nullptr ) );
         }
      }
      else
      {
         using Local = decltype( BindElementDofs(
            std::declval< const Integrands & >(), std::declval< const Context & >(),
            std::declval< const OperatorContext & >(),
            ReadLocal< TrialName >( std::declval< const InputRow & >() ),
            AddLocal< TestName >( std::declval< OutputRow & >() ) ) );
         return detail::TermSharedSlots< Policy, Local, OperatorContext >(
            static_cast< TermTuple * >( nullptr ) );
      }
   }
}

} // namespace details
} // namespace gendil

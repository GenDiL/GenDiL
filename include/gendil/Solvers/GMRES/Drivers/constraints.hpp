// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/**
 * @file
 * @brief Decide which GMRES overloads admit the supplied argument types.
 *
 * These partial checks reject unsupported vector categories and unavailable
 * system applications. They do not certify callback bodies, metric consistency,
 * preconditioner linearity, capacities, or collective completion.
 */

#include "gendil/Algebra/Vector/localoperations.hpp"
#include "gendil/Algebra/vectormapconcept.hpp"
#include "gendil/Solvers/GMRES/Workspace/workspace.hpp"
#include "gendil/Solvers/GMRES/arnoldi.hpp"
#include "gendil/Solvers/GMRES/preconditioning.hpp"
#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"
#include <concepts>
#include <type_traits>
#include <utility>

namespace gendil::gmres_detail
{

// Use the owner's borrowed operands when checking the operations a solve needs.
using OwningWorkspaceView =
    decltype(MakeGMRESWorkspaceView(std::declval<GMRESWorkspace &>()));
template <class Workspace>
using SolveBufferType = std::conditional_t<std::is_same_v<Workspace, GMRESWorkspace>,
                                           OwningWorkspaceView, Workspace>;
template <class B>
using WorkspaceWorkVector = std::remove_reference_t<decltype(std::declval<B &>().work)>;
template <class B>
using WorkspaceBasisVector =
    std::remove_reference_t<decltype(std::declval<B &>().basis(0))>;
template <class B>
using WorkspaceDirection =
    std::remove_reference_t<decltype(std::declval<B &>().directions(0))>;

/** Admit host solves whose selected vector access and system applications exist.
 * Check the operands actually used by the selected variant: right-preconditioned
 * expansion applies A to a direction, not to a basis vector. This is a partial
 * type check, not a check of callback bodies, runtime storage, or solver validity.
 */
template <class Policy, PreconditioningMode mode, class Map, class RHS, class Solution,
          class Workspace>
inline constexpr bool SupportsHostSolve = []
{
   using B = SolveBufferType<Workspace>;
   if constexpr (!requires(B &b) {
                    b.work;
                    b.basis(0);
                 })
   {
      return false;
   }
   else
   {
      using Work = WorkspaceWorkVector<B>;
      constexpr bool device = is_device_configuration_v<Policy>;
      if constexpr (!VectorMap<Map, Solution, Work> ||
                    !KernelReadableVector<device, RHS> ||
                    !KernelAccessibleVector<device, Solution> ||
                    !KernelAccessibleVector<device, Work>)
      {
         return false;
      }
      else if constexpr (mode == PreconditioningMode::None)
      {
         return VectorMap<Map, WorkspaceBasisVector<B>, Work>;
      }
      else if constexpr (requires(B &b) { b.directions(0); })
      {
         return VectorMap<Map, WorkspaceDirection<B>, Work>;
      }
      else
      {
         return false;
      }
   }
}();

/// Recognize the two orthogonalization methods implemented by these drivers.
template <class Orthogonalization>
inline constexpr bool IsGMRESOrthogonalization =
    std::is_same_v<Orthogonalization, ModifiedGramSchmidt> ||
    std::is_same_v<Orthogonalization, ClassicalGramSchmidtTwoPass>;

/** Admit local solves with writable local entries and the selected applications.
 * Only the required solution-to-work and basis/direction-to-work signatures are
 * checked; local storage, callback semantics, and collective use remain contracts.
 */
template <PreconditioningMode mode, class Map, class Solution, class B>
inline constexpr bool SupportsLocalSolve = []
{
   if constexpr (!requires(B &b) {
                    b.work;
                    b.basis(0);
                 })
   {
      return false;
   }
   else
   {
      using Work = WorkspaceWorkVector<B>;
      if constexpr (!details::WritableLocalEntries<Work>() ||
                    !details::WritableLocalEntries<Solution>())
      {
         return false;
      }
      else if constexpr (!requires(const Map &map, const Solution &input, Work &output) {
                            { Apply(map, input, output) } -> std::same_as<void>;
                         })
      {
         return false;
      }
      else if constexpr (mode == PreconditioningMode::None)
      {
         return requires(const Map &map, const WorkspaceBasisVector<B> &input,
                         Work &output) {
            { Apply(map, input, output) } -> std::same_as<void>;
         };
      }
      else if constexpr (requires(B &b) { b.directions(0); })
      {
         return requires(const Map &map, const WorkspaceDirection<B> &input,
                         Work &output) {
            { Apply(map, input, output) } -> std::same_as<void>;
         };
      }
      else
      {
         return false;
      }
   }
}();

} // namespace gendil::gmres_detail

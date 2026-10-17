// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/** @file @brief Solve through the legacy caller-owned split-array GMRES interface.
 *
 * Bind the original map, arithmetic, tolerance, and iteration conventions to the
 * shared restart loop without packing the supplied vectors or scalar arrays.
 */

#include "gendil/Algebra/Vector/localoperations.hpp"
#include "gendil/Algebra/Vector/reductions.hpp"
#include "gendil/Solvers/GMRES/Drivers/constraints.hpp"
#include "gendil/Solvers/GMRES/Drivers/operands.hpp"
#include "gendil/Solvers/GMRES/Workspace/workspaceview.hpp"
#include "gendil/Solvers/GMRES/arnoldi.hpp"
#include "gendil/Solvers/GMRES/leastsquares.hpp"
#include "gendil/Solvers/GMRES/restartedgmres.hpp"
#include "gendil/Utilities/KernelContext/kernelconfiguration.hpp"
#include <iterator>
#include <limits>
#include <type_traits>

namespace gendil::gmres_detail
{

/// Read an available native length for legacy validation, without probing backing storage.
template <class Storage>
   requires (requires(const Storage &storage) { GetVectorSize(storage); } ||
             requires(const Storage &storage) { std::size(storage); })
size_t LegacyLength(const Storage &storage)
{
   if constexpr (requires { GetVectorSize(storage); })
      return GetVectorSize(storage);
   else
      return std::size(storage);
}

/** Adapt legacy caller-owned split arrays without packing or allocating.
 *
 * A zero budget or restart still computes the true residual rhs - A*solution:
 * rhs, solution, work, and basis[0] must be valid matching vectors. Dense arrays
 * may be empty.
 * Iteration additionally requires restart+1 basis vectors, (restart+1)*restart
 * Hessenberg entries, restart cosine/sine/coefficient entries, and restart+1
 * reduced-RHS entries. Native lengths are checked before numerical callbacks;
 * capacities of unsized storage, including raw pointers, remain preconditions.
 *
 * GenDiL views copy entries through shared local copy. Other supported operands
 * retain historical assignment/ADL arithmetic. Their assignment must copy
 * mathematical entries without rebinding, resizing, or allocating storage.
 * Arbitrary assignable descriptors are not thereby supported. Scalar strides
 * and origins remain properties of the supplied native views. Vector operands
 * must also provide the selected historical three-argument AXPY operation.
 */
template <class Map, class RHS, class Dot, class VectorArray, class VectorType,
          class Array, class Orthogonalization>
auto LegacySolve(const Map &map, const RHS &rhs, Dot &dot, Integer max_iters,
                 Integer restart, Real tolerance, VectorType &solution,
                 VectorArray &basis, VectorType &work, Array &hessenberg, Array &cosine,
                 Array &sine, Array &reduced_rhs, Array &reduced_solution,
                 Orthogonalization orthogonalization)
{
   static_assert(IsGMRESOrthogonalization<Orthogonalization>);
   const bool iterate = max_iters != 0 && restart != 0;
   [[maybe_unused]] const size_t limit = std::numeric_limits<size_t>::max();
   GENDIL_VERIFY(restart < limit && (restart == 0 || restart + 1 <= limit / restart),
                 "Legacy GMRES dense size overflow.");
   const size_t rows = restart + 1;
   const size_t required_basis = iterate ? rows : 1;
   const auto check_capacity = [](const auto &storage, size_t required)
   {
      if constexpr (requires { LegacyLength(storage); })
         GENDIL_VERIFY(LegacyLength(storage) >= required,
                       "Legacy GMRES caller storage is too short.");
   };
   const auto check_length = [](const auto &first, const auto &second)
   {
      if constexpr (requires { LegacyLength(first); LegacyLength(second); })
         GENDIL_VERIFY(LegacyLength(first) == LegacyLength(second),
                       "Legacy GMRES vector lengths do not match.");
   };
   check_capacity(basis, required_basis);
   check_capacity(hessenberg, iterate ? rows * restart : 0);
   check_capacity(cosine, iterate ? restart : 0);
   check_capacity(sine, iterate ? restart : 0);
   check_capacity(reduced_rhs, iterate ? rows : 0);
   check_capacity(reduced_solution, iterate ? restart : 0);
   check_length(rhs, solution);
   check_length(work, solution);
   check_length(rhs, work);
   for (size_t j = 0; j < required_basis; ++j)
   {
      auto &&vector = basis[j];
      check_length(vector, solution);
      check_length(vector, rhs);
      check_length(vector, work);
   }

   auto vectors = MakeView(ReferenceContainer{basis}, VectorLayout{rows});
   auto h =
       MakeView(ReferenceContainer{hessenberg},
                StridedLayout<2, DynamicExtents<2>>{std::array<GlobalIndex, 2>{1, rows},
                                                    DynamicExtents<2>{rows, restart}});
   auto cs = MakeView(ReferenceContainer{cosine}, VectorLayout{restart});
   auto sn = MakeView(ReferenceContainer{sine}, VectorLayout{restart});
   auto g = MakeView(ReferenceContainer{reduced_rhs}, VectorLayout{rows});
   auto y = MakeView(ReferenceContainer{reduced_solution}, VectorLayout{restart});
   NoDirections directions;
   auto buffers = TieGMRESBuffers(vectors, work, directions, h, cs, sn, g, y);
   HostKernelConfiguration context;

   // Vector operations retain legacy assignment and ADL dispatch.
   auto apply_system = [&](auto &input, auto &output)
   {
      (void)map(input, output);
   };
   auto copy_vector = [&](const auto &input, auto &output)
   {
      if constexpr (!is_view_v<std::remove_cvref_t<decltype(input)>> &&
                    !is_view_v<std::remove_cvref_t<decltype(output)>> &&
                    requires { output = input; })
      {
         output = input;
      }
      else
      {
         LocalCopy(context, input, output);
      }
   };
   auto axpy = [](Real alpha, const auto &input, auto &output)
   {
      using gendil::Axpy;
      Axpy(alpha, input, output);
   };
   auto divide_vector = [&](auto &output, Real length)
   {
      if constexpr (!is_view_v<std::remove_cvref_t<decltype(output)>> &&
                    requires { output /= length; })
      {
         output /= length;
      }
      else
      {
         LocalDivide(context, output, length);
      }
   };
   auto norm = [&](const auto &input)
   {
      return NormFromDot(dot, input);
   };
   auto project = [&](auto &sequence, Integer count, const auto &input, auto &c)
   {
      ProjectFromDot(dot, sequence, count, input, c);
   };
   auto combine = [&](auto &sequence, Integer count, const auto &c, Real alpha, auto &out)
   {
      for (Integer j = 0; j < count; ++j)
      {
         auto &&vector = sequence(j);
         axpy(alpha * c[j], vector, out);
      }
   };

   // Reduced coefficients and the selected orthogonalization.
   auto store_coefficient = [](auto &c, Integer j, Real value)
   {
      StoreCoefficient(c, j, value);
   };
   auto accumulate_coefficients =
       [&](auto &destination, const auto &source, Integer count)
   {
      auto x = MakeView(ReferenceContainer{source}, VectorLayout{count});
      auto z = MakeView(ReferenceContainer{destination}, VectorLayout{count});
      Axpy(context, Real(1), x, z);
   };
   auto orthogonalize = [&](Integer column)
   {
      auto hessenberg_column = MakeHessenbergColumnView(h, column);
      if constexpr (std::is_same_v<Orthogonalization, ModifiedGramSchmidt>)
      {
         return OrthogonalizeModifiedGramSchmidt(
             dot, axpy, store_coefficient, vectors, column + 1, work, hessenberg_column);
      }
      else
      {
         return OrthogonalizeClassicalGramSchmidtTwoPass(
             project, combine, accumulate_coefficients, vectors, column + 1, work,
             hessenberg_column, y);
      }
   };

   // Expansion and true-residual composition.
   auto expand = [&](Integer column)
   {
      auto &&vector = vectors(column);
      apply_system(vector, work);
   };
   auto build_arnoldi_column = [&](Integer column)
   {
      return BuildArnoldiColumn(work, column, expand, orthogonalize, norm);
   };
   auto &&residual = vectors(0);
   auto compute_true_residual_norm = [&]
   {
      return ComputeTrueResidualNorm(apply_system, rhs, solution, residual, work,
                                     copy_vector, axpy, norm);
   };

   // Reduced least-squares operations.
   auto initialize_reduced_problem = [&](Real beta)
   {
      InitializeReducedLeastSquares(cs, sn, g, y, restart, beta);
   };
   auto update_reduced_qr = [&](Integer column, Real next)
   {
      return UpdateReducedQR(h, cs, sn, g, column, next);
   };
   auto solve_reduced_system = [&](Integer count)
   {
      return BackSubstitute(h, g, y, count);
   };

   // Correction and generic invocation.
   auto apply_correction = [&](Integer count)
   {
      combine(vectors, count, y, Real(1), solution);
   };

   // Historical tolerance/budget translation stays outside the generic algorithm.
   const Real rhs_norm = norm(rhs);
   GMRESOptions options;
   options.restart = restart;
   options.max_iterations = restart == 0 ? 0 : max_iters;
   options.relative_tolerance = tolerance;
   options.absolute_tolerance = rhs_norm == 0 ? tolerance : 0;
   return RestartedGMRES(
       options,
       buffers,
       rhs_norm,
       compute_true_residual_norm,
       build_arnoldi_column,
       apply_correction,
       copy_vector,
       divide_vector,
       initialize_reduced_problem,
       update_reduced_qr,
       solve_reduced_system);
}

} // namespace gendil::gmres_detail

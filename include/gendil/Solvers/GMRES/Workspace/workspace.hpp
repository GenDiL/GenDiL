// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief Reuse owned GMRES scratch across solves and right-hand sides. */

#include "gendil/Algebra/Vector/access.hpp"
#include "gendil/Algebra/Vector/slices.hpp"
#include "gendil/Solvers/GMRES/Workspace/reductionworkspace.hpp"
#include "gendil/Solvers/GMRES/Workspace/workspaceview.hpp"

#include <cstddef>
#include <limits>
#include <vector>

namespace gendil {

/**
 * @brief Allocate GMRES scratch once and reuse it for successive linear solves.
 *
 * Use this owner when the host program calls GMRES, including when its vector
 * operations run on a GPU. For a CPU solve with A, rhs, and solution already
 * supplied, include @c gendil/Solvers/GMRES/gmres.hpp and keep the workspace
 * between calls:
 * @code
 * GMRESOptions options;
 * options.restart = 20;
 * options.relative_tolerance = 1.e-10;
 * auto workspace = MakeGMRESWorkspace(GetVectorSize(solution), options.restart);
 * EuclideanDot<HostKernelConfiguration> dot{workspace.reduction_workspace};
 * auto result = GMRES(A, rhs, solution, dot, options, workspace);
 * // solution now holds the approximation; result.success reports convergence.
 * @endcode
 *
 * The Krylov basis stores the orthonormal search vectors. An additional work
 * vector holds system applications; right-preconditioned solves also need
 * preconditioned directions. The small dense arrays hold the reduced
 * least-squares problem and its rotations and correction coefficients.
 *
 * Basis vectors occupy slices of one owning Vector, contiguous in each memory
 * space that Vector uses. Work and directions have separate Vector owners;
 * the dense arrays share one host std::vector. Preparation reuses sufficient
 * capacities at unchanged vector length. It reconstructs vector storage when
 * that length changes; it does not prepare arithmetic callbacks' resources.
 *
 * MakeGMRESWorkspaceView lends this storage without copying vector entries,
 * allocating, or synchronizing memory. Keep the owner alive, unmoved, and
 * unresized while a borrowed view is in use. Solves prepare fresh views after
 * any growth. Explicit dot/projection implementations provision their own
 * reduction resources; reduction_workspace is optional convenience storage and
 * starts empty. The CPU dot above needs no reduction allocation. Device
 * arithmetic must be supplied with the scratch its selected operations require.
 * Use separate scratch for simultaneous solves. For scratch already supplied
 * by the caller, or for a solve inside a kernel, use GMRESWorkspaceView instead
 * of this owner.
 *
 * The fields are public because this type is an owning bundle rather than an
 * abstraction boundary. Mutating them directly is a low-level operation: all
 * sizes and capacities must continue to describe the associated storage.
 */
struct GMRESWorkspace
{
   /// Logical length shared by every basis, work, and direction vector.
   size_t vector_size = 0;
   /// Reserved Arnoldi steps; basis storage has one additional vector.
   size_t restart_capacity = 0;
   /// Number of reserved preconditioned directions.
   size_t direction_capacity = 0;

   /// Contiguous storage for the Krylov basis vectors.
   Vector basis_storage;
   /// Temporary system application or combined correction.
   Vector work_vector;
   /// One reusable fixed-right direction, or stored flexible directions.
   Vector preconditioned_direction_storage;
   /// Optional arithmetic scratch; algorithm preparation leaves it unchanged.
   Vector reduction_workspace;
   /// Packed host Hessenberg, rotation, reduced-RHS, and coefficient storage.
   std::vector<Real> dense_workspace;
};

/**
 * @brief Prepare owning scratch while preserving sufficient allocations.
 *
 * Request zero directions for unpreconditioned GMRES, one for fixed right
 * preconditioning, or @p restart for flexible right preconditioning. With an
 * unchanged vector length, shorter requests keep existing capacities. A new
 * vector length resets the recorded capacities to the requested values and
 * reconstructs vector storage; dense storage grows as needed but never shrinks.
 * These changes can invalidate previously borrowed views.
 *
 * The restart must be positive. For retained restart c, direction count d,
 * and vector length n, storage holds (c+1)*n basis entries, n work entries,
 * d*n direction entries, and at least c*(c+1)+4*c+1 dense entries. Host checks
 * reject unrepresentable allocation sizes. Reduction scratch is not modified.
 */
inline void PrepareGMRESAlgorithmWorkspace(GMRESWorkspace& workspace,
                                          const size_t vector_size,
                                          const Integer restart,
                                          const size_t direction_count = 0)
{
   GENDIL_VERIFY(restart > 0, "GMRES restart must be positive.");
   const size_t requested_restart = static_cast<size_t>(restart);
   const bool new_extent = workspace.vector_size != vector_size;
   const size_t next_restart =
      new_extent || workspace.restart_capacity < requested_restart
         ? requested_restart : workspace.restart_capacity;
   const size_t next_directions =
      new_extent || workspace.direction_capacity < direction_count
         ? direction_count : workspace.direction_capacity;
   const size_t limit = std::numeric_limits<size_t>::max();
   GENDIL_VERIFY(next_restart < limit, "GMRES restart capacity overflow.");
   const size_t leading_dimension = next_restart + 1;
   GENDIL_VERIFY(vector_size == 0 || leading_dimension <= limit / vector_size,
      "GMRES Krylov allocation size overflow.");
   GENDIL_VERIFY(vector_size == 0 || next_directions <= limit / vector_size,
      "GMRES preconditioned-direction allocation size overflow.");
   const size_t dense_size =
      gmres_detail::GMRESDenseSize(next_restart, leading_dimension);

   if (new_extent || workspace.restart_capacity < requested_restart)
   {
      workspace.basis_storage = Vector(leading_dimension * vector_size);
   }
   if (new_extent) { workspace.work_vector = Vector(vector_size); }
   if (new_extent || workspace.direction_capacity < direction_count)
   {
      workspace.preconditioned_direction_storage =
         Vector(next_directions * vector_size);
   }
   if (workspace.dense_workspace.size() < dense_size)
   {
      workspace.dense_workspace.resize(dense_size);
   }
   workspace.vector_size = vector_size;
   workspace.restart_capacity = next_restart;
   workspace.direction_capacity = next_directions;
}

/// Construct and prepare an owning workspace without copying vector entries.
inline GMRESWorkspace MakeGMRESWorkspace(const size_t vector_size,
                                         const Integer restart,
                                         const size_t direction_count = 0)
{
   GMRESWorkspace workspace;
   PrepareGMRESAlgorithmWorkspace(workspace, vector_size, restart, direction_count);
   return workspace;
}

/**
 * @brief Borrow the prepared owner without allocation or synchronization.
 *
 * Checks the prepared owner's declared capacities against its storage, then
 * returns slices of the basis and directions, a reference to the work vector,
 * and views of the packed dense arrays. Only borrowing descriptors are copied.
 * A shorter active restart still uses the owner's stored Hessenberg leading
 * dimension, restart_capacity+1.
 *
 * The owner must remain alive, unmoved, and unprepared again until the returned
 * view is no longer used. This view does not automatically rebind after growth.
 */
inline auto MakeGMRESWorkspaceView(GMRESWorkspace& workspace)
{
   GENDIL_VERIFY(workspace.restart_capacity > 0,
      "Prepare GMRES workspace capacity before borrowing a view.");
   const size_t limit = std::numeric_limits<size_t>::max();
   GENDIL_VERIFY(workspace.restart_capacity < limit,
      "GMRES restart capacity overflow.");
   const size_t leading_dimension = workspace.restart_capacity + 1;
   GENDIL_VERIFY(workspace.vector_size == 0 ||
      leading_dimension <= limit / workspace.vector_size,
      "GMRES Krylov workspace size overflow.");
   GENDIL_VERIFY(workspace.vector_size == 0 ||
      workspace.direction_capacity <= limit / workspace.vector_size,
      "GMRES direction workspace size overflow.");
   GENDIL_VERIFY(workspace.basis_storage.Size() >=
      leading_dimension * workspace.vector_size,
      "GMRES basis storage does not match its declared capacity.");
   GENDIL_VERIFY(workspace.work_vector.Size() == workspace.vector_size,
      "GMRES work vector does not match its declared extent.");
   GENDIL_VERIFY(workspace.preconditioned_direction_storage.Size() >=
      workspace.direction_capacity * workspace.vector_size,
      "GMRES direction storage does not match its declared capacity.");
   auto basis = MakeVectorSlices(workspace.basis_storage,
      workspace.vector_size, leading_dimension);
   auto directions = MakeVectorSlices(workspace.preconditioned_direction_storage,
      workspace.vector_size, workspace.direction_capacity);
   auto dense = MakeGMRESPackedDense(
      MakeVectorView(workspace.dense_workspace.data(), workspace.dense_workspace.size()),
      workspace.restart_capacity, leading_dimension);
   // Only descriptors are copied. Slices reference owner fields, not these locals.
   return GMRESWorkspaceView<decltype(basis), Vector&, decltype(directions),
      decltype(dense.hessenberg), decltype(dense.cosines), decltype(dense.sines),
      decltype(dense.reduced_rhs), decltype(dense.coefficients)>{
      basis, workspace.work_vector, directions, dense.hessenberg,
      dense.cosines, dense.sines, dense.reduced_rhs, dense.coefficients};
}

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Budget temporary storage for GMRES vector products and projections.
 *
 * Host-controlled device operations borrow a reduction Vector. A cooperative
 * solve inside a kernel instead uses its context's temporary shared-memory
 * arena. The helpers below size these distinct resources in Real entries,
 * independently of the basis vectors and reduced-system arrays.
 */

#include "gendil/Utilities/KernelContext/isthreadeddim.hpp"
#include "gendil/prelude.hpp"
#include "gendil/Algebra/Vector/reductions.hpp"

#include <cstddef>
#include <limits>

namespace gendil {

/**
 * @brief Size the temporary shared arena needed by cooperative GMRES reductions.
 *
 * Use this value when sizing KernelContext for a local GMRES solve with the
 * supplied Euclidean operations. For t participating threads and restart m,
 * the bound is max(m,2*t) Real entries, not bytes: projections need up to m
 * entries and the scaled norm needs 2*t. Non-threaded contexts return zero.
 *
 * This excludes persistent solver storage. Budget persistent shared scratch
 * plus the maximum of the reduction requirement, peak operator scratch, and
 * peak preconditioner scratch: these temporary phases do not overlap. The
 * supplied reductions restore their arena mark before returning. Custom
 * callbacks must account for any additional temporary storage they require.
 */
template<class Context, Integer Restart>
inline constexpr size_t required_local_gmres_reduction_shared_memory_v =
   is_threaded_v<Context>
      ? (static_cast<size_t>(Restart) > 2 * Context::GetNumberOfThreads()
         ? static_cast<size_t>(Restart) : 2 * Context::GetNumberOfThreads())
      : 0;

/**
 * @brief Obtain a convenient reduction-Vector size for host-driven device GMRES.
 *
 * For a nonempty vector of length n and restart m, the bound is
 * (ceil(n/256)+1)*max(m,2) Real entries. It covers the supplied partitioned
 * Euclidean products, scaled norms, batched projections, and coefficient
 * transfers used by fused corrections. The factor 256 is their reduction
 * chunk length; at least two columns cover the scaled-norm storage.
 *
 * This optional upper bound allocates nothing and is not a solver-enforced
 * minimum. The selected arithmetic operations own their resources and define
 * their minimum sizes; custom callbacks may have different requirements.
 * CPU arithmetic needs no reduction Vector. For n == 0 this helper still
 * returns max(m,2), a conservative convention rather than a requirement:
 * an empty system performs no reductions and needs no iteration scratch.
 */
inline size_t RequiredGMRESReductionWorkspaceSize(const size_t vector_size,
                                                 const size_t restart)
{
   const size_t columns = restart < 2 ? 2 : restart;
   return vector_size == 0 ? columns :
      RequiredProjectionReductionWorkspaceSize(vector_size, columns);
}

} // namespace gendil

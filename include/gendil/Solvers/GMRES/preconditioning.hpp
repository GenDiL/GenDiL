// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/** @file @brief Identify how a GMRES solve uses a right preconditioner.
 *
 * A preconditioner supplies an approximate inverse action P: Arnoldi expands
 * the system along P(v) instead of v. Fixed-right GMRES applies a fixed linear
 * P during expansion and again to the combined correction. Flexible GMRES
 * stores the individual P(v) directions and combines them without reapplying P,
 * so the preconditioner may change between calls.
 *
 * This header contains only internal selection metadata. The drivers bind the
 * preconditioner applications and correction operations to the shared algorithm.
 */
namespace gendil::gmres_detail
{
/// Choose ordinary basis vectors, one reusable direction, or retained directions.
enum class PreconditioningMode
{
   None,
   FixedRight,
   FlexibleRight
};
/// Empty placeholder for a solve that never invokes a preconditioner.
struct NoPreconditioner
{
};
} // namespace gendil::gmres_detail

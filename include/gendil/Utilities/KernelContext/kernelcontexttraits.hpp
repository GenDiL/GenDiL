// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Placement traits distinguish host and device execution/memory behavior.
#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"

// Logical-threading traits select register-only helpers vs threaded helpers.
#include "gendil/Utilities/KernelContext/isthreadeddim.hpp"

// Batching/audit traits guard unsupported BatchSize > 1 device paths.
#include "gendil/Utilities/KernelContext/batchingeligibility.hpp"

/**
 * @file kernelcontexttraits.hpp
 * @brief Umbrella include for KernelConfiguration and KernelContext traits.
 *
 * GenDiL keeps three separate concepts explicit:
 *
 * - Placement traits (`is_host_configuration_v`, `is_device_configuration_v`)
 *   answer where a kernel configuration executes and which memory/access
 *   path should be used.
 * - Logical-threading traits (`is_threaded_dim_v`, `is_threaded_v`) answer
 *   whether a work item has a non-empty logical thread layout and should use
 *   threaded/shared helper implementations.
 * - Batching/audit traits (`is_batched_device_configuration_v`,
 *   `is_unbatched_operator_configuration_allowed_v`, and
 *   `GENDIL_REQUIRE_UNBATCHED_OPERATOR`) answer whether a device
 *   configuration has multiple batch lanes and whether a path is allowed to
 *   instantiate for it.
 *
 * In particular,
 * `DeviceKernelConfiguration<ThreadBlockLayout<>, ..., BatchSize>` is device
 * placed and may be batched, but it is not logically threaded. Helper
 * implementation dispatch must use `is_threaded_v`, not placement traits.
 */

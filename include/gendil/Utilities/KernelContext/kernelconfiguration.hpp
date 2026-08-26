// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file kernelconfiguration.hpp
 * @brief Public umbrella for kernel layouts, contexts, policies, and traits.
 *
 * Include this header when selecting a host or device execution policy or when
 * writing code generic over KernelContext.  Narrow implementation headers may
 * still be included directly when only one facility is required.
 */

// Logical thread-block layouts shared by host and device configurations.
#include "gendil/Utilities/KernelContext/threadlayout.hpp"

// Per-work-item execution context and shared-memory sizing utilities.
#include "gendil/Utilities/KernelContext/kernelcontext.hpp"

// Host, legacy thread-first, batched device, and serial configurations.
#include "gendil/Utilities/KernelContext/KernelConfigurations/kernelconfigurations.hpp"

// Placement, logical-threading, and batching eligibility traits.
#include "gendil/Utilities/KernelContext/kernelcontexttraits.hpp"

// Compile-time validation of thread-layout coverage for tensor shapes.
#include "gendil/Utilities/KernelContext/threadedshapecoverage.hpp"

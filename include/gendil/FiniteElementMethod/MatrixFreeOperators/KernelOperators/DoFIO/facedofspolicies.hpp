// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file facedofspolicies.hpp
 * @brief Configuration policies for reading and writing facet DoFs.
 *
 * Direct-global access is the default. Kernel configurations may select the
 * full-shared alternatives by defining face_read_dofs_policy or
 * face_write_dofs_policy.
 */

#include <type_traits>

namespace gendil {

/** @brief Stage a complete oriented facet in shared memory before reading. */
struct FullSharedFaceReadDofsPolicy
{
};

/** @brief Read facet DoFs through an oriented view of global storage. */
struct DirectGlobalFaceReadDofsPolicy
{
};

/** @brief Stage a complete oriented facet in shared memory before writing. */
struct FullSharedFaceWriteDofsPolicy
{
};

/** @brief Write facet DoFs through an oriented view of global storage. */
struct DirectGlobalFaceWriteDofsPolicy
{
};

/** @brief Extract a kernel configuration's facet-read policy. */
template < typename KernelConfiguration, typename = void >
struct face_read_dofs_policy
{
   using type = DirectGlobalFaceReadDofsPolicy;
};

template < typename KernelConfiguration >
struct face_read_dofs_policy<
   KernelConfiguration,
   std::void_t< typename KernelConfiguration::face_read_dofs_policy > >
{
   using type = typename KernelConfiguration::face_read_dofs_policy;
};

/** @brief Selected facet-read policy, defaulting to direct-global access. */
template < typename KernelConfiguration >
using face_read_dofs_policy_t =
   typename face_read_dofs_policy<
      std::remove_cvref_t< KernelConfiguration > >::type;

/** @brief Extract a kernel configuration's facet-write policy. */
template < typename KernelConfiguration, typename = void >
struct face_write_dofs_policy
{
   using type = DirectGlobalFaceWriteDofsPolicy;
};

template < typename KernelConfiguration >
struct face_write_dofs_policy<
   KernelConfiguration,
   std::void_t< typename KernelConfiguration::face_write_dofs_policy > >
{
   using type = typename KernelConfiguration::face_write_dofs_policy;
};

/** @brief Selected facet-write policy, defaulting to direct-global access. */
template < typename KernelConfiguration >
using face_write_dofs_policy_t =
   typename face_write_dofs_policy<
      std::remove_cvref_t< KernelConfiguration > >::type;

} // namespace gendil

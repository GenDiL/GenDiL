// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief Borrow equally sized vectors from one contiguous vector. */

#include "gendil/Algebra/Vector/access.hpp"

#include <cstddef>
#include <limits>
#include <type_traits>

namespace gendil {

/**
 * @brief Treat a named vector as a sequence of adjacent vector slices.
 *
 * @code
 * Vector storage(12);
 * auto slices = MakeVectorSlices(storage, 4, 3);
 * auto second = slices(1);
 * WriteHostVector(second)[2] = 7; // Writes storage[6].
 * @endcode
 *
 * The parent is borrowed, never copied or normalized. Keep it alive and stable,
 * including when it is itself a view descriptor. Each access creates a native
 * subview without allocating or choosing host/device storage. A parent subview's
 * origin is shared by individual slices and bulk access through ContiguousStorage.
 */
template<class Parent>
struct VectorSlices
{
   Parent& storage;
   size_t extent;
   size_t count;

   GENDIL_HOST_DEVICE
   auto operator()(const size_t index) const
   {
      return MakeVectorView(storage, index * extent, extent);
   }

   GENDIL_HOST_DEVICE
   Parent& ContiguousStorage() const { return storage; }

   GENDIL_HOST_DEVICE
   size_t ContiguousStride() const { return extent; }

   GENDIL_HOST_DEVICE
   size_t Size() const { return count; }
};

/// Borrow count slices of extent entries; temporary parent descriptors are rejected.
template<class Parent>
   requires std::is_lvalue_reference_v<Parent&&> &&
      requires(Parent& parent)
      {
         GetVectorSize(parent);
         MakeVectorView(parent, size_t{0}, size_t{0});
      }
GENDIL_HOST_DEVICE
auto MakeVectorSlices(Parent&& parent, const size_t extent, const size_t count)
{
   [[maybe_unused]] const size_t limit = std::numeric_limits<size_t>::max();
   GENDIL_VERIFY(extent == 0 || count <= limit / extent,
      "Vector collection size overflow.");
   GENDIL_VERIFY(extent * count <= GetVectorSize(parent),
      "Vector collection exceeds its parent vector.");
   return VectorSlices<std::remove_reference_t<Parent>>{parent, extent, count};
}

} // namespace gendil

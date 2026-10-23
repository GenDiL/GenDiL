// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Runtime-sized, one-dimensional vector view layout.
 */

#include "gendil/Utilities/View/view.hpp"

#include <cstddef>

namespace gendil {

/**
 * @brief Identity layout for a one-dimensional vector view.
 *
 * The layout stores the logical extent while mapping index @p i directly to
 * container key @p i. Physical contiguity is a property of the associated
 * container: pointer-backed vector views are contiguous, while owner-bound
 * views delegate access to their parent vector.
 * Its one runtime extent supports vector sizing and subrange validation.
 * Compile-time tensor sizes belong to fixed layouts, not this vector adaptor.
 */
struct VectorLayout
{
   static constexpr Integer rank = 1;

   GlobalIndex extent{};

   GENDIL_HOST_DEVICE
   constexpr VectorLayout() = default;

   template<detail::ExtentIndex Index>
   GENDIL_HOST_DEVICE
   constexpr explicit VectorLayout(Index size)
      : extent(detail::CheckedExtentCast<GlobalIndex>(size))
   { }

   template < size_t Axis >
      requires ( Axis == 0 )
   GENDIL_HOST_DEVICE
   constexpr GlobalIndex Get( AxisTag< Axis > ) const
   {
      return extent;
   }

   template<size_t Axis>
      requires (Axis == 0)
   GENDIL_HOST_DEVICE
   static constexpr auto Stride(AxisTag<Axis>)
   {
      return std::integral_constant<GlobalIndex, 1>{};
   }

   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex Stride(size_t)
   {
      return 1;
   }

   template<class Index>
   GENDIL_HOST_DEVICE
   constexpr GlobalIndex Offset(const Index index) const
   {
      return static_cast<GlobalIndex>(index);
   }

   GENDIL_HOST_DEVICE
   constexpr GlobalIndex Size() const
   {
      return extent;
   }
};

/** @brief Create a runtime-sized vector view over contiguous raw storage. */
template<class T, detail::ExtentIndex Index>
GENDIL_HOST_DEVICE
auto MakeVectorView(T* const data, const Index extent)
{
   return MakeView(data, VectorLayout{extent});
}

/** @brief Create a runtime-sized vector view from a shallow pointer handle. */
template<class T, class IndexType, detail::ExtentIndex ExtentIndex>
GENDIL_HOST_DEVICE
auto MakeVectorView(
   PointerContainer<T, IndexType> data,
   const ExtentIndex extent)
{
   return MakeView(data, VectorLayout{extent});
}

} // namespace gendil

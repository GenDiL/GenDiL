// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief Translate a layout's container offsets by a fixed base. */

#include "gendil/Utilities/View/Layouts/strides.hpp"

#include <cstddef>
#include <utility>

namespace gendil {

/**
 * @brief Compose a layout with a base offset into the same container.
 *
 * Logical extents and strides are inherited from the parent layout. Checked
 * construction belongs to the factory selecting the addressed container;
 * ordinary indexing performs only the translated offset calculation.
 */
template<class ParentLayout>
struct OffsetLayout
{
   static constexpr size_t rank = ParentLayout::rank;

   [[no_unique_address]] ParentLayout parent;
   GlobalIndex base_offset;

   template<class... Indices>
   GENDIL_HOST_DEVICE
   constexpr GlobalIndex Offset(Indices&&... indices) const
   {
      return base_offset + parent.Offset(std::forward<Indices>(indices)...);
   }

   template<size_t Axis>
      requires has_extent_v<Axis, ParentLayout>
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) Get(AxisTag<Axis> axis) const
   {
      return parent.Get(axis);
   }

   template<size_t Axis>
      requires requires(const ParentLayout& layout)
      {
         layout.Stride(AxisTag<Axis>{});
      }
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) Stride(AxisTag<Axis> axis) const
   {
      return parent.Stride(axis);
   }

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) Stride(const size_t axis) const
      requires requires(const ParentLayout& layout)
      {
         layout.Stride(axis);
      }
   {
      return parent.Stride(axis);
   }

   GENDIL_HOST_DEVICE
   constexpr GlobalIndex Size() const
      requires has_complete_extents_v<ParentLayout>
   {
      return GetLogicalSize(parent);
   }

   GENDIL_HOST_DEVICE
   constexpr GlobalIndex BaseOffset() const { return base_offset; }
};

template<class ParentLayout, detail::ExtentIndex InputIndex>
GENDIL_HOST_DEVICE
constexpr auto MakeOffsetLayout(
   ParentLayout parent,
   const InputIndex base_offset)
{
   return OffsetLayout<ParentLayout>{std::move(parent),
      detail::CheckedExtentCast<GlobalIndex>(base_offset)};
}

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/get.hpp"

#include <utility>

namespace gendil
{

namespace details
{
   template < typename KernelContext, typename Shape, typename Indices >
   struct ThreadedShapeCoveredImpl;

   template <
      typename KernelContext,
      typename Shape,
      size_t... I >
   struct ThreadedShapeCoveredImpl<
      KernelContext,
      Shape,
      std::index_sequence< I... > >
   {
      static constexpr bool value =
         ( ( KernelContext::template GetBlockDim< I >() >=
             seq_get_v< I, Shape > ) && ... );
   };
}

template < typename KernelContext, typename Shape >
struct ThreadedShapeCovered;

template < typename KernelContext, size_t... ShapeDims >
struct ThreadedShapeCovered<
   KernelContext,
   std::index_sequence< ShapeDims... > >
{
   static constexpr size_t num_mapped_thread_dims =
      KernelContext::thread_block_dim < sizeof...( ShapeDims )
         ? KernelContext::thread_block_dim
         : sizeof...( ShapeDims );

   static constexpr bool value =
      details::ThreadedShapeCoveredImpl<
         KernelContext,
         std::index_sequence< ShapeDims... >,
         std::make_index_sequence< num_mapped_thread_dims > >::value;
};

template < typename KernelContext, typename Shape >
inline constexpr bool threaded_shape_covered_v =
   ThreadedShapeCovered< KernelContext, Shape >::value;

} // namespace gendil

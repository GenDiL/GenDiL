// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil {

template <Integer LowDim, Integer HighDim, Integer LocalFaceIndexLow>
struct CartesianCrossDimConformingIntermesh
{
   static_assert(LowDim >= 1, "LowDim must be >= 1");
   static_assert(HighDim >= LowDim, "Require HighDim >= LowDim");
   static_assert(0 <= LocalFaceIndexLow && LocalFaceIndexLow < 2*LowDim,
                  "LocalFaceIndexLow out of range");

   // ---- Low-side face selection (axis/sign in LowDim) ----
   static constexpr Integer axis = LocalFaceIndexLow % LowDim;
   static constexpr int sign = (LocalFaceIndexLow < LowDim ? -1 : +1); // −face or +face
   static constexpr Integer local_face_index_low = LocalFaceIndexLow;
   static constexpr Integer local_face_index_high = axis + (sign == -1? HighDim : 0);


   // Useful compile-time lists of axes
   static constexpr std::array<int, LowDim-1> low_tan_axes = []{
      std::array<int, LowDim-1> t{};
      int m = 0;
      for (int d = 0; d < LowDim; ++d) if (d != axis) t[m++] = d;
      return t;
   }();

   using geometry_low       = HyperCube<LowDim>;
   using geometry_high      = HyperCube<HighDim>;
   using orientation_low    = IdentityOrientation<LowDim>;
   using orientation_high   = IdentityOrientation<HighDim>;
   using conf_low           = ConformingFaceMap<LowDim>;
   using conf_high          = ConformingFaceMap<HighDim>;

   using minus_normal_type  = CanonicalVector<LowDim,  axis, +sign>;
   using plus_normal_type  = CanonicalVector<HighDim,  axis, -sign>;

   using minus_view_type =
      FaceView<
         std::integral_constant<Integer, local_face_index_low>,
         geometry_low, orientation_low, minus_normal_type, conf_low
      >;
   using plus_view_type =
      FaceView<
         std::integral_constant<Integer, local_face_index_high>,
         geometry_high, orientation_high, plus_normal_type, conf_high
      >;
   using face_info_type = GlobalFaceInfo<minus_view_type, plus_view_type>;

   // ---- Stored sizes and mapping ----
   std::array<GlobalIndex, LowDim>  sizes_low  {};  // low mesh counts
   std::array<GlobalIndex, HighDim> sizes_high {};  // high mesh counts

   // Number of (minus,plus) pairs = product of high tangential extents (all d != normal_axis_high)
   GlobalIndex num_pairs{0};

   // ---- Constructor ----
   GENDIL_HOST_DEVICE
   CartesianCrossDimConformingIntermesh(
      std::array<GlobalIndex, LowDim>  sl,
      std::array<GlobalIndex, HighDim> sh)
   : sizes_low(sl), sizes_high(sh)
   {
      // mark used high tangential axes
      for (int d = 0; d < LowDim-1; ++d)
      {
         if ( d != axis ) assert(sizes_low[d] == sizes_high[d] && "Tangential sizes must match");
      }

      // num_pairs = product over all high *tangential* axes (i.e. all d != axis)
      num_pairs = 1;
      for (int d = 0; d < HighDim; ++d)
      {
         if (d != axis) num_pairs *= sizes_high[d];
      }
   }

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const { return num_pairs; }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex& pair_index) const
   {
      // Enumerate over the full high tangential grid (normal axis extent = 1)
      auto face_extents_high = sizes_high;
      face_extents_high[axis] = 1;
      const auto idx_face = GetStructuredSubIndices(pair_index, face_extents_high);

      // ---- Build low cell index: copy mapped tangentials; set boundary slice on axis
      std::array<GlobalIndex, LowDim> idx_low{};
      // set tangentials
      for (int d = 0; d < LowDim; ++d)
      {
         idx_low[d] = idx_face[d];
      }
      // set normal (boundary slice)
      idx_low[axis] = (sign > 0 ? sizes_low[axis] - 1 : 0);

      // ---- Build high cell index: copy all tangential components; set boundary slice on normal_axis_high
      std::array<GlobalIndex, HighDim> idx_high = idx_face;
      idx_high[axis] = (sign > 0 ? sizes_high[axis] - 1 : 0);

      const GlobalIndex minus_linear = ComputeLinearIndex(idx_low,  sizes_low);
      const GlobalIndex plus_linear  = ComputeLinearIndex(idx_high, sizes_high);

      // Compose views (FaceView carries local face index as a value; orientation/normals are canonical)
      minus_view_type minus{ minus_linear };
      plus_view_type  plus { plus_linear };

      return face_info_type{ minus, plus };
   }
};

} // namespace gendil

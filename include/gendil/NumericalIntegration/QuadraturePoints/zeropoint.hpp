// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Geometries/point.hpp"

namespace gendil {

struct ZeroPoint
{
   using index_type = GlobalIndex;

   GENDIL_HOST_DEVICE
   static constexpr Integer GetNumPoints()
   {
      return 1;
   }

   GENDIL_HOST_DEVICE
   static constexpr Point<1> GetPoint( index_type index )
   {
      return Point<1>{ Real(0.0) };
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetCoord( index_type index )
   {
      return Real(0.0);
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      return Real(1.0);
   }
};

}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Geometries/point.hpp"

namespace gendil {

// Gauss-Legendre (GL)
template < Integer NumPoints >
struct GaussLegendrePoints;

template < >
struct GaussLegendrePoints< 1 >
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
      return Point<1>{ Real(0.5) };
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetCoord( index_type index)
   {
      return Real(0.5);
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      return Real(1.0);
   }
};

template < >
struct GaussLegendrePoints< 2 >
{
   using index_type = GlobalIndex;

   GENDIL_HOST_DEVICE
   static constexpr Integer GetNumPoints()
   {
      return 2;
   }

   GENDIL_HOST_DEVICE
   static constexpr Point<1> GetPoint( index_type index )
   {
      return Point<1>{ GetCoord( index ) };
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetCoord( index_type index )
   {
      constexpr Real coord[] = { Real(0.21132486540518711775), Real(0.78867513459481288225) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(0.5), Real(0.5) };
      return weights[ index ];
   }
};

template < >
struct GaussLegendrePoints< 3 >
{
   using index_type = GlobalIndex;

   GENDIL_HOST_DEVICE
   static constexpr Integer GetNumPoints()
   {
      return 3;
   }

   GENDIL_HOST_DEVICE
   static constexpr Point<1> GetPoint( index_type index )
   {
      return Point<1>{ GetCoord( index ) };

   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetCoord( index_type index )
   {
      constexpr Real coord[] = { Real(0.11270166537925831148), Real(0.5), Real(0.88729833462074168852) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(5./18.), Real(4./9.), Real(5./18.) };
      return weights[ index ];
   }
};

template < >
struct GaussLegendrePoints< 4 >
{
   using index_type = GlobalIndex;

   GENDIL_HOST_DEVICE
   static constexpr Integer GetNumPoints()
   {
      return 4;
   }

   GENDIL_HOST_DEVICE
   static constexpr Point<1> GetPoint( index_type index )
   {
      return Point<1>{ GetCoord( index ) };
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetCoord( index_type index )
   {
      constexpr Real coord[] = { Real(0.0694318442029737), Real(0.330009478207572), Real(0.669990521792428), Real(0.930568155797026) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(0.173927422568727), Real(0.326072577431273), Real(0.326072577431273), Real(0.173927422568727) };
      return weights[ index ];
   }
};

template < >
struct GaussLegendrePoints< 5 >
{
   using index_type = GlobalIndex;

   GENDIL_HOST_DEVICE
   static constexpr Integer GetNumPoints()
   {
      return 5;
   }

   GENDIL_HOST_DEVICE
   static constexpr Point<1> GetPoint( index_type index )
   {
      return Point<1>{ GetCoord( index ) };
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetCoord( index_type index )
   {
      constexpr Real coord[] = { Real(0.046910077036680), Real(0.230765344947158), Real(0.500000000000000), Real(0.769234655052842), Real(0.953089922969332) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(0.118463442528095), Real(0.239314335249683), Real(0.284444444444444), Real(0.239314335249683), Real(0.118463442528095) };
      return weights[ index ];
   }
};

template < >
struct GaussLegendrePoints< 6 >
{
   using index_type = GlobalIndex;

   GENDIL_HOST_DEVICE
   static constexpr Integer GetNumPoints()
   {
      return 6;
   }

   GENDIL_HOST_DEVICE
   static constexpr Point<1> GetPoint( index_type index )
   {
      return Point<1>{ GetCoord( index ) };
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetCoord( index_type index )
   {
      constexpr Real coord[] = { Real(0.033769652428984240), Real(0.169395306766868), Real(0.380690406958402), Real(0.619309593041598), Real(0.830604693233132), Real(0.966234757101576) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(0.085662246189585), Real(0.180380786524069), Real(0.233956967286346), Real(0.233956967286346), Real(0.180380786524069), Real(0.085662246189585) };
      return weights[ index ];
   }
};

template < >
struct GaussLegendrePoints< 7 >
{
   using index_type = GlobalIndex;

   GENDIL_HOST_DEVICE
   static constexpr Integer GetNumPoints()
   {
      return 7;
   }

   GENDIL_HOST_DEVICE
   static constexpr Point<1> GetPoint( index_type index )
   {
      return Point<1>{ GetCoord( index ) };
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetCoord( index_type index )
   {
      constexpr Real coord[] = { Real(0.0254460438286207), Real(0.129234407200303), Real(0.297077424311301), Real(0.500000000000000), Real(0.702922575688699), Real(0.87076559279970), Real(0.97455395617138) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(0.0647424830844348), Real(0.139852695744638), Real(0.190915025252559), Real(0.208979591836735), Real(0.190915025252559),  Real(0.139852695744638), Real(0.0647424830844348) };
      return weights[ index ];
   }
};

template < >
struct GaussLegendrePoints< 8 >
{
   using index_type = GlobalIndex;

   GENDIL_HOST_DEVICE
   static constexpr Integer GetNumPoints()
   {
      return 8;
   }

   GENDIL_HOST_DEVICE
   static constexpr Point<1> GetPoint( index_type index )
   {
      return Point<1>{ GetCoord( index ) };
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetCoord( index_type index )
   {
      constexpr Real coord[] = { Real(0.0198550717512319), Real(0.101666761293187), Real(0.237233795041836), Real(0.408282678752175), Real(0.591717321247825), Real(0.76276620495816), Real(0.89833323870681), Real(0.98014492824877) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(0.0506142681451881), Real(0.111190517226687), Real(0.156853322938944), Real(0.181341891689181), Real(0.181341891689181),  Real(0.156853322938944), Real(0.111190517226687), Real(0.0506142681451881) };
      return weights[ index ];
   }
};

}

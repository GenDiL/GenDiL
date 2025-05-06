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
      return Point<1>{ 0.5 };
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetCoord( index_type index)
   {
      return 0.5;
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      return 1.0;
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
      constexpr Real coord[] = { 0.21132486540518711775, 0.78867513459481288225 };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 0.5, 0.5 };
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
      constexpr Real coord[] = { 0.11270166537925831148, 0.5, 0.88729833462074168852 };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 5./18., 4./9., 5./18. };
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
      constexpr Real coord[] = { 0.0694318442029737, 0.330009478207572, 0.669990521792428, 0.930568155797026 };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727 };
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
      constexpr Real coord[] = { 0.0469100770306680, 0.230765344947158, 0.500000000000000, 0.769234655052842, 0.953089922969332 };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 0.118463442528095, 0.239314335249683, 0.284444444444444, 0.239314335249683, 0.118463442528095 };
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
      constexpr Real coord[] = { 0.0337652428984240, 0.169395306766868, 0.380690406958402, 0.619309593041598, 0.830604693233132, 0.966234757101576 };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 0.085662246189585, 0.180380786524069, 0.233956967286346, 0.233956967286346, 0.180380786524069, 0.085662246189585 };
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
      constexpr Real coord[] = { 0.0254460438286207, 0.129234407200303, 0.297077424311301, 0.500000000000000, 0.702922575688699, 0.87076559279970, 0.97455395617138 };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 0.0647424830844348, 0.139852695744638, 0.190915025252559, 0.208979591836735, 0.190915025252559,  0.139852695744638, 0.0647424830844348 };
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
      constexpr Real coord[] = { 0.0198550717512319, 0.101666761293187, 0.237233795041836, 0.408282678752175, 0.591717321247825, 0.76276620495816, 0.89833323870681, 0.98014492824877 };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 0.0506142681451881, 0.111190517226687, 0.156853322938944, 0.181341891689181, 0.181341891689181,  0.156853322938944, 0.111190517226687, 0.0506142681451881 };
      return weights[ index ];
   }
};

}

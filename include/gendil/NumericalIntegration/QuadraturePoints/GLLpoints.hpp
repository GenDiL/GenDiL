// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Geometries/point.hpp"

namespace gendil {

// Gauss-Lobatto-Legendre (GLL)
template < int NumPoints >
struct GaussLobattoLegendrePoints;

template < >
struct GaussLobattoLegendrePoints< 1 >
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
   static constexpr Real GetCoord( index_type index )
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
struct GaussLobattoLegendrePoints< 2 >
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
      constexpr Real coord[] = { 0., 1.0 };
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
struct GaussLobattoLegendrePoints< 3 >
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
      constexpr Real coord[] = { 0., 0.5, 1. };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 1./6., 2./3., 1./6. };
      return weights[ index ];
   }
};

template < >
struct GaussLobattoLegendrePoints< 4 >
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
      constexpr Real val = 1./5.*2.23606797749979;//sqrt(5.);
      constexpr Real coord[] = { 0., 0.5 * (1. - val) , 0.5 * (1. + val), 1. };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 1./12., 5./12., 5./12., 1./12. };
      return weights[ index ];
   }
};

template < >
struct GaussLobattoLegendrePoints< 5 >
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
      constexpr Real coord[] = { 0.0, 0.1726731646460115, 0.5, 0.8273268353539884, 1.0 };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 0.05, 49.0/180.0, 16.0/45.0, 49.0/180.0, 0.05 };
      return weights[ index ];
   }
};

template < >
struct GaussLobattoLegendrePoints< 6 >
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
      constexpr Real coord[] = { 0.0, 0.11747233803526752, 0.35738424175967753, 0.6426157582403225, 0.8825276619647324, 1.0 };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 0.03333333333333333, 0.1892374781489234, 0.27742918851774323, 0.27742918851774323, 0.1892374781489234, 0.03333333333333333 };
      return weights[ index ];
   }
};

template < >
struct GaussLobattoLegendrePoints< 7 >
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
      constexpr Real coord[] = { 0.0, 0.08488805186071652, 0.265575603264643, 0.5, 0.734424396735357, 0.9151119481392835, 1.0 };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 0.023809523809523808, 0.13841302368078304, 0.21587269060493122, 0.2438095238095237, 0.21587269060493122, 0.13841302368078304, 0.023809523809523808 };
      return weights[ index ];
   }
};

template < >
struct GaussLobattoLegendrePoints< 8 >
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
      constexpr Real coord[] = { 0.0, 0.06412992574519649, 0.20414990928342902, 0.3953503910487605, 0.6046496089512394, 0.795850090716571, 0.9358700742548035, 1.0 };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { 0.017857142857142856, 0.10535211357175303, 0.17056134624175218, 0.20622939732935197, 0.20622939732935197, 0.17056134624175218, 0.10535211357175303, 0.017857142857142856 };
      return weights[ index ];
   }
};

}

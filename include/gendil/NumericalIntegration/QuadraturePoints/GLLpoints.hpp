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
      return Point<1>{ Real(0.5) };
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetCoord( index_type index )
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
      constexpr Real coord[] = { Real(0.), Real(1.0) };
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
      constexpr Real coord[] = { Real(0.), Real(0.5), Real(1.) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(1./6.), Real(2./3.), Real(1./6.) };
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
      constexpr Real val = Real(1./5.*2.23606797749979);//sqrt(5.);
      constexpr Real coord[] = { Real(0.), Real(0.5 * (1. - val)), Real(0.5 * (1. + val)), Real(1.) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(1./12.), Real(5./12.), Real(5./12.), Real(1./12.) };
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
      constexpr Real coord[] = { Real(0.0), Real(0.1726731646460115), Real(0.5), Real(0.8273268353539884), Real(1.0) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(0.05), Real(49.0/180.0), Real(16.0/45.0), Real(49.0/180.0), Real(0.05) };
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
      constexpr Real coord[] = { Real(0.0), Real(0.11747233803526752), Real(0.35738424175967753), Real(0.6426157582403225), Real(0.8825276619647324), Real(1.0) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(0.03333333333333333), Real(0.1892374781489234), Real(0.27742918851774323), Real(0.27742918851774323), Real(0.1892374781489234), Real(0.03333333333333333) };
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
      constexpr Real coord[] = { Real(0.0), Real(0.08488805186071652), Real(0.265575603264643), Real(0.5), Real(0.734424396735357), Real(0.9151119481392835), Real(1.0) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(0.023809523809523808), Real(0.13841302368078304), Real(0.21587269060493122), Real(0.2438095238095237), Real(0.21587269060493122), Real(0.13841302368078304), Real(0.023809523809523808) };
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
      constexpr Real coord[] = { Real(0.0), Real(0.06412992574519649), Real(0.20414990928342902), Real(0.3953503910487605), Real(0.6046496089512394), Real(0.795850090716571), Real(0.9358700742548035), Real(1.0) };
      return coord[ index ];
   }

   GENDIL_HOST_DEVICE
   static constexpr Real GetWeight( index_type index )
   {
      constexpr Real weights[] = { Real(0.017857142857142856), Real(0.10535211357175303), Real(0.17056134624175218), Real(0.20622939732935197), Real(0.20622939732935197), Real(0.17056134624175218), Real(0.10535211357175303), Real(0.017857142857142856) };
      return weights[ index ];
   }
};

}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "lagrangeshapefunctions.hpp"
#include "gendil/NumericalIntegration/QuadraturePoints/GLLpoints.hpp"

namespace gendil {

/**
 * @brief Simple structure representing Gauss-Lobatto-Legendre (GLL) 1D shape functions.
 * 
 * @tparam Order The polynomial order of the shape functions.
 */
template < size_t Order >
struct GaussLobattoLegendreShapeFunctions :
    public LagrangeShapeFunctions< GaussLobattoLegendrePoints< Order+1 > >
{ };

template < size_t Order >
GENDIL_HOST_DEVICE
constexpr auto GetDofsSizes( GaussLobattoLegendreShapeFunctions< Order > )
{
   return Order+1;
}

}

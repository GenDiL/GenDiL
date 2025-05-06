// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include "gendil/FiniteElementMethod/finiteelementorders.hpp"
#include "gendil/NumericalIntegration/QuadraturePoints/numpoints.hpp"

namespace gendil {

// TODO: Rename to instantiate_ElementDof?
template < typename OrdersStruct,
           Integer... extra_dims >
struct instantiate_array;

template < Integer... Orders >
struct instantiate_array<
          FiniteElementOrders< Orders... > >
{
   using type = SerialRecursiveArray< Real, (Orders + 1)... >;
};

// TODO: Rename to instantiate_QuadraturePointsValues?
template < Integer... NumPoints,
           Integer... extra_dims >
struct instantiate_array<
          IntegrationRuleNumPoints< NumPoints... >,
          extra_dims... >
{
   using type = SerialRecursiveArray< Real, NumPoints..., extra_dims... >;
};

template < Integer... NumPoints,
           Integer... extra_dims >
struct instantiate_array<
          std::integer_sequence< Integer, NumPoints... >,
          extra_dims... >
{
   using type = SerialRecursiveArray< Real, NumPoints..., extra_dims... >;
};

}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <type_traits>

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/tensorproductdata.hpp"

namespace gendil
{

template<Integer I, class... Entries>
GENDIL_HOST_DEVICE
auto GetCoord(
   const TensorProductData<Entries...>& quad_data,
   Integer q)
{
   const auto& qdata_i = GetTensorProductEntry<I>(quad_data);
   using QDataI = std::remove_cvref_t<decltype(qdata_i)>;

   if constexpr (requires { qdata_i.coord(q); })
   {
      return qdata_i.coord(q);
   }
   else if constexpr (requires { QDataI::GetCoord(q); })
   {
      return QDataI::GetCoord(q);
   }
   else if constexpr (requires { qdata_i.GetCoord(q); })
   {
      return qdata_i.GetCoord(q);
   }
   else
   {
      static_assert(
         dependent_false_v<
            TensorProductData<Entries...>,
            QDataI>,
         "GetCoord<I>(quad_data, q) requires a static point set or "
         "mapped point-set object.");
   }
}

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/Utilities/staticmap.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/readquadraturelocalvalues.hpp"

namespace gendil
{

template < StaticString Name >
struct FiniteElementField : FieldBase
{
   static constexpr auto name = Name;

   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext& kernel_context,
      const WeakFormContext& weak_form_context,
      const OperatorContext& operator_context,
      const ElementContext& element_context,
      const QuadPtContext& quad_pt_context,
      const Fields& fields ) const
   {
      return ReadQuadratureLocalValues(
         kernel_context,
         quad_pt_context.quad_index,
         fields.template get<NameTag<Name>>().values );
   }
};

template < StaticString Name >
std::ostream& operator<<(std::ostream& os, const FiniteElementField<Name>& finiteElementField)
{
   return os << "phi_" << finiteElementField.name.view();
}

} // namespace gendil

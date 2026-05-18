// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"

namespace gendil
{

// Do NOT repeat default template argument here (already in forward declaration in dslbase.hpp)
template < StaticString Name, FieldShape Shape >
struct TrialSpace : FieldBase
{
   static constexpr auto name = Name;
   static constexpr auto field_shape = Shape;

   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext & kernel_context,
      const WeakFormContext & weak_form_context,
      const OperatorContext & operator_context,
      const ElementContext & element_context,
      const QuadPtContext & quad_pt_context,
      const Fields & fields ) const
   {
      return ReadQuadratureLocalValues( kernel_context, quad_pt_context.quad_index, fields.template get<NameTag<Name>>().values );
   }
};

// Alias for vector-valued trial space
template < StaticString Name >
using VectorTrialSpace = TrialSpace<Name, FieldShape::Vector>;

template < StaticString Name, FieldShape Shape >
std::ostream& operator<<(std::ostream& os, const TrialSpace<Name, Shape>& trialSpace)
{
   return os << "u_" << trialSpace.name.view();
}

} // namespace gendil

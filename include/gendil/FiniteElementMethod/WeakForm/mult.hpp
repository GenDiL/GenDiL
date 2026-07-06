// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/productkind.hpp"

namespace gendil
{

template < FieldExpr LHS, FieldExpr RHS >
struct MultFieldExpr : FieldBase
{
   LHS lhs;
   RHS rhs;

   GENDIL_HOST_DEVICE
   MultFieldExpr(const LHS& lhs_, const RHS& rhs_)
      : lhs(lhs_), rhs(rhs_)
   {}

   template < typename ... Args >
   GENDIL_HOST_DEVICE
   auto operator()(Args&&... args) const
   {
      return lhs(std::forward<Args>(args)...) * rhs(std::forward<Args>(args)...);
   }
};

template < FieldExpr LHS, FieldExpr RHS >
std::ostream& operator<<(std::ostream& os, const MultFieldExpr<LHS, RHS>& prod)
{
   return os << "(" << prod.lhs << " * " << prod.rhs << ")";
}

/**
 * @brief Legacy generic operator* → MultFieldExpr
 *
 * This is the fallback multiplication operator for cases NOT handled by
 * ProductExpr (ScalarTimes, MatVec, and MatMat).
 *
 * Constraint: Excludes is_productexpr_syntax_candidate_v to ensure
 * mutual exclusion with ProductExpr operator*.
 *
 * Unsupported or intentionally ambiguous products remain in this legacy path.
 */
template < FieldExpr LHS, FieldExpr RHS >
   requires (!is_productexpr_syntax_candidate_v<LHS, RHS>)
GENDIL_HOST_DEVICE
auto operator*(const LHS& lhs, const RHS& rhs)
{
   return MultFieldExpr<LHS, RHS>(lhs, rhs);
}

} // namespace gendil

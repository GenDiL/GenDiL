// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil
{

// Field shape classification - semantic rank without concrete dimensions
enum class FieldShape
{
   Scalar,  // Rank 0: single-valued field
   Vector,  // Rank 1: vector-valued field (extent resolved at evaluation time)
   Matrix   // Rank 2: matrix-valued field (extents resolved at evaluation time)
};

struct NodeBase {};
struct CoefficientInputBase {};
struct FieldBase : NodeBase, CoefficientInputBase {};
struct DomainBase : NodeBase {};
struct FormBase  : NodeBase {};

template<typename T>
concept FieldExpr = std::derived_from<std::remove_cvref_t<T>, FieldBase>;

template<typename T>
concept FormExpr  = std::derived_from<std::remove_cvref_t<T>, FormBase>;

template<typename T>
concept DomainExpr = std::derived_from<std::remove_cvref_t<T>, DomainBase>;

template<typename T>
concept CoefficientInput = std::derived_from<std::remove_cvref_t<T>, CoefficientInputBase>;

// Forward declaration of the terms used in weak form expressions to avoid circular dependencies in includes.
template < StaticString Name, FieldShape Shape = FieldShape::Scalar > struct TrialSpace;
template < StaticString Name, FieldShape Shape = FieldShape::Scalar > struct TestSpace;
template < FieldExpr Expr > struct GradientExpr;
template < FieldExpr Expr > struct NegExpr;
template < FieldExpr Expr > struct AverageExpr;
template < FieldExpr Expr > struct JumpExpr;
template < FieldExpr AdvExpr, FieldExpr Expr > struct UpwindExpr;
template < FieldExpr LHS, FieldExpr RHS > struct DotExpr;
template < FieldExpr LHS, FieldExpr RHS > struct InnerExpr;
template < FieldExpr LHS, FieldExpr RHS > struct OuterExpr;
template < FieldExpr LHS, FieldExpr RHS > struct MultFieldExpr;
template < FieldExpr LHS, FieldExpr RHS > struct ProductExpr;
template < FieldExpr Expr > struct TransposeExpr;
template < FieldExpr Head, FieldExpr... Tail > struct SumExpr;
template < DomainExpr Domain, FieldExpr Expr > struct Integrand;
template < typename Map > struct SumFormExpr;
template < typename Key, typename T > struct Entry;
template < typename... Entries > struct StaticMap;
template < StaticString Name, FieldShape Shape, typename Fn, CoefficientInput... Inputs > struct Coefficient;
struct ScaleExpr;
struct Normal;
template < StaticString Name > struct FiniteElementField;

} // namespace gendil

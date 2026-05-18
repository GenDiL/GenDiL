// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/Algebra/staticvector.hpp"

namespace gendil
{

template <typename T, static_vector<T> VecType>
constexpr auto operator*(const T& scalar, const VecType& vec);

template < DomainExpr Domain, FieldExpr Expr >
struct Integrand : FormBase
{
   using domain_type = Domain;
   using expression_type = Expr;
   domain_type domain;
   expression_type expr;

   Integrand(const domain_type & domain_, const expression_type & expr_)
      : domain(domain_), expr(expr_)
   {}

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
      //!FIXME detJ is the volume detJ, needs to be scaled for face detJ.
      return quad_pt_context.weight * quad_pt_context.det_J * expr(kernel_context, weak_form_context, operator_context, element_context, quad_pt_context, fields);
   }
};

template < DomainExpr Domain, FieldExpr Expr >
auto integrate(const Domain & domain, const Expr & expr)
{
   return Integrand<Domain, Expr>(domain, expr);
}

template < DomainExpr Domain, FieldExpr Expr >
std::ostream& operator<<(std::ostream& os, const Integrand<Domain, Expr>& integrand)
{
   return os << "Integral< " << integrand.domain << " >( " << integrand.expr << " )";
}

template < StaticString DomainName >
struct Cells : DomainBase
{
   static constexpr auto name = DomainName;
};

template<class T>
struct is_cell_integrand : std::false_type {};

template<StaticString DomainName, FieldExpr Expr>
struct is_cell_integrand< Integrand<Cells<DomainName>, Expr> > : std::true_type {};

template<class T>
concept CellIntegrand =
   is_cell_integrand<std::remove_cvref_t<T>>::value;

template < StaticString Name >
std::ostream& operator<<(std::ostream& os, const Cells<Name>& domain)
{
   return os << "Cells_" << domain.name.view();
}

template < StaticString DomainName >
struct InteriorFacets : DomainBase
{
   static constexpr auto name = DomainName;
};

template<class T>
struct is_interior_facet_integrand : std::false_type {};

template<StaticString DomainName, FieldExpr Expr>
struct is_interior_facet_integrand< Integrand<InteriorFacets<DomainName>, Expr> >
   : std::true_type {};

template<class T>
concept InteriorFacetIntegrand =
   is_interior_facet_integrand<std::remove_cvref_t<T>>::value;

template < StaticString Name >
std::ostream& operator<<(std::ostream& os, const InteriorFacets<Name>& domain)
{
   return os << "InteriorFacets_" << domain.name.view();
}

template < StaticString DomainName >
struct BoundaryFacets : DomainBase
{
   static constexpr auto name = DomainName;
};

template<class T>
struct is_boundary_facet_integrand : std::false_type {};

template<StaticString DomainName, FieldExpr Expr>
struct is_boundary_facet_integrand< Integrand<BoundaryFacets<DomainName>, Expr> >
   : std::true_type {};

template<class T>
concept BoundaryFacetIntegrand =
   is_boundary_facet_integrand<std::remove_cvref_t<T>>::value;

template < StaticString Name >
std::ostream& operator<<(std::ostream& os, const BoundaryFacets<Name>& domain)
{
   return os << "BoundaryFacets_" << domain.name.view();
}

template < StaticString LHSDomainName, StaticString RHSDomainName >
struct InterfaceFacets : DomainBase
{
   static constexpr auto lhs_name = LHSDomainName;
   static constexpr auto rhs_name = RHSDomainName;
};

template < StaticString LHSDomainName, StaticString RHSDomainName >
std::ostream& operator<<(std::ostream& os, const InterfaceFacets<LHSDomainName, RHSDomainName>& domain)
{
   return os << "InterfaceFacets_" << domain.lhs_name.view() << "_" << domain.rhs_name.view();
}

} // namespace gendil

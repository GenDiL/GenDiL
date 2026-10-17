// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"

namespace gendil
{

// Select cells or facets symbolically by their domain names. The weak-form
// context connects these names to the mesh or partition used for integration.
template < StaticString DomainName >
struct Cells : DomainBase
{
   static constexpr auto name = DomainName;
};

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

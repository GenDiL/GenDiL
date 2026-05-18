// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"

namespace gendil {

template<class CellView>
struct ElementContext
{
  GlobalIndex element_index;
  CellView cell;

  GENDIL_HOST_DEVICE
  GlobalIndex ElementIndex() const { return element_index; }

  GENDIL_HOST_DEVICE
  const CellView& Cell() const { return cell; }
};

// Factory: domain fetched from weak form context by compile-time name
template<typename WeakFormContext, typename Integrand>
GENDIL_HOST_DEVICE
auto MakeElementContext(const WeakFormContext & wf_ctx, const Integrand & integrand, const GlobalIndex & element_index)
{
  constexpr auto DomainName = integrand.domain.name;
  // Domain is whatever you stored under DomainKey<DomainName> (likely a mesh view)
  const auto& mesh = wf_ctx.template domain<DomainName>();
  return ElementContext<decltype(mesh.GetCell(element_index))>{ element_index, mesh.GetCell(element_index) };
}

template<typename FESpace>
GENDIL_HOST_DEVICE
auto MakeElementContext(const GlobalIndex & element_index, const FESpace & fe_space)
{
  return ElementContext<decltype(fe_space.GetCell(element_index))>{ element_index, fe_space.GetCell(element_index) };
}

} // namespace gendil
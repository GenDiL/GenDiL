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

template<typename FESpace>
GENDIL_HOST_DEVICE
auto MakeElementContext(const GlobalIndex & element_index, const FESpace & fe_space)
{
  return ElementContext<decltype(fe_space.GetCell(element_index))>{ element_index, fe_space.GetCell(element_index) };
}

} // namespace gendil

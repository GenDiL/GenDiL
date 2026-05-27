// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"

namespace gendil {

template<typename FaceInfo, typename PlusCellView>
struct FacetContext : FaceInfo
{
   PlusCellView plus_cell;

   GENDIL_HOST_DEVICE
   FacetContext(const FaceInfo& face_info, const PlusCellView& plus_cell_)
      : FaceInfo(face_info), plus_cell(plus_cell_)
   {}
};

// Factory: domain fetched from weak form context by compile-time name
template<typename WeakFormContext, typename Integrand, typename FaceInfo>
GENDIL_HOST_DEVICE
auto MakeInteriorFacetContext(const WeakFormContext & wf_ctx, const Integrand & /*integrand*/, const FaceInfo & face_info)
{
   if constexpr (requires_plus_side_jacobian_v<Integrand>)
   {
      constexpr auto DomainName = Integrand::domain_type::name;
      const auto& mesh = wf_ctx.template domain<DomainName>();
      auto plus_cell = mesh.GetCell(face_info.PlusSide().GetCellIndex());
      ApplyOrientationToCell(face_info.PlusSide().GetOrientation(), plus_cell);
      return FacetContext{ face_info, plus_cell };
   }
   else
   {
      return face_info;
   }
}

} // namespace gendil

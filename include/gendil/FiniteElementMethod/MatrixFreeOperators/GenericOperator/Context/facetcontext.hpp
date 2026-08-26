// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"
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

template<class Channels>
inline constexpr bool global_interior_channels_require_plus_side_jacobian_v =
   std::remove_cvref_t<Channels>::template contains<GradientPlusChannel>();

template<class Integrand>
inline constexpr bool local_interior_context_requires_plus_side_jacobian_v =
   requires_plus_side_jacobian_v<Integrand>;

template<class Integrand, class Channels>
inline constexpr bool global_interior_context_requires_plus_side_jacobian_v =
   requires_plus_side_jacobian_v<Integrand> ||
   global_interior_channels_require_plus_side_jacobian_v<Channels>;

template<typename WeakFormContext, typename Integrand, typename FaceInfo>
GENDIL_HOST_DEVICE
auto MakeLocalInteriorFacetContext(
   const WeakFormContext& wf_ctx,
   const Integrand& integrand,
   const FaceInfo& face_info)
{
   if constexpr (local_interior_context_requires_plus_side_jacobian_v<Integrand>)
   {
      const auto& mesh =
         GetCellIntegrationDomainMesh(integrand, wf_ctx);

      auto plus_cell =
         mesh.GetCell(face_info.PlusSide().GetCellIndex());

      ApplyOrientationToCell(
         face_info.PlusSide().GetOrientation(),
         plus_cell);

      return FacetContext{ face_info, plus_cell };
   }
   else
   {
      return face_info;
   }
}

template<
   typename FaceDomain,
   typename Integrand,
   typename Channels,
   typename FaceInfo>
GENDIL_HOST_DEVICE
auto MakeGlobalInteriorFacetContext(
   const FaceDomain& face_domain,
   const Integrand& /*integrand*/,
   const Channels& /*channels*/,
   const FaceInfo& face_info)
{
   if constexpr (
      global_interior_context_requires_plus_side_jacobian_v<
         Integrand,
         Channels>)
   {
      static_assert(
         requires (const FaceDomain& domain)
         {
            domain.GetPlusCellMesh();
         },
         "Global interior facet context requires a face execution space "
         "that exposes GetPlusCellMesh() when plus-side geometry "
         "is needed.");

      const auto& plus_mesh = face_domain.GetPlusCellMesh();

      auto plus_cell =
         plus_mesh.GetCell(face_info.PlusSide().GetCellIndex());

      ApplyOrientationToCell(
         face_info.PlusSide().GetOrientation(),
         plus_cell);

      return FacetContext{ face_info, plus_cell };
   }
   else
   {
      return face_info;
   }
}

} // namespace gendil

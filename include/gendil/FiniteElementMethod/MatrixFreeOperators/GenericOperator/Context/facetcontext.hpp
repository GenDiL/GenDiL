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

template<class Space>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetFacetContextCellDomainSpace(
   const CellIntegrationDomain<Space>& domain)
{
   using SpaceType = std::remove_cvref_t<Space>;
   static_assert(
      is_cell_finite_element_space_v<SpaceType>,
      "Facet context construction requires a selected homogeneous "
      "CellIntegrationDomain<Space>. Mixed or raw domains must be normalized "
      "and restricted before context construction.");
   return (domain.space);
}

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
   const Integrand& /*integrand*/,
   const FaceInfo& face_info)
{
   if constexpr (local_interior_context_requires_plus_side_jacobian_v<Integrand>)
   {
      constexpr auto DomainName = Integrand::domain_type::name;
      const auto& mesh =
         GetFacetContextCellDomainSpace(wf_ctx.template domain<DomainName>());

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
            domain.GetPlusCellFiniteElementSpace();
         },
         "Global interior facet context requires a face execution space "
         "that exposes GetPlusCellFiniteElementSpace() when plus-side geometry "
         "is needed.");

      const auto& plus_space = face_domain.GetPlusCellFiniteElementSpace();

      auto plus_cell =
         plus_space.GetCell(face_info.PlusSide().GetCellIndex());

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

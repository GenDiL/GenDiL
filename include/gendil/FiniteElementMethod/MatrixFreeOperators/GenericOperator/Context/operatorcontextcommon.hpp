// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticmap.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globalfacefieldbinding.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/doftoquad.hpp"

namespace gendil
{

template<class IntegrationRule, class DomainView>
constexpr auto MakeMeshQuadData(const DomainView& /*dom*/)
{
   using Domain = std::remove_cvref_t<DomainView>;
   using QD = typename Domain::cell_type::template QuadData<IntegrationRule>;
   return QD{};
}

template<class IntegrationRule, class SpaceView>
constexpr auto MakeFiniteElementQuadData(const SpaceView& space)
{
   using Space = std::remove_cvref_t<SpaceView>;
   static_assert(
      !is_boundary_face_field_binding_v<Space> &&
      !is_interior_face_field_binding_v<Space>,
      "MakeFiniteElementQuadData builds cell/volume finite element qdata. "
      "Face field bindings must use the global facet qdata builder "
      "so minus/plus sides are represented explicitly.");
   (void)space;
   using FE    = typename Space::finite_element_type;
   using Shape = typename FE::shape_functions;
   return MakeDofToQuad<Shape, IntegrationRule>();
}

template<class IR, class DomainEntry>
constexpr auto domain_entry_to_mesh_qd_tuple(const DomainEntry& e)
{
   using Key = typename DomainEntry::key_type;
   auto qd   = MakeMeshQuadData<IR>(e.value);
   using QD  = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

template<class IR, class FEFieldEntry>
constexpr auto fe_field_entry_to_elem_qd_tuple(const FEFieldEntry& e)
{
   using Key       = typename FEFieldEntry::key_type;
   const auto& fev = e.value;
   auto qd         = MakeFiniteElementQuadData<IR>(fev.space);
   using QD        = std::remove_cvref_t<decltype(qd)>;
   return std::tuple{ Entry<Key, QD>{ static_cast<QD>(qd) } };
}

} // namespace gendil

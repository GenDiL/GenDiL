// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"

namespace gendil {

template<class MinusView, class PlusView>
GENDIL_HOST_DEVICE
auto SwapGlobalFaceInfo(const GlobalFaceInfo<MinusView, PlusView>& face_info)
{
   return GlobalFaceInfo<PlusView, MinusView>{ face_info.PlusSide(), face_info.MinusSide() };
}

} // namespace gendil

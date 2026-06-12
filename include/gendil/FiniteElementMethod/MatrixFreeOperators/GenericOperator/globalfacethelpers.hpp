// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/facereaddofspolicy.hpp"

namespace gendil {

template<class MinusView, class PlusView>
GENDIL_HOST_DEVICE
auto SwapGlobalFaceInfo(const GlobalFaceInfo<MinusView, PlusView>& face_info)
{
   return GlobalFaceInfo<PlusView, MinusView>{ face_info.PlusSide(), face_info.MinusSide() };
}

template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FiniteElementSpace >
struct GlobalGenericFaceRequiredSharedMemory
{
private:
   using ShapeFunctions =
      typename FiniteElementSpace::finite_element_type::shape_functions;
   using FaceReadPolicy =
      face_read_dofs_policy_t< KernelConfiguration >;
   using FaceWritePolicy =
      face_write_dofs_policy_t< KernelConfiguration >;

   static constexpr bool is_threaded =
      is_threaded_v< KernelConfiguration >;
   static constexpr bool scalar_dofs =
      !is_vector_shape_functions_v< ShapeFunctions >;
   static constexpr bool direct_global_read =
      std::is_same_v<
         FaceReadPolicy,
         DirectGlobalFaceReadDofsPolicy >;
   static constexpr bool direct_global_write =
      scalar_dofs &&
      std::is_same_v<
         FaceWritePolicy,
         DirectGlobalFaceWriteDofsPolicy >;

   static constexpr size_t read_requirement =
      !is_threaded || direct_global_read
         ? 0
         : FiniteElementSpace::finite_element_type::GetNumDofs();
   static constexpr size_t write_requirement =
      !is_threaded || direct_global_write
         ? 0
         : FiniteElementSpace::finite_element_type::GetNumDofs();
   static constexpr size_t interpolation_and_test_requirement =
      required_shared_memory_v< KernelConfiguration, IntegrationRule >;

public:
   static constexpr size_t value =
      Max(
         read_requirement,
         write_requirement,
         interpolation_and_test_requirement );
};

template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FiniteElementSpace >
inline constexpr size_t global_generic_face_required_shared_memory_v =
   GlobalGenericFaceRequiredSharedMemory<
      KernelConfiguration,
      IntegrationRule,
      FiniteElementSpace >::value;

} // namespace gendil

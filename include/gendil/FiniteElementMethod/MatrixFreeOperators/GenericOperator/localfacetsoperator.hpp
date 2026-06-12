// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/cellintegrand.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/localinteriorfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/localboundaryfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"

namespace gendil {

template<
  class KernelPolicy,
  class WeakForm,
  class WeakFormContext,
  class OperatorContext,
  class DofsInView,
  class DofsOutView>
void LocalFacetsGenericExplicitOperator(
  const WeakForm& weak_form,
  const WeakFormContext& wf_ctx,
  const OperatorContext& op_ctx,
  const DofsInView& dofs_in,
  DofsOutView& dofs_out)
{
   GENDIL_REQUIRE_UNBATCHED_OPERATOR( KernelPolicy );

   using I = std::remove_cvref_t<WeakForm>;

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TestName  = requirements<I>::test_name;

   static_assert(TrialName != StaticString{"Error"}, "LocalFacetsGenericExplicitOperator: missing TrialSpace in weak_form.");
   static_assert(TestName  != StaticString{"Error"}, "LocalFacetsGenericExplicitOperator: missing TestSpace in weak_form.");

   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   const auto& test_space  = wf_ctx.template fe_field<TestName>().space;
   using IntegrationRule = decltype(op_ctx.integration_rule());

   constexpr size_t required_shared_mem =
      local_generic_cell_required_shared_memory_v<
         KernelPolicy,
         IntegrationRule,
         std::remove_cvref_t<decltype(trial_space)>,
         WeakForm,
         DofsInView,
         DofsOutView>;

   mesh::CellIterator<KernelPolicy>(
      trial_space,
      [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
      {
         (void)wf_ctx;
         (void)op_ctx;
         (void)weak_form;

         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];
         KernelContext<KernelPolicy, required_shared_mem> kernel(_shared_mem);

         auto u_elem = ReadDofs(kernel, trial_space, element_index, dofs_in);

         using VType = decltype(ReadDofs(kernel, test_space, element_index, dofs_out));
         VType v_elem{};

         ElementContext element_context{ element_index, trial_space.GetCell( element_index ) };

         if constexpr (has_cell_contributions_v<WeakForm>)
         {
            GenericCellIntegrandOperator(
               kernel,
               wf_ctx,
               op_ctx,
               element_context,
               weak_form,
               u_elem,
               v_elem
            );
         }

         if constexpr (has_interior_facet_contributions_v<WeakForm>)
         {
            LocalInteriorFacetOperator(
               kernel,
               wf_ctx,
               op_ctx,
               element_context,
               trial_space,
               weak_form,
               dofs_in,
               u_elem,
               v_elem
            );
         }

         if constexpr (has_boundary_facet_contributions_v<WeakForm>)
         {
            LocalBoundaryFacetOperator(
               kernel,
               wf_ctx,
               op_ctx,
               element_context,
               trial_space,
               weak_form,
               u_elem,
               v_elem
            );
         }

         WriteDofs(kernel, test_space, element_index, v_elem, dofs_out);
      }
   );
}

} // namespace gendil

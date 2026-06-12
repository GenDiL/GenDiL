// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/cellintegrand.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/globalinteriorfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/globalboundaryfacet.hpp"

namespace gendil {

template<
  class KernelPolicy,
  class WeakForm,
  class WeakFormContext,
  class OperatorContext,
  class DofsInView,
  class DofsOutView>
void GlobalFacetsGenericExplicitOperator(
  const WeakForm& weak_form,
  const WeakFormContext& wf_ctx,
  const OperatorContext& op_ctx,
  const DofsInView& dofs_in,
  DofsOutView& dofs_out)
{
   using I = std::remove_cvref_t<WeakForm>;

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TestName  = requirements<I>::test_name;

   static_assert(TrialName != StaticString{"Error"}, "GlobalFacetsGenericExplicitOperator: missing TrialSpace in weak_form.");
   static_assert(TestName  != StaticString{"Error"}, "GlobalFacetsGenericExplicitOperator: missing TestSpace in weak_form.");
   static_assert(
      global_facet_domain_requirements_satisfied_v<WeakForm, WeakFormContext>,
      "GlobalFacetsGenericExplicitOperator: hybrid local/global facet "
      "execution is not supported. When any global face domain is present, "
      "every InteriorFacets<Name> term must have MakeInteriorFaceDomain<Name>(...) "
      "and every BoundaryFacets<Name> term must have "
      "MakeBoundaryFaceDomain<Name>(...).");

   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   const auto& test_space  = wf_ctx.template fe_field<TestName>().space;
   using IntegrationRule = decltype(op_ctx.integration_rule());

   constexpr size_t required_shared_mem =
      required_shared_memory_v<KernelPolicy, IntegrationRule>;

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

         WriteDofs(kernel, test_space, element_index, v_elem, dofs_out);
      }
   );

   if constexpr (has_interior_facet_contributions_v<WeakForm>)
   {
      GenericGlobalInteriorFacetOperator<KernelPolicy>(
         wf_ctx,
         op_ctx,
         weak_form,
         dofs_in,
         dofs_out);
   }

   if constexpr (has_boundary_facet_contributions_v<WeakForm>)
   {
      GenericGlobalBoundaryFacetOperator<KernelPolicy>(
         wf_ctx,
         op_ctx,
         weak_form,
         dofs_in,
         dofs_out);
   }
}

} // namespace gendil

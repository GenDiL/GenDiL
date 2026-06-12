// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/globalfacethelpers.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/localinteriorfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"

namespace gendil {

template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename TrialSpace,
   typename TestSpace,
   typename FaceInfo,
   typename Integrand,
   typename DofsInView,
   typename DofsOutView>
GENDIL_HOST_DEVICE
void GenericGlobalInteriorFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const TrialSpace& trial_space,
   const TestSpace& test_space,
   const FaceInfo& face_info,
   const Integrand& integrand,
   const DofsInView& dofs_in,
   DofsOutView& dofs_out)
{
   if constexpr (InteriorFacetIntegrand<Integrand>)
   {
      auto u_minus = ReadDofs(
         kernel_context,
         trial_space,
         face_info.MinusSide(),
         dofs_in);
      auto u_plus = ReadDofs(
         kernel_context,
         trial_space,
         face_info.PlusSide(),
         dofs_in);

      using MinusOut = decltype(ReadDofs(
         kernel_context,
         test_space,
         face_info.MinusSide(),
         dofs_out));
      MinusOut v_minus{};

      ElementContext minus_element_context{
         face_info.MinusSide().GetCellIndex(),
         trial_space.GetCell(face_info.MinusSide().GetCellIndex())
      };

      // Compatibility adapter: evaluate the canonical minus row with the
      // local/current-row facet integrand semantics.
      LocalInteriorFacetIntegrandOperator(
         kernel_context,
         weak_form_context,
         operator_context,
         minus_element_context,
         face_info,
         integrand,
         u_minus,
         u_plus,
         v_minus);

      WriteAddDofs(
         kernel_context,
         test_space,
         face_info.MinusSide(),
         v_minus,
         dofs_out);

      auto swapped_face_info = SwapGlobalFaceInfo(face_info);

      using PlusOut = decltype(ReadDofs(
         kernel_context,
         test_space,
         swapped_face_info.MinusSide(),
         dofs_out));
      PlusOut v_plus{};

      ElementContext plus_element_context{
         swapped_face_info.MinusSide().GetCellIndex(),
         trial_space.GetCell(swapped_face_info.MinusSide().GetCellIndex())
      };

      // Compatibility adapter: swap sides so the canonical plus Cell becomes
      // the local/current row evaluated by the local facet integrand.
      LocalInteriorFacetIntegrandOperator(
         kernel_context,
         weak_form_context,
         operator_context,
         plus_element_context,
         swapped_face_info,
         integrand,
         u_plus,
         u_minus,
         v_plus);

      WriteAddDofs(
         kernel_context,
         test_space,
         swapped_face_info.MinusSide(),
         v_plus,
         dofs_out);
   }
}

template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename TrialSpace,
   typename TestSpace,
   typename FaceInfo,
   typename Map,
   typename DofsInView,
   typename DofsOutView>
GENDIL_HOST_DEVICE
void GenericGlobalInteriorFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const TrialSpace& trial_space,
   const TestSpace& test_space,
   const FaceInfo& face_info,
   const SumFormExpr<Map>& sum_integrand,
   const DofsInView& dofs_in,
   DofsOutView& dofs_out)
{
   std::apply(
      [&] (auto const & ... entries)
      {
         (
            [&]
            {
               if constexpr (has_interior_facet_contributions_v<decltype(entries.value)>)
               {
                  GenericGlobalInteriorFacetIntegrandOperator(
                     kernel_context,
                     weak_form_context,
                     operator_context,
                     trial_space,
                     test_space,
                     face_info,
                     entries.value,
                     dofs_in,
                     dofs_out);
               }
            }(),
            ...
         );
      },
      sum_integrand.map.entries
   );
}

template<
   typename KernelPolicy,
   typename WeakFormContext,
   typename OperatorContext,
   typename FaceDomain,
   typename Integrand,
   typename DofsInView,
   typename DofsOutView>
void GenericGlobalInteriorFacetDomainOperator(
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const FaceDomain& face_domain,
   const Integrand& integrand,
   const DofsInView& dofs_in,
   DofsOutView& dofs_out)
{
   constexpr auto TrialName = requirements<Integrand>::trial_name;
   constexpr auto TestName  = requirements<Integrand>::test_name;

   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   const auto& test_space  = wf_ctx.template fe_field<TestName>().space;
   const auto& face_space = face_domain.finite_element_space;

   static_assert(
      std::is_same_v<
         std::remove_cvref_t<decltype(face_space)>,
         std::remove_cvref_t<decltype(trial_space)>>,
      "Global interior face domains currently require the active trial finite "
      "element space to match the face-domain finite element space.");
   static_assert(
      std::is_same_v<
         std::remove_cvref_t<decltype(face_space)>,
         std::remove_cvref_t<decltype(test_space)>>,
      "Global interior face domains currently require the active test finite "
      "element space to match the face-domain finite element space.");

   using IntegrationRule = decltype(op_ctx.integration_rule());
   constexpr size_t required_shared_mem =
      global_generic_interior_facet_required_shared_memory_v<
         KernelPolicy,
         IntegrationRule,
         std::remove_cvref_t<decltype(face_space)>,
         Integrand,
         DofsInView,
         DofsOutView>;
   constexpr size_t shared_memory_block_size =
      KernelContext<
         KernelPolicy,
         required_shared_mem >::shared_memory_block_size;

   mesh::ForEachFaceMesh(
      face_domain.face_mesh,
      [&] (const auto& face_mesh) mutable
      {
         mesh::GlobalFaceIterator<KernelPolicy>(
            face_mesh,
            [=] GENDIL_HOST_DEVICE (GlobalIndex face_index) mutable
            {
               (void)wf_ctx;
               (void)op_ctx;
               (void)integrand;

               GENDIL_SHARED Real _shared_mem[
                  shared_memory_block_size == 0
                     ? 1
                     : shared_memory_block_size ];
               KernelContext<KernelPolicy, required_shared_mem> kernel(_shared_mem);

               const auto face_info = face_mesh.GetGlobalFaceInfo(face_index);

               GenericGlobalInteriorFacetIntegrandOperator(
                  kernel,
                  wf_ctx,
                  op_ctx,
                  trial_space,
                  test_space,
                  face_info,
                  integrand,
                  dofs_in,
                  dofs_out);
            }
         );
      }
   );
}

template<
   typename KernelPolicy,
   typename WeakFormContext,
   typename OperatorContext,
   typename Integrand,
   typename DofsInView,
   typename DofsOutView>
void GenericGlobalInteriorFacetOperator(
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const Integrand& integrand,
   const DofsInView& dofs_in,
   DofsOutView& dofs_out)
{
   if constexpr (InteriorFacetIntegrand<Integrand>)
   {
      constexpr auto DomainName = Integrand::domain_type::name;
      static_assert(
         std::remove_cvref_t<WeakFormContext>::template
            has_interior_face_domain<DomainName>(),
         "Generic global facet operator cannot evaluate InteriorFacets<Name> "
         "without MakeInteriorFaceDomain<Name>(...). Hybrid local/global "
         "facet execution is not supported.");
      const auto& face_domain =
         wf_ctx.template interior_face_domain<DomainName>();
      GenericGlobalInteriorFacetDomainOperator<KernelPolicy>(
         wf_ctx,
         op_ctx,
         face_domain,
         integrand,
         dofs_in,
         dofs_out);
   }
}

template<
   typename KernelPolicy,
   typename WeakFormContext,
   typename OperatorContext,
   typename Map,
   typename DofsInView,
   typename DofsOutView>
void GenericGlobalInteriorFacetOperator(
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const SumFormExpr<Map>& sum_integrand,
   const DofsInView& dofs_in,
   DofsOutView& dofs_out)
{
   std::apply(
      [&] (auto const & ... entries)
      {
         (
            [&]
            {
               if constexpr (has_interior_facet_contributions_v<decltype(entries.value)>)
               {
                  GenericGlobalInteriorFacetOperator<KernelPolicy>(
                     wf_ctx,
                     op_ctx,
                     entries.value,
                     dofs_in,
                     dofs_out);
               }
            }(),
            ...
         );
      },
      sum_integrand.map.entries
   );
}

} // namespace gendil

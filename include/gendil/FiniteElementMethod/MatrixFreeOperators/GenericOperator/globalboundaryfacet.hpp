// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/globaloperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/localboundaryfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/restrictedweakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"

namespace gendil {

template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename FaceDomain,
   typename TrialSpace,
   typename TestSpace,
   typename FaceInfo,
   typename Integrand,
   typename DofsInView,
   typename DofsOutView>
GENDIL_HOST_DEVICE
void GenericGlobalBoundaryFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const FaceDomain& face_domain,
   const TrialSpace& trial_space,
   const TestSpace& test_space,
   const FaceInfo& face_info,
   const Integrand& integrand,
   const DofsInView& dofs_in,
   DofsOutView& dofs_out)
{
   if constexpr (BoundaryFacetIntegrand<Integrand>)
   {
      auto u_elem = ReadDofs(
         kernel_context,
         trial_space,
         face_info.MinusSide(),
         dofs_in);

      using DofsOut = decltype(ReadDofs(
         kernel_context,
         test_space,
         face_info.MinusSide(),
         dofs_out));
      DofsOut v_elem{};

      ElementContext element_context{
         face_info.MinusSide().GetCellIndex(),
         face_domain.GetCellFiniteElementSpace().GetCell(
            face_info.MinusSide().GetCellIndex())
      };

      LocalBoundaryFacetIntegrandOperator(
         kernel_context,
         weak_form_context,
         operator_context,
         element_context,
         face_info,
         integrand,
         u_elem,
         v_elem);

      WriteAddDofs(
         kernel_context,
         test_space,
         face_info.MinusSide(),
         v_elem,
         dofs_out);
   }
}

template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename FaceDomain,
   typename TrialSpace,
   typename TestSpace,
   typename FaceInfo,
   typename Map,
   typename DofsInView,
   typename DofsOutView>
GENDIL_HOST_DEVICE
void GenericGlobalBoundaryFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const FaceDomain& face_domain,
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
               if constexpr (has_boundary_facet_contributions_v<decltype(entries.value)>)
               {
                  GenericGlobalBoundaryFacetIntegrandOperator(
                     kernel_context,
                     weak_form_context,
                     operator_context,
                     face_domain,
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
   typename FaceDomain,
   typename Integrand,
   typename IntegrationRule,
   typename DofsInView,
   typename DofsOutView>
void GenericGlobalBoundaryFacetDomainOperator(
   const WeakFormContext& wf_ctx,
   const FaceDomain& face_domain,
   const Integrand& integrand,
   const IntegrationRule& integration_rule,
   const DofsInView& dofs_in,
   DofsOutView& dofs_out)
{
   constexpr auto TrialName = requirements<Integrand>::trial_name;
   constexpr auto TestName  = requirements<Integrand>::test_name;

   const auto& trial_binding = wf_ctx.template fe_field<TrialName>().space;
   const auto& test_binding  = wf_ctx.template fe_field<TestName>().space;
   const auto& trial_space = trial_binding.GetMinusFiniteElementSpace();
   const auto& test_space  = test_binding.GetMinusFiniteElementSpace();

   constexpr size_t required_shared_mem =
      Max(
         generic_operator_integrand_required_shared_memory_v<
            KernelPolicy,
            std::remove_cvref_t<IntegrationRule>>,
         generic_operator_face_read_scratch_requirement_v<
            KernelPolicy,
            std::remove_cvref_t<decltype(trial_space)>,
            DofsInView>,
         generic_operator_face_write_scratch_requirement_v<
            KernelPolicy,
            std::remove_cvref_t<decltype(test_space)>,
            DofsOutView>);
   constexpr size_t shared_memory_block_size =
      KernelContext<
         KernelPolicy,
         required_shared_mem >::shared_memory_block_size;

   const auto& face_mesh = face_domain.GetFaceMesh();
   auto facet_op_ctx =
      MakeFacetOperatorContext(
         wf_ctx,
         integration_rule,
         face_domain);

   mesh::GlobalFaceIterator<KernelPolicy>(
      face_mesh,
      [=] GENDIL_HOST_DEVICE (GlobalIndex face_index) mutable
      {
         (void)wf_ctx;
         (void)facet_op_ctx;
         (void)integrand;

         GENDIL_SHARED Real _shared_mem[
            shared_memory_block_size == 0
               ? 1
               : shared_memory_block_size ];
         KernelContext<KernelPolicy, required_shared_mem> kernel(_shared_mem);

         const auto face_info = face_mesh.GetGlobalFaceInfo(face_index);

         GenericGlobalBoundaryFacetIntegrandOperator(
            kernel,
            wf_ctx,
            facet_op_ctx,
            face_domain,
            trial_space,
            test_space,
            face_info,
            integrand,
            dofs_in,
            dofs_out);
      }
   );
}

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   StaticString DomainName,
   size_t FaceI,
   class FacePart,
   class CellSpace,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalBoundaryFaceDomainOperator(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   BoundaryFacets<DomainName> domain_tag,
   const BoundaryFaceExecutionBatch<
      DomainName,
      FaceI,
      FacePart,
      CellSpace>& batch,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   auto batch_ctx =
      MakeRestrictedWeakFormContext<TrialName, TestName>(
         wf_ctx,
         domain_tag,
         batch);

   const auto& trial_binding =
      batch_ctx.template fe_field<TrialName>().space;
   const auto& test_binding =
      batch_ctx.template fe_field<TestName>().space;
   const auto& trial_space = trial_binding.GetMinusFiniteElementSpace();
   const auto& test_space = test_binding.GetMinusFiniteElementSpace();

   auto dofs_in = MakeReadOnlyElementTensorView<KernelPolicy>(
      trial_space,
      dofs_vector_in);
   auto dofs_out = MakeReadWriteElementTensorView<KernelPolicy>(
      test_space,
      dofs_vector_out);

   GenericGlobalBoundaryFacetDomainOperator<KernelPolicy>(
      batch_ctx,
      batch,
      weak_form,
      integration_rule,
      dofs_in,
      dofs_out);
}

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/elementcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/globaloperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/restrictedweakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"

namespace gendil {

template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename FaceDomain,
   typename TrialMinusSpace,
   typename TrialPlusSpace,
   typename TestMinusSpace,
   typename TestPlusSpace,
   typename FaceInfo,
   typename Integrand,
   typename DofsInMinusView,
   typename DofsInPlusView,
   typename DofsOutMinusView,
   typename DofsOutPlusView>
GENDIL_HOST_DEVICE
void GenericCanonicalGlobalInteriorChannelOperator(
   KernelContext& kernel_context,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const FaceDomain& face_domain,
   const TrialMinusSpace& trial_minus_space,
   const TrialPlusSpace& trial_plus_space,
   const TestMinusSpace& test_minus_space,
   const TestPlusSpace& test_plus_space,
   const FaceInfo& face_info,
   const Integrand& integrand,
   const DofsInMinusView& dofs_in_minus,
   const DofsInPlusView& dofs_in_plus,
   DofsOutMinusView& dofs_out_minus,
   DofsOutPlusView& dofs_out_plus)
{
   constexpr auto TestName  = requirements<Integrand>::test_name;

   auto u_minus = ReadDofs(
      kernel_context,
      trial_minus_space,
      face_info.MinusSide(),
      dofs_in_minus);
   auto u_plus = ReadDofs(
      kernel_context,
      trial_plus_space,
      face_info.PlusSide(),
      dofs_in_plus);

   const auto& test_qd =
      op_ctx.template finite_element_facet_quad_data<TestName>();

   auto fields = InterpolateFields(
      kernel_context,
      wf_ctx,
      op_ctx,
      face_info,
      integrand,
      u_minus,
      u_plus);
   auto channels =
      LowerGlobalInteriorFacetIntegrandToPullbackChannels(integrand);

   auto face_integration_rule =
      GetFaceIntegrationRule(
         face_info.MinusSide(),
         op_ctx.facet_integration_rules());

   using Channels = std::remove_cvref_t<decltype(channels)>;

   auto Du_minus =
      MakeQuadraturePointContainerForSpace<
         Channels::template contains<ValueMinusChannel>(),
         Channels::template contains<GradientMinusChannel>()>(
         kernel_context,
         face_integration_rule,
         test_minus_space);
   auto Du_plus =
      MakeQuadraturePointContainerForSpace<
         Channels::template contains<ValuePlusChannel>(),
         Channels::template contains<GradientPlusChannel>()>(
         kernel_context,
         face_integration_rule,
         test_plus_space);

   ElementContext minus_element_context{
      face_info.MinusSide().GetCellIndex(),
      face_domain.GetMinusCellFiniteElementSpace().GetCell(
         face_info.MinusSide().GetCellIndex())
   };
   auto face_context =
      MakeGlobalInteriorFacetContext(
         face_domain,
         integrand,
         channels,
         face_info);

   QuadraturePointLoop(
      kernel_context,
      face_integration_rule,
      [&] (const auto& quad_index)
      {
         auto facet_quad_pt_context =
            MakeGlobalInteriorFacetQuadraturePointContext(
               kernel_context,
               wf_ctx,
               op_ctx,
               minus_element_context,
               face_context,
               integrand,
               channels,
               quad_index);

         WriteGlobalInteriorFacetChannelContributions(
            kernel_context,
            wf_ctx,
            op_ctx,
            minus_element_context,
            facet_quad_pt_context,
            fields,
            channels,
            quad_index,
            Du_minus,
            Du_plus);
      });

   using MinusOut = decltype(ReadDofs(
      kernel_context,
      test_minus_space,
      face_info.MinusSide(),
      dofs_out_minus));
   using PlusOut = decltype(ReadDofs(
      kernel_context,
      test_plus_space,
      face_info.PlusSide(),
      dofs_out_plus));
   MinusOut v_minus{};
   PlusOut v_plus{};

   ApplyAddGlobalInteriorFacetTestFunctionsForSide(
      kernel_context,
      face_info.MinusSide(),
      test_qd.MinusSide(),
      Du_minus,
      v_minus);
   ApplyAddGlobalInteriorFacetTestFunctionsForSide(
      kernel_context,
      face_info.PlusSide(),
      test_qd.PlusSide(),
      Du_plus,
      v_plus);

   WriteAddDofs(
      kernel_context,
      test_minus_space,
      face_info.MinusSide(),
      v_minus,
      dofs_out_minus);
   WriteAddDofs(
      kernel_context,
      test_plus_space,
      face_info.PlusSide(),
      v_plus,
      dofs_out_plus);
}

template<
   typename KernelPolicy,
   typename WeakFormContext,
   typename FaceDomain,
   typename Integrand,
   typename IntegrationRule,
   typename DofsInMinusView,
   typename DofsInPlusView,
   typename DofsOutMinusView,
   typename DofsOutPlusView>
void GenericCanonicalGlobalInteriorFacetDomainOperator(
   const WeakFormContext& wf_ctx,
   const FaceDomain& face_domain,
   const Integrand& integrand,
   const IntegrationRule& integration_rule,
   const DofsInMinusView& dofs_in_minus,
   const DofsInPlusView& dofs_in_plus,
   DofsOutMinusView& dofs_out_minus,
   DofsOutPlusView& dofs_out_plus)
{
   constexpr auto TrialName = requirements<Integrand>::trial_name;
   constexpr auto TestName  = requirements<Integrand>::test_name;

   const auto& trial_binding =
      wf_ctx.template fe_field<TrialName>().space;
   const auto& test_binding =
      wf_ctx.template fe_field<TestName>().space;
   const auto& trial_minus_space =
      trial_binding.GetMinusFiniteElementSpace();
   const auto& trial_plus_space =
      trial_binding.GetPlusFiniteElementSpace();
   const auto& test_minus_space =
      test_binding.GetMinusFiniteElementSpace();
   const auto& test_plus_space =
      test_binding.GetPlusFiniteElementSpace();

   constexpr size_t required_shared_mem =
      two_space_global_interior_required_shared_memory_v<
         KernelPolicy,
         std::remove_cvref_t<IntegrationRule>,
         std::remove_cvref_t<decltype(trial_minus_space)>,
         std::remove_cvref_t<decltype(trial_plus_space)>,
         std::remove_cvref_t<decltype(test_minus_space)>,
         std::remove_cvref_t<decltype(test_plus_space)>,
         Integrand,
         DofsInMinusView,
         DofsInPlusView,
         DofsOutMinusView,
         DofsOutPlusView>;
   constexpr size_t shared_memory_block_size =
      KernelContext<
         KernelPolicy,
         required_shared_mem>::shared_memory_block_size;

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

         GenericCanonicalGlobalInteriorChannelOperator(
            kernel,
            wf_ctx,
            facet_op_ctx,
            face_domain,
            trial_minus_space,
            trial_plus_space,
            test_minus_space,
            test_plus_space,
            face_info,
            integrand,
            dofs_in_minus,
            dofs_in_plus,
            dofs_out_minus,
            dofs_out_plus);
      });
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
   class MinusCellSpace,
   class PlusCellSpace,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalInteriorFaceDomainOperator(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   InteriorFacets<DomainName> domain_tag,
   const InteriorFaceExecutionBatch<
      DomainName,
      FaceI,
      FacePart,
      MinusCellSpace,
      PlusCellSpace>& batch,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   auto batch_ctx =
      MakeRestrictedWeakFormContext<TrialName, TestName>(
         wf_ctx,
         domain_tag,
         batch);

   static_assert(
      !has_unqualified_interior_test_trace_v<WeakForm>,
      "Interior facet test dependencies must use explicit trace syntax. "
      "Use minus(v), plus(v), jump(v), average(v), minus(grad(v)), "
      "plus(grad(v)), jump(grad(v)), or average(grad(v)); unqualified v or "
      "grad(v) is invalid on interior facets.");
   static_assert(
      !has_unqualified_side_dependent_inputs_v<WeakForm>,
      "Interior facet trial/coefficient finite-element dependencies must use "
      "explicit trace syntax. Use minus(expr), plus(expr), jump(expr), "
      "average(expr), or a trace-aware operator such as upwind(...).");

   const auto& trial_binding =
      batch_ctx.template fe_field<TrialName>().space;
   const auto& test_binding =
      batch_ctx.template fe_field<TestName>().space;
   const auto& trial_minus_space =
      trial_binding.GetMinusFiniteElementSpace();
   const auto& trial_plus_space =
      trial_binding.GetPlusFiniteElementSpace();
   const auto& test_minus_space =
      test_binding.GetMinusFiniteElementSpace();
   const auto& test_plus_space =
      test_binding.GetPlusFiniteElementSpace();

   ValidateNonconformingGlobalInteriorFacetTransformSupport<
      std::remove_cvref_t<decltype(batch)>,
      std::remove_cvref_t<decltype(trial_minus_space)>,
      std::remove_cvref_t<decltype(trial_plus_space)>,
      std::remove_cvref_t<decltype(test_minus_space)>,
      std::remove_cvref_t<decltype(test_plus_space)>>();

   auto dofs_in_minus = MakeReadOnlyElementTensorView<KernelPolicy>(
      trial_minus_space,
      dofs_vector_in);
   auto dofs_in_plus = MakeReadOnlyElementTensorView<KernelPolicy>(
      trial_plus_space,
      dofs_vector_in);
   auto dofs_out_minus = MakeReadWriteElementTensorView<KernelPolicy>(
      test_minus_space,
      dofs_vector_out);
   auto dofs_out_plus = MakeReadWriteElementTensorView<KernelPolicy>(
      test_plus_space,
      dofs_vector_out);

   GenericCanonicalGlobalInteriorFacetDomainOperator<KernelPolicy>(
      batch_ctx,
      batch,
      weak_form,
      integration_rule,
      dofs_in_minus,
      dofs_in_plus,
      dofs_out_minus,
      dofs_out_plus);
}

} // namespace gendil

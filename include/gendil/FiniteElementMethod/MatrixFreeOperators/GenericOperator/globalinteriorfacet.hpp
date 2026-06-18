// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/globalfacethelpers.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/globaloperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/localinteriorfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/restrictedweakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"

namespace gendil {

template<class Space>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetInteriorFaceOperatorVolumeSpace(
   const Space& space)
{
   using SpaceType = std::remove_cvref_t<Space>;
   static_assert(
      supports_one_sided_face_qdata_v<SpaceType>,
      "GetInteriorFaceOperatorVolumeSpace is a one-sided same-space "
      "compatibility helper. Two-space interior face finite element spaces "
      "require side-specific minus/plus volume spaces and qdata.");

   if constexpr (is_face_finite_element_space_v<SpaceType>)
   {
      static_assert(
         is_interior_face_finite_element_space_v<SpaceType>,
         "GetInteriorFaceOperatorVolumeSpace requires an interior face finite "
         "element space or a homogeneous volume finite element space.");
      static_assert(
         is_same_space_interior_face_finite_element_space_v<SpaceType> ||
         std::is_same_v<
            std::remove_cvref_t<decltype(space.GetMinusFiniteElementSpace())>,
            std::remove_cvref_t<decltype(space.GetPlusFiniteElementSpace())>>,
         "GenericExplicitOperator cannot evaluate cross-space interior face "
         "batches yet; two-space global face lowering is not supported in "
         "this milestone.");
      return space.GetMinusFiniteElementSpace();
   }
   else
   {
      return (space);
   }
}

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
      auto swapped_operator_context =
         SwapFacetOperatorContext(operator_context);
      LocalInteriorFacetIntegrandOperator(
         kernel_context,
         weak_form_context,
         swapped_operator_context,
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
   typename FaceDomain,
   typename Integrand,
   typename IntegrationRule,
   typename DofsInView,
   typename DofsOutView>
void GenericGlobalInteriorFacetDomainOperator(
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
   const auto& trial_space =
      GetInteriorFaceOperatorVolumeSpace(trial_binding);
   const auto& test_space =
      GetInteriorFaceOperatorVolumeSpace(test_binding);
   const auto& face_space = face_domain.GetMinusFiniteElementSpace();

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

   constexpr size_t required_shared_mem =
      global_generic_interior_facet_required_shared_memory_v<
         KernelPolicy,
         std::remove_cvref_t<IntegrationRule>,
         std::remove_cvref_t<decltype(face_space)>,
         Integrand,
         DofsInView,
         DofsOutView>;
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

         GenericGlobalInteriorFacetIntegrandOperator(
            kernel,
            wf_ctx,
            facet_op_ctx,
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
   class FaceSpace,
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
      FaceSpace>& batch,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   static_assert(
      is_same_space_interior_face_finite_element_space_v<FaceSpace>,
      "GenericExplicitOperator cannot evaluate cross-space interior face "
      "batches yet; two-space global face lowering is not supported in this "
      "milestone.");

   const auto& face_space = batch.GetInteriorFaceFiniteElementSpace();
   const auto& finite_element_space = batch.GetMinusCellFiniteElementSpace();
   auto batch_ctx =
      MakeRestrictedWeakFormContext<TrialName, TestName>(
         wf_ctx,
         domain_tag,
         batch);

   auto dofs_in = MakeReadOnlyElementTensorView<KernelPolicy>(
      finite_element_space,
      dofs_vector_in);
   auto dofs_out = MakeReadWriteElementTensorView<KernelPolicy>(
      finite_element_space,
      dofs_vector_out);

   GenericGlobalInteriorFacetDomainOperator<KernelPolicy>(
      batch_ctx,
      face_space,
      weak_form,
      integration_rule,
      dofs_in,
      dofs_out);
}

} // namespace gendil

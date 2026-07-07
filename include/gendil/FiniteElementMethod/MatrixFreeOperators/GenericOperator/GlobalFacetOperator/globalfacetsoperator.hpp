// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/CellOperator/celloperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/domainfiniteelementspaceiteration.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globaloperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/GlobalFacetOperator/globalinteriorfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/GlobalFacetOperator/globalboundaryfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/restrictedweakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/readdofs.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/writedofs.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumformexpr.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

namespace gendil {

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   StaticString DomainName,
   size_t CellI,
   class CellSpace,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericCellDomainOperator(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   Cells<DomainName> domain_tag,
   const CellExecutionBatch<DomainName, CellI, CellSpace>& batch,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   const auto& cell_space = batch.GetCellFiniteElementSpace();
   auto batch_ctx =
      MakeRestrictedWeakFormContext<TrialName, TestName>(
         wf_ctx,
         domain_tag,
         batch);
   auto batch_op_ctx = MakeCellOnlyOperatorContext(batch_ctx, integration_rule);

   auto dofs_in =
      MakeReadOnlyElementTensorView<KernelPolicy>(cell_space, dofs_vector_in);
   auto dofs_out =
      MakeReadWriteElementTensorView<KernelPolicy>(cell_space, dofs_vector_out);

   using DofsInView = decltype(dofs_in);
   using DofsOutView = decltype(dofs_out);
   using BatchOperatorContext = decltype(batch_op_ctx);
   using BatchIntegrationRule =
      decltype(std::declval<BatchOperatorContext>().integration_rule());

   constexpr size_t required_shared_mem =
      local_generic_cell_required_shared_memory_v<
         KernelPolicy,
         BatchIntegrationRule,
         std::remove_cvref_t<decltype(cell_space)>,
         WeakForm,
         DofsInView,
         DofsOutView>;
   constexpr size_t shared_memory_block_size =
      KernelContext<
         KernelPolicy,
         required_shared_mem>::shared_memory_block_size;

   mesh::CellIterator<KernelPolicy>(
      cell_space,
      [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
      {
         (void)batch_ctx;
         (void)batch_op_ctx;
         (void)weak_form;

         GENDIL_SHARED Real _shared_mem[
            shared_memory_block_size == 0
               ? 1
               : shared_memory_block_size];
         KernelContext<KernelPolicy, required_shared_mem> kernel(_shared_mem);

         auto u_elem = ReadDofs(kernel, cell_space, element_index, dofs_in);

         using VType = decltype(ReadDofs(
            kernel,
            cell_space,
            element_index,
            dofs_out));
         VType v_elem{};

         ElementContext element_context{
            element_index,
            cell_space.GetCell(element_index)};

         GenericCellIntegrandOperator(
            kernel,
            batch_ctx,
            batch_op_ctx,
            element_context,
            weak_form,
            u_elem,
            v_elem);

         WriteAddDofs(kernel, cell_space, element_index, v_elem, dofs_out);
      });
}

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalCellPhase(
   const WeakForm&,
   const WeakFormContext&,
   const IntegrationRule&,
   const DofsInVector&,
   DofsOutVector&)
{}

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   StaticString DomainName,
   FieldExpr Expr,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalCellPhase(
   const Integrand<Cells<DomainName>, Expr>& integrand,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   ForEachCellFiniteElementSpace(
      wf_ctx,
      Cells<DomainName>{},
      [&] (const auto& batch)
      {
         GenericCellDomainOperator<
            TrialName,
            TestName,
            KernelPolicy>(
               integrand,
               wf_ctx,
               Cells<DomainName>{},
               batch,
               integration_rule,
               dofs_vector_in,
               dofs_vector_out);
      });
}

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   class Map,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalCellPhase(
   const SumFormExpr<Map>& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   std::apply(
      [&] (const auto&... entries)
      {
         (
            GenericGlobalCellPhase<
               TrialName,
               TestName,
               KernelPolicy>(
                  entries.value,
                  wf_ctx,
                  integration_rule,
                  dofs_vector_in,
                  dofs_vector_out),
            ...
         );
      },
      weak_form.map.entries);
}

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalInteriorFacePhase(
   const WeakForm&,
   const WeakFormContext&,
   const IntegrationRule&,
   const DofsInVector&,
   DofsOutVector&)
{}

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   StaticString DomainName,
   FieldExpr Expr,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalInteriorFacePhase(
   const Integrand<InteriorFacets<DomainName>, Expr>& integrand,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   static_assert(
      std::remove_cvref_t<WeakFormContext>::template
         has_interior_face_domain<DomainName>(),
      "GenericGlobalDomainOperator: InteriorFacets<Name> requires an "
      "interior face domain registered under Name when global traversal is "
      "selected.");

   ForEachInteriorFaceFiniteElementSpace(
      wf_ctx,
      InteriorFacets<DomainName>{},
      [&] (const auto& batch)
      {
         using Batch = std::remove_cvref_t<decltype(batch)>;
         if constexpr (is_interior_face_execution_batch_v<Batch>)
         {
            GenericGlobalInteriorFaceDomainOperator<
               TrialName,
               TestName,
               KernelPolicy>(
                  integrand,
                  wf_ctx,
                  InteriorFacets<DomainName>{},
                  batch,
                  integration_rule,
                  dofs_vector_in,
                  dofs_vector_out);
         }
         else
         {
            static_assert(
               dependent_false_v<Batch>,
               "GenericGlobalDomainOperator: InteriorFacets<Name> resolved "
               "to a non-global execution batch.");
         }
      });
}

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   class Map,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalInteriorFacePhase(
   const SumFormExpr<Map>& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   std::apply(
      [&] (const auto&... entries)
      {
         (
            GenericGlobalInteriorFacePhase<
               TrialName,
               TestName,
               KernelPolicy>(
                  entries.value,
                  wf_ctx,
                  integration_rule,
                  dofs_vector_in,
                  dofs_vector_out),
            ...
         );
      },
      weak_form.map.entries);
}

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalBoundaryFacePhase(
   const WeakForm&,
   const WeakFormContext&,
   const IntegrationRule&,
   const DofsInVector&,
   DofsOutVector&)
{}

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   StaticString DomainName,
   FieldExpr Expr,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalBoundaryFacePhase(
   const Integrand<BoundaryFacets<DomainName>, Expr>& integrand,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   static_assert(
      std::remove_cvref_t<WeakFormContext>::template
         has_boundary_face_domain<DomainName>(),
      "GenericGlobalDomainOperator: BoundaryFacets<Name> requires a "
      "boundary face domain registered under Name when global traversal is "
      "selected.");

   ForEachBoundaryFaceFiniteElementSpace(
      wf_ctx,
      BoundaryFacets<DomainName>{},
      [&] (const auto& batch)
      {
         using Batch = std::remove_cvref_t<decltype(batch)>;
         if constexpr (is_boundary_face_execution_batch_v<Batch>)
         {
            GenericGlobalBoundaryFaceDomainOperator<
               TrialName,
               TestName,
               KernelPolicy>(
                  integrand,
                  wf_ctx,
                  BoundaryFacets<DomainName>{},
                  batch,
                  integration_rule,
                  dofs_vector_in,
                  dofs_vector_out);
         }
         else
         {
            static_assert(
               dependent_false_v<Batch>,
               "GenericGlobalDomainOperator: BoundaryFacets<Name> resolved "
               "to a non-global execution batch.");
         }
      });
}

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   class Map,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalBoundaryFacePhase(
   const SumFormExpr<Map>& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   std::apply(
      [&] (const auto&... entries)
      {
         (
            GenericGlobalBoundaryFacePhase<
               TrialName,
               TestName,
               KernelPolicy>(
                  entries.value,
                  wf_ctx,
                  integration_rule,
                  dofs_vector_in,
                  dofs_vector_out),
            ...
         );
      },
      weak_form.map.entries);
}

template<
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericGlobalDomainOperator(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   static_assert(
      global_facet_domain_requirements_satisfied_v<WeakForm, WeakFormContext>,
      "GenericGlobalDomainOperator: hybrid local/global facet execution is "
      "not supported. When any global face domain is present, every "
      "InteriorFacets<Name> term must have an interior face domain registered "
      "under Name and every BoundaryFacets<Name> term must have a boundary "
      "face domain registered under Name.");

   GenericGlobalCellPhase<TrialName, TestName, KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      dofs_vector_in,
      dofs_vector_out);

   GenericGlobalInteriorFacePhase<TrialName, TestName, KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      dofs_vector_in,
      dofs_vector_out);

   GenericGlobalBoundaryFacePhase<TrialName, TestName, KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      dofs_vector_in,
      dofs_vector_out);
}

} // namespace gendil

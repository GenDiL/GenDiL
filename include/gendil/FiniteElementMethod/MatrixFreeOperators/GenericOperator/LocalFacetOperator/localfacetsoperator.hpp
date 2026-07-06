// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/CellOperator/celloperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/domainfiniteelementspaceiteration.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/localoperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/LocalFacetOperator/localinteriorfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/LocalFacetOperator/localboundaryfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/restrictedweakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/readdofs.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/writedofs.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumformexpr.hpp"

namespace gendil {

template<class Form>
struct generic_operator_local_domain_names
{
   using type = type_list<>;
};

template<StaticString Name, FieldExpr Expr>
struct generic_operator_local_domain_names<Integrand<Cells<Name>, Expr>>
{
   using type = type_list<NameTag<Name>>;
};

template<StaticString Name, FieldExpr Expr>
struct generic_operator_local_domain_names<Integrand<InteriorFacets<Name>, Expr>>
{
   using type = type_list<NameTag<Name>>;
};

template<StaticString Name, FieldExpr Expr>
struct generic_operator_local_domain_names<Integrand<BoundaryFacets<Name>, Expr>>
{
   using type = type_list<NameTag<Name>>;
};

template<class Key, class T>
struct generic_operator_local_domain_names<Entry<Key, T>>
   : generic_operator_local_domain_names<T> {};

template<class... Entries>
struct generic_operator_local_domain_names<StaticMap<Entries...>>
{
   using type = unique_t<concat_many_t<
      typename generic_operator_local_domain_names<Entries>::type...>>;
};

template<class Map>
struct generic_operator_local_domain_names<SumFormExpr<Map>>
   : generic_operator_local_domain_names<Map> {};

template<class Form>
using generic_operator_local_domain_names_t =
   typename generic_operator_local_domain_names<std::remove_cvref_t<Form>>::type;

template<class Fn, class... NameTags>
void ForEachGenericOperatorLocalDomainName(type_list<NameTags...>, Fn&& fn)
{
   (fn(NameTags{}), ...);
}

template<
   StaticString SelectedName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class ElementContext,
   class WeakForm,
   class ElementDofsIn,
   class ElementDofsOut>
GENDIL_HOST_DEVICE
void GenericCellIntegrandOperatorForDomain(
   KernelContext& kernel,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext& element_context,
   const WeakForm& weak_form,
   const ElementDofsIn& u_elem,
   ElementDofsOut& v_elem)
{
   if constexpr (CellIntegrand<WeakForm>)
   {
      if constexpr (WeakForm::domain_type::name == SelectedName)
      {
         GenericCellIntegrandOperator(
            kernel,
            wf_ctx,
            op_ctx,
            element_context,
            weak_form,
            u_elem,
            v_elem);
      }
   }
}

template<
   StaticString SelectedName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class ElementContext,
   class Map,
   class ElementDofsIn,
   class ElementDofsOut>
GENDIL_HOST_DEVICE
void GenericCellIntegrandOperatorForDomain(
   KernelContext& kernel,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext& element_context,
   const SumFormExpr<Map>& weak_form,
   const ElementDofsIn& u_elem,
   ElementDofsOut& v_elem)
{
   std::apply(
      [&] (auto const&... entries)
      {
         (
            GenericCellIntegrandOperatorForDomain<SelectedName>(
               kernel,
               wf_ctx,
               op_ctx,
               element_context,
               entries.value,
               u_elem,
               v_elem),
            ...
         );
      },
      weak_form.map.entries);
}

template<
   StaticString SelectedName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class ElementContext,
   class TrialSpace,
   class WeakForm,
   class DofsInView,
   class ElementDofsIn,
   class ElementDofsOut>
GENDIL_HOST_DEVICE
void LocalInteriorFacetOperatorForDomain(
   KernelContext& kernel,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext& element_context,
   const TrialSpace& trial_space,
   const WeakForm& weak_form,
   const DofsInView& dofs_in,
   const ElementDofsIn& u_elem,
   ElementDofsOut& v_elem)
{
   if constexpr (InteriorFacetIntegrand<WeakForm>)
   {
      if constexpr (WeakForm::domain_type::name == SelectedName)
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
            v_elem);
      }
   }
}

template<
   StaticString SelectedName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class ElementContext,
   class TrialSpace,
   class Map,
   class DofsInView,
   class ElementDofsIn,
   class ElementDofsOut>
GENDIL_HOST_DEVICE
void LocalInteriorFacetOperatorForDomain(
   KernelContext& kernel,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext& element_context,
   const TrialSpace& trial_space,
   const SumFormExpr<Map>& weak_form,
   const DofsInView& dofs_in,
   const ElementDofsIn& u_elem,
   ElementDofsOut& v_elem)
{
   std::apply(
      [&] (auto const&... entries)
      {
         (
            LocalInteriorFacetOperatorForDomain<SelectedName>(
               kernel,
               wf_ctx,
               op_ctx,
               element_context,
               trial_space,
               entries.value,
               dofs_in,
               u_elem,
               v_elem),
            ...
         );
      },
      weak_form.map.entries);
}

template<
   StaticString SelectedName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class ElementContext,
   class TrialSpace,
   class WeakForm,
   class ElementDofsIn,
   class ElementDofsOut>
GENDIL_HOST_DEVICE
void LocalBoundaryFacetOperatorForDomain(
   KernelContext& kernel,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext& element_context,
   const TrialSpace& trial_space,
   const WeakForm& weak_form,
   const ElementDofsIn& u_elem,
   ElementDofsOut& v_elem)
{
   if constexpr (BoundaryFacetIntegrand<WeakForm>)
   {
      if constexpr (WeakForm::domain_type::name == SelectedName)
      {
         LocalBoundaryFacetOperator(
            kernel,
            wf_ctx,
            op_ctx,
            element_context,
            trial_space,
            weak_form,
            u_elem,
            v_elem);
      }
   }
}

template<
   StaticString SelectedName,
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class ElementContext,
   class TrialSpace,
   class Map,
   class ElementDofsIn,
   class ElementDofsOut>
GENDIL_HOST_DEVICE
void LocalBoundaryFacetOperatorForDomain(
   KernelContext& kernel,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext& element_context,
   const TrialSpace& trial_space,
   const SumFormExpr<Map>& weak_form,
   const ElementDofsIn& u_elem,
   ElementDofsOut& v_elem)
{
   std::apply(
      [&] (auto const&... entries)
      {
         (
            LocalBoundaryFacetOperatorForDomain<SelectedName>(
               kernel,
               wf_ctx,
               op_ctx,
               element_context,
               trial_space,
               entries.value,
               u_elem,
               v_elem),
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
   StaticString DomainName,
   size_t CellI,
   class CellSpace,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericLocalCellBatchOperator(
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
   auto batch_op_ctx = MakeOperatorContext(batch_ctx, integration_rule);

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

   mesh::CellIterator<KernelPolicy>(
      cell_space,
      [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
      {
         (void)batch_ctx;
         (void)batch_op_ctx;
         (void)weak_form;

         GENDIL_SHARED Real _shared_mem[required_shared_mem];
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

         GenericCellIntegrandOperatorForDomain<DomainName>(
            kernel,
            batch_ctx,
            batch_op_ctx,
            element_context,
            weak_form,
            u_elem,
            v_elem);

         LocalInteriorFacetOperatorForDomain<DomainName>(
            kernel,
            batch_ctx,
            batch_op_ctx,
            element_context,
            cell_space,
            weak_form,
            dofs_in,
            u_elem,
            v_elem);

         LocalBoundaryFacetOperatorForDomain<DomainName>(
            kernel,
            batch_ctx,
            batch_op_ctx,
            element_context,
            cell_space,
            weak_form,
            u_elem,
            v_elem);

         WriteAddDofs(kernel, cell_space, element_index, v_elem, dofs_out);
      });
}

template<
   StaticString DomainName,
   StaticString TrialName,
   StaticString TestName,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericLocalDomainOperatorForName(
   const WeakForm& weak_form,
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
         GenericLocalCellBatchOperator<
            TrialName,
            TestName,
            KernelPolicy>(
               weak_form,
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
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericLocalDomainOperator(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   GENDIL_REQUIRE_UNBATCHED_OPERATOR(KernelPolicy);

   using DomainNames = generic_operator_local_domain_names_t<WeakForm>;

   ForEachGenericOperatorLocalDomainName(
      DomainNames{},
      [&] (auto domain_name_tag)
      {
         constexpr auto DomainName =
            std::remove_cvref_t<decltype(domain_name_tag)>::name;
         GenericLocalDomainOperatorForName<
            DomainName,
            TrialName,
            TestName,
            KernelPolicy>(
               weak_form,
               wf_ctx,
               integration_rule,
               dofs_vector_in,
               dofs_vector_out);
      });
}

} // namespace gendil

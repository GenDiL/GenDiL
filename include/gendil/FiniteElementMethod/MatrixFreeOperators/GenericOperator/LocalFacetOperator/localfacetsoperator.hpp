// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/CellOperator/celloperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/domainiteration.hpp"
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
struct generic_operator_domain_names
{
   using type = type_list<>;
};

template<StaticString Name, FieldExpr Expr>
struct generic_operator_domain_names<
   Integrand<Cells<Name>, Expr>>
{
   using type = type_list<NameTag<Name>>;
};

template<StaticString Name, FieldExpr Expr>
struct generic_operator_domain_names<
   Integrand<InteriorFacets<Name>, Expr>>
{
   using type = type_list<NameTag<Name>>;
};

template<StaticString Name, FieldExpr Expr>
struct generic_operator_domain_names<
   Integrand<BoundaryFacets<Name>, Expr>>
{
   using type = type_list<NameTag<Name>>;
};

template<class Key, class T>
struct generic_operator_domain_names<Entry<Key, T>>
   : generic_operator_domain_names<T> {};

template<class... Entries>
struct generic_operator_domain_names<StaticMap<Entries...>>
{
   using type = unique_t<concat_many_t<
      typename generic_operator_domain_names<Entries>::type...>>;
};

template<class Map>
struct generic_operator_domain_names<SumFormExpr<Map>>
   : generic_operator_domain_names<Map> {};

template<class Form>
using generic_operator_domain_names_t =
   typename generic_operator_domain_names<
      std::remove_cvref_t<Form>>::type;

template<class Fn, class... NameTags>
void ForEachGenericOperatorDomainName(type_list<NameTags...>, Fn&& fn)
{
   (fn(NameTags{}), ...);
}

template<StaticString Name, class EntryType>
constexpr auto SelectGenericOperatorDomainEntry(const EntryType& entry)
{
   using IntegrandType = typename EntryType::value_type;
   if constexpr (IntegrandType::domain_type::name == Name)
   {
      return std::tuple{ entry };
   }
   else
   {
      return std::tuple{};
   }
}

template<StaticString Name, class Map>
constexpr auto MakeGenericOperatorDomainWeakForm(
   const SumFormExpr<Map>& weak_form)
{
   auto selected_entries = std::apply(
      [] (const auto&... entries)
      {
         return std::tuple_cat(
            SelectGenericOperatorDomainEntry<Name>(entries)...);
      },
      weak_form.map.entries);

   static_assert(
      std::tuple_size_v<decltype(selected_entries)> > 0,
      "GenericLocalDomainOperator selected an empty named mesh subform.");

   auto selected_map = tuple_to_map(std::move(selected_entries));
   return SumFormExpr<decltype(selected_map)>{
      std::move(selected_map) };
}

template<StaticString Name, DomainExpr Domain, FieldExpr Expr>
constexpr auto MakeGenericOperatorDomainWeakForm(
   const Integrand<Domain, Expr>& integrand)
{
   static_assert(
      Domain::name == Name,
      "GenericLocalDomainOperator selected the wrong named mesh integrand.");
   return integrand;
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
   class DomainMesh,
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
   const DomainMesh& domain_mesh,
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
            domain_mesh,
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
   class DomainMesh,
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
   const DomainMesh& domain_mesh,
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
               domain_mesh,
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
   class DomainMesh,
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
   const DomainMesh& domain_mesh,
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
            domain_mesh,
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
   class DomainMesh,
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
   const DomainMesh& domain_mesh,
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
               domain_mesh,
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
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   StaticString DomainName,
   size_t CellI,
   class CellMesh,
   bool Partitioned,
   class IntegrationRule,
   class DofsInVector,
   class DofsOutVector>
void GenericLocalCellBatchOperator(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   Cells<DomainName> domain_tag,
   const SelectedCellExecutionDomain<
      DomainName,
      CellI,
      CellMesh,
      Partitioned>& batch,
   const IntegrationRule& integration_rule,
   const DofsInVector& dofs_vector_in,
   DofsOutVector& dofs_vector_out)
{
   constexpr auto TrialName = requirements<WeakForm>::trial_name;
   constexpr auto TestName = requirements<WeakForm>::test_name;

   auto batch_ctx =
      MakeRestrictedWeakFormContext<WeakForm>(
         wf_ctx,
         domain_tag,
         batch);
   auto batch_op_ctx = MakeOperatorContext(batch_ctx, integration_rule);
   const auto& domain_mesh = batch.GetCellMesh();
   const auto& trial_space =
      batch_ctx.template fe_field<TrialName>().space;
   const auto& test_space =
      batch_ctx.template fe_field<TestName>().space;

   auto dofs_in =
      MakeReadOnlyElementTensorView<KernelPolicy>(
         trial_space,
         dofs_vector_in);
   auto dofs_out =
      MakeReadWriteElementTensorView<KernelPolicy>(
         test_space,
         dofs_vector_out);

   using DofsInView = decltype(dofs_in);
   using DofsOutView = decltype(dofs_out);
   using BatchOperatorContext = decltype(batch_op_ctx);
   using BatchIntegrationRule =
      decltype(std::declval<BatchOperatorContext>().integration_rule());

   constexpr size_t required_shared_mem =
      local_generic_cell_required_shared_memory_v<
         KernelPolicy,
         BatchIntegrationRule,
         std::remove_cvref_t<decltype(trial_space)>,
         WeakForm,
         DofsInView,
         DofsOutView>;
   using Context = KernelContext<KernelPolicy, required_shared_mem>;

   mesh::CellIterator<KernelPolicy>(
      domain_mesh,
      [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
      {
         (void)batch_ctx;
         (void)batch_op_ctx;
         (void)weak_form;

         GENDIL_SHARED Real _shared_mem[Context::shared_memory_block_size];
         Context kernel(_shared_mem);

         auto u_elem =
            ReadDofs(kernel, trial_space, element_index, dofs_in);

         using VType = decltype(ReadDofs(
            kernel,
            test_space,
            element_index,
            dofs_out));
         VType v_elem{};

         ElementContext element_context{
            element_index,
            domain_mesh.GetCell(element_index)};

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
            domain_mesh,
            trial_space,
            weak_form,
            dofs_in,
            u_elem,
            v_elem);

         LocalBoundaryFacetOperatorForDomain<DomainName>(
            kernel,
            batch_ctx,
            batch_op_ctx,
            element_context,
            domain_mesh,
            trial_space,
            weak_form,
            u_elem,
            v_elem);

         WriteAddDofs(
            kernel,
            test_space,
            element_index,
            v_elem,
            dofs_out);
      });
}

template<
   StaticString DomainName,
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
   ForEachCellExecutionDomain(
      wf_ctx,
      Cells<DomainName>{},
      [&] (const auto& batch)
      {
         GenericLocalCellBatchOperator<KernelPolicy>(
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
   using DomainNames =
      generic_operator_domain_names_t<WeakForm>;
   static_assert(
      !std::is_same_v<DomainNames, type_list<>>,
      "GenericLocalDomainOperator requires at least one integration domain.");

   GENDIL_REQUIRE_UNBATCHED_OPERATOR(KernelPolicy);

   ForEachGenericOperatorDomainName(
      DomainNames{},
      [&] (auto domain_name_tag)
      {
         constexpr auto DomainName =
            std::remove_cvref_t<decltype(domain_name_tag)>::name;
         auto domain_weak_form =
            MakeGenericOperatorDomainWeakForm<DomainName>(weak_form);
         GenericLocalDomainOperatorForName<
            DomainName,
            KernelPolicy>(
               domain_weak_form,
               wf_ctx,
               integration_rule,
               dofs_vector_in,
               dofs_vector_out);
      });
}

} // namespace gendil

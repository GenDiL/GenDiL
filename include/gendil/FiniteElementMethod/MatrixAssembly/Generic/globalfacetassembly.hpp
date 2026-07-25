// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrationdomain.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

template<class Form, class WFContext>
struct weak_form_uses_partition_integration_domain : std::false_type {};

template<class Domain, FieldExpr Expr, class WFContext>
struct weak_form_uses_partition_integration_domain<
   Integrand<Domain, Expr>,
   WFContext>
{
   static constexpr auto Name = Domain::name;
   using Context = std::remove_cvref_t<WFContext>;
   static_assert(
      Context::template has_domain<Name>(),
      "GenericAssembly: weak-form integrand requires an integration domain "
      "registered under its domain name.");
   using NormalizedDomain = std::remove_cvref_t<
      decltype(std::declval<const Context&>().template domain<Name>())>;
   using IntegrationDomain = std::remove_cvref_t<
      decltype(std::declval<NormalizedDomain>().domain)>;
   static constexpr bool value =
      is_partition_integration_domain_v<IntegrationDomain>;
};

template<class Key, class T, class WFContext>
struct weak_form_uses_partition_integration_domain<
   Entry<Key, T>,
   WFContext>
   : weak_form_uses_partition_integration_domain<T, WFContext> {};

template<class... Entries, class WFContext>
struct weak_form_uses_partition_integration_domain<
   StaticMap<Entries...>,
   WFContext>
   : std::bool_constant<
        (
           weak_form_uses_partition_integration_domain<
              Entries,
              WFContext>::value ||
           ...)> {};

template<class Map, class WFContext>
struct weak_form_uses_partition_integration_domain<
   SumFormExpr<Map>,
   WFContext>
   : weak_form_uses_partition_integration_domain<Map, WFContext> {};

template<class WeakForm, class WFContext>
inline constexpr bool weak_form_uses_partition_integration_domain_v =
   weak_form_uses_partition_integration_domain<
      std::remove_cvref_t<WeakForm>,
      std::remove_cvref_t<WFContext>>::value;

template<class WeakForm, class WFContext>
consteval void ValidateSparseAssemblyDomainSupport()
{
   static_assert(
      !weak_form_uses_partition_integration_domain_v<
         WeakForm,
         WFContext>,
      "GenericAssembly: PartitionIntegrationDomain is unsupported by sparse "
      "assembly; use a MeshIntegrationDomain or matrix-free GenericOperator.");
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class SparseMatrixType>
void GenericGlobalFacetAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   SparseMatrixType& sparse_matrix)
{
   (void)weak_form;
   (void)wf_ctx;
   (void)integration_rule;
   (void)sparse_matrix;
   static_assert(
      dependent_false_v<KernelPolicy, WeakForm>,
      "GenericGlobalFacetAssembly: PartitionIntegrationDomain is unsupported "
      "by sparse assembly; use a MeshIntegrationDomain or matrix-free "
      "GenericOperator.");
}

} // namespace gendil

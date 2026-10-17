// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/weakformvalidation.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/LocalFacetOperator/localfacetsoperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/GlobalFacetOperator/globalfacetsoperator.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/weakformtraits.hpp"
#include "gendil/Algebra/vectormapconcept.hpp"
#include "gendil/Algebra/Vector/operations.hpp"
#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"

namespace gendil {

namespace generic_operator_detail {

template<class Form>
inline constexpr bool has_callable_input_v =
   !is_empty_type_list_v<typename form_signature_t<Form>::callable_inputs>;

// Clear the already acquired view, keeping one owner access per invocation.
template<bool OnDevice, class Access>
void ClearOutputAccess(Access access, size_t size)
{
   KernelLoop<OnDevice>(size, [access] GENDIL_HOST_DEVICE (size_t i) mutable
   {
      access[i] = Real{0};
   });
}

template<class Context, class... Fields>
size_t GetMaximumAlgebraicDofExtent(const Context& context, type_list<Fields...>)
{
   size_t extent = 0;
   auto add = [&]<class Field>() {
      static_assert(Context::template HasFiniteElementField<Field::name>(),
         "Weak-form execution requires a registered finite-element space.");
      if constexpr (Context::template HasFiniteElementField<Field::name>())
      {
         const auto field_extent = static_cast<size_t>(
            GetAlgebraicDofExtent(context.template GetSpace<Field::name>()));
         if (field_extent > extent) { extent = field_extent; }
      }
   };
   (add.template operator()<Fields>(), ...);
   return extent;
}

// Storage access is acquired once by the caller, independently of traversal.
template<class Policy, class Form, class Context, class Rule, class... Access>
void ApplyOnDomains(const Form& form, const Context& context, const Rule& rule,
                const Access&... access)
{
   if constexpr (is_zero<Form>)
      return;
   else if constexpr (generic_operator_domain_kind_v<Form, Context> == GenericOperatorDomainKind::Mesh)
      GenericLocalFacetDomainOperator<Policy>(form, context, rule, access...);
   else
      GenericGlobalFacetDomainOperator<Policy>(form, context, rule, access...);
}

// Callable dimensions remain required even when an affine transform removes reads.
template <
   typename WeakForm,
   typename WeakFormContext,
   typename Output,
   typename... Input >
GENDIL_INLINE
void ValidateVectorSizes(
   const WeakFormContext & weak_form_context,
   Output & output,
   const Input &... input )
{
   using Signature = form_signature_t< WeakForm >;
   GENDIL_VERIFY(
      GetVectorSize( output ) >= GetMaximumAlgebraicDofExtent(
         weak_form_context,
         typename Signature::logical_outputs{} ),
      "Weak-form output vector is smaller than its required algebraic extent." );
   ( ( GENDIL_VERIFY(
      GetVectorSize( input ) >= GetMaximumAlgebraicDofExtent(
         weak_form_context,
         typename Signature::callable_inputs{} ),
      "Weak-form input vector is smaller than its required algebraic extent." ) ), ... );
}

// The caller validates first. Acquire each required vector once, clear only for
// replacement, and borrow the input only for this invocation.
template <
   bool Add,
   typename KernelPolicy,
   typename WeakForm,
   typename WeakFormContext,
   typename IntegrationRule,
   typename Output,
   typename... Input >
GENDIL_INLINE
void ApplyToVectors(
   const WeakForm & weak_form,
   const WeakFormContext & weak_form_context,
   const IntegrationRule & integration_rule,
   Output & output,
   const Input &... input )
{
   if constexpr ( !( Add && is_zero< WeakForm > ) )
   {
      constexpr bool OnDevice = is_device_configuration_v< KernelPolicy >;
      auto output_data = [&]
      {
         if constexpr ( Add )
         {
            return ReadWriteKernelVector< OnDevice >( output );
         }
         else
         {
            return WriteKernelVector< OnDevice >( output );
         }
      }();

      if constexpr ( !Add )
      {
         ClearOutputAccess< OnDevice >( output_data, GetVectorSize( output ) );
      }

      if constexpr ( !is_zero< WeakForm > )
      {
         using Signature = form_signature_t< WeakForm >;
         if constexpr ( !is_empty_type_list_v< typename Signature::active_inputs > )
         {
            auto input_data = ReadKernelVector< OnDevice >( std::get< 0 >( std::tie( input... ) ) );
            ApplyOnDomains< KernelPolicy >(
               weak_form, weak_form_context, integration_rule, input_data, output_data );
         }
         else
         {
            ApplyOnDomains< KernelPolicy >(
               weak_form, weak_form_context, integration_rule, output_data );
         }
      }
   }
}

template <
   bool Add,
   typename KernelPolicy,
   typename WeakForm,
   typename WeakFormContext,
   typename IntegrationRule,
   typename Output,
   typename... Input >
void Apply(
   const WeakForm & weak_form,
   const WeakFormContext & weak_form_context,
   const IntegrationRule & integration_rule,
   Output & output,
   const Input &... input )
{
   ValidateWeakFormContext( weak_form, weak_form_context );

   // Keep invalid domains and invocation arities out of numerical instantiation.
   constexpr auto DomainKind = generic_operator_domain_kind_v< WeakForm, WeakFormContext >;
   if constexpr ( DomainKind != GenericOperatorDomainKind::Invalid &&
                  DomainKind != GenericOperatorDomainKind::Mixed )
   {
      constexpr bool ArityMatches = sizeof...( Input ) == ( has_callable_input_v< WeakForm > ? 1 : 0 );
      static_assert( ArityMatches,
         "Input-free residuals require the output-only GenericExplicitOperator overload." );
      if constexpr ( ArityMatches )
      {
         generic_operator_detail::ValidateVectorSizes< WeakForm >( weak_form_context, output, input... );
         generic_operator_detail::ApplyToVectors< Add, KernelPolicy >(
            weak_form, weak_form_context, integration_rule, output, input... );
      }
   }
}

} // namespace generic_operator_detail

// Direct numerical invocation is additive, like ApplyAdd on the public operator.
template<class KernelPolicy, class WeakForm, class WeakFormContext,
         class IntegrationRule, class Input, class Output>
void GenericExplicitOperator(const WeakForm& form, const WeakFormContext& context,
   const IntegrationRule& rule, const Input& input, Output& output)
{
   generic_operator_detail::Apply<true, KernelPolicy>(form, context, rule, output, input);
}

template<class KernelPolicy, class WeakForm, class WeakFormContext,
         class IntegrationRule, class Output>
   requires (!generic_operator_detail::has_callable_input_v<WeakForm>)
void GenericExplicitOperator(const WeakForm& form, const WeakFormContext& context,
   const IntegrationRule& rule, Output& output)
{
   generic_operator_detail::Apply<true, KernelPolicy>(form, context, rule, output);
}

/**
 * Evaluate named residual fields on cells and facets using a common input and
 * output vector, or an output vector alone for input-free forms. Replacement
 * clears the complete output; ApplyAdd preserves its existing values.
 */
template<class KernelPolicy, class IntegrationRule, class WeakForm, class WeakFormContext>
class PullbackGenericOperator
{
   WeakForm weak_form;
   WeakFormContext wf_ctx;
   IntegrationRule ir;

public:
   PullbackGenericOperator(WeakForm form, WeakFormContext context, IntegrationRule rule)
      : weak_form(std::move(form)), wf_ctx(std::move(context)), ir(std::move(rule))
   {
      ValidateWeakFormContext(weak_form, wf_ctx);
   }

   size_t InputSize() const
   {
      return generic_operator_detail::GetMaximumAlgebraicDofExtent(
         wf_ctx, typename form_signature_t<WeakForm>::callable_inputs{});
   }

   size_t OutputSize() const
   {
      return generic_operator_detail::GetMaximumAlgebraicDofExtent(
         wf_ctx, typename form_signature_t<WeakForm>::logical_outputs{});
   }

   template<class Input, class Output>
      requires generic_operator_detail::has_callable_input_v<WeakForm> &&
         KernelReadableVector<is_device_configuration_v<KernelPolicy>, Input> &&
         KernelWritableVector<is_device_configuration_v<KernelPolicy>, Output>
   void operator()(const Input& input, Output& output) const
   {
      generic_operator_detail::Apply<false, KernelPolicy>(weak_form, wf_ctx, ir, output, input);
   }

   template<class Input, class Output>
      requires generic_operator_detail::has_callable_input_v<WeakForm> &&
         KernelReadableVector<is_device_configuration_v<KernelPolicy>, Input> &&
         KernelWritableVector<is_device_configuration_v<KernelPolicy>, Output>
   void ApplyAdd(const Input& input, Output& output) const
   {
      generic_operator_detail::Apply<true, KernelPolicy>(weak_form, wf_ctx, ir, output, input);
   }

   template<class Output>
      requires (!generic_operator_detail::has_callable_input_v<WeakForm>) &&
         KernelWritableVector<is_device_configuration_v<KernelPolicy>, Output>
   void operator()(Output& output) const
   {
      generic_operator_detail::Apply<false, KernelPolicy>(weak_form, wf_ctx, ir, output);
   }

   template<class Output>
      requires (!generic_operator_detail::has_callable_input_v<WeakForm>) &&
         KernelWritableVector<is_device_configuration_v<KernelPolicy>, Output>
   void ApplyAdd(Output& output) const
   {
      generic_operator_detail::Apply<true, KernelPolicy>(weak_form, wf_ctx, ir, output);
   }
};

/** Construct a residual operator; preparation and bindings are obtained per call. */
template<class KernelPolicy, class IntegrationRule, class WeakForm, class WeakFormContext>
auto MakePullbackGenericOperator(WeakForm form, WeakFormContext context, IntegrationRule rule)
{
   return PullbackGenericOperator<KernelPolicy, IntegrationRule, WeakForm, WeakFormContext>(
      std::move(form), std::move(context), std::move(rule));
}

template<class KernelPolicy, class IntegrationRule, class WeakForm, class WeakFormContext>
auto MakeGenericOperator(WeakForm form, WeakFormContext context, IntegrationRule rule)
{
   return MakePullbackGenericOperator<KernelPolicy>(
      std::move(form), std::move(context), std::move(rule));
}

template<class KernelPolicy, class IntegrationRule, class WeakForm, class WeakFormContext>
auto MakeWeakFormResidualOperator(WeakForm form, WeakFormContext context, IntegrationRule rule)
{
   return MakePullbackGenericOperator<KernelPolicy>(
      std::move(form), std::move(context), std::move(rule));
}

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontextcommon.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/executioncontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/weakformrequirements.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/localoperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globaloperatorcontext.hpp"

namespace gendil
{

// Borrow the space for an integrand's sole test field. Its identity is declared
// by the integrand and does not depend on whole-form callback read roles.
template < typename Integrand, typename ExecutionContext >
   requires requires { weak_form_traits::only_test_field_name_v<Integrand>; }
GENDIL_HOST_DEVICE
constexpr const auto & GetOutputSpace(
   const Integrand & /*integrand*/,
   ExecutionContext & execution_context )
{
   constexpr auto TestName = weak_form_traits::only_test_field_name_v<Integrand>;
   return execution_context.template GetSpace<TestName>();
}

// Bind rows for the supplied original terms using their already-resolved
// signatures. Complete-form callback roles survive selection of a single term
// or a borrowed tuple; neither the form nor operator context is retained.
template<class WeakForm, class Parent, class OperatorContext, class... Handles>
   requires (std::is_lvalue_reference_v<Parent&&> && requires {
      typename std::remove_cvref_t<Parent>::execution_context_tag;
      typename OperatorContext::form_signature;
   })
GENDIL_HOST_DEVICE auto BindElementDofs(
   const WeakForm&, Parent&& parent, const OperatorContext&, Handles... handles)
{
   using Terms = typename weak_form_traits::integrand_sequence<std::remove_cvref_t<WeakForm>>::type;
   return [&]<class... Integrands>(type_list<Integrands...>) {
      using Requirements = typename weak_form_execution_detail::signature_requirements<type_list<
         typename OperatorContext::template signature_for<std::remove_cvref_t<Integrands>>...>>::type;
      return BindElementDofs<Requirements>(parent, handles...);
   }(Terms{});
}

// Local probe tensors own a current or neighbor row, independently of the
// canonical side of each visited facet. Interpolation selects the side's reads;
// these bindings only expose the already-owned DOFs under their field names.
template<bool CurrentRow, class WeakForm, class Parent, class OperatorContext, class... Handles>
GENDIL_HOST_DEVICE auto BindLocalFacetElementDofs(
   const WeakForm&, Parent& parent, const OperatorContext&, Handles... handles)
{
   using Terms = typename weak_form_traits::integrand_sequence<std::remove_cvref_t<WeakForm>>::type;
   return [&]<class... Integrands>(type_list<Integrands...>) {
      using Signatures = type_list<typename OperatorContext::template signature_for<
         std::remove_cvref_t<Integrands>>...>;
      using Reads = typename weak_form_execution_detail::signature_requirements<Signatures>::reads;
      using Writes = std::conditional_t<CurrentRow, unique_t<concat_many_t<
         typename weak_form_execution_detail::write_fields<typename OperatorContext::template
            signature_for<std::remove_cvref_t<Integrands>>::test_outputs, FieldSide::Current>::type...>>,
         type_list<>>;
      return BindElementDofs<FieldAccessRequirements<Reads, Writes>>(parent, handles...);
   }(Terms{});
}

// Facet probes select reads by side. Local interiors write only the current
// row; canonical face bindings deliver the selected side's output channels.
template<SemanticSide Side, class WeakForm, class Parent, class OperatorContext, class... Handles>
   requires ((Side == SemanticSide::Minus || Side == SemanticSide::Plus) &&
      std::is_lvalue_reference_v<Parent&&> && requires {
         typename std::remove_cvref_t<Parent>::execution_context_tag;
         typename OperatorContext::form_signature;
      })
GENDIL_HOST_DEVICE auto BindElementDofs(
   const WeakForm&, Parent&& parent, const OperatorContext&, Handles... handles)
{
   using Terms = typename weak_form_traits::integrand_sequence<std::remove_cvref_t<WeakForm>>::type;
   return [&]<class... Integrands>(type_list<Integrands...>) {
      static_assert(((OperatorContext::template signature_for<
         std::remove_cvref_t<Integrands>>::entity == SemanticEntity::InteriorFacet) && ...),
         "Side-aware local bindings require interior-facet integrands.");
      constexpr auto writes_for = []<class Signature>() {
         if constexpr (is_empty_type_list_v<typename Signature::test_outputs>)
         {
            return type_list<>{};
         }
         else
         {
            using Output = weak_form_traits::only_field_t<typename Signature::test_outputs>;
            using Space = std::remove_cvref_t<decltype(
               std::declval<const Parent&>().template GetSpace<Output::name>())>;
            static_assert(!is_boundary_face_field_binding_v<Space>,
               "Interior-facet local bindings require a volume or interior-face space.");
            if constexpr (is_interior_face_field_binding_v<Space>)
               return typename facet_side_requirements_t<Signature, Side>::writes{};
            else if constexpr (Side == SemanticSide::Minus)
               return typename weak_form_execution_detail::write_fields<
                  typename Signature::test_outputs, FieldSide::Current>::type{};
            else
               return type_list<>{};
         }
      };
      using Requirements = FieldAccessRequirements<
         unique_t<concat_many_t<typename facet_side_requirements_t<
            typename OperatorContext::template signature_for<std::remove_cvref_t<Integrands>>, Side>::reads...>>,
         unique_t<concat_many_t<decltype(writes_for.template operator()<
            typename OperatorContext::template signature_for<std::remove_cvref_t<Integrands>>>())...>>>;
      return BindElementDofs<Requirements>(parent, handles...);
   }(Terms{});
}

} // namespace gendil

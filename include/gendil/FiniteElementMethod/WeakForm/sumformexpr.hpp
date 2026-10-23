// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumfieldexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/formsymbols.hpp"

namespace gendil {

// Own original integrands and retain only compact declaration metadata when a
// transformation removes a read or replaces a term.
template<class TermsTuple, class Declarations>
struct SumFormExpr : FormBase
{
   using terms_type = TermsTuple;
   using declarations_type = Declarations;
   TermsTuple terms;

   constexpr explicit SumFormExpr(TermsTuple terms_)
      : terms(std::move(terms_)) {}
};

template<class T> struct is_sum_form_expr : std::false_type {};
template<class Terms, class Symbols>
struct is_sum_form_expr<SumFormExpr<Terms, Symbols>> : std::true_type {};
template<class T>
inline constexpr bool is_sum_form_expr_v = is_sum_form_expr<std::remove_cvref_t<T>>::value;
// Borrow terms only while their owning lvalue remains alive. A single term
// produces a tuple of references; a sum exposes its existing tuple directly.
template<FormExpr Form>
GENDIL_HOST_DEVICE constexpr decltype(auto) AsFormTerms(Form& form)
{
   if constexpr (is_sum_form_expr_v<Form>)
      return (form.terms);
   else
      return std::tie(form);
}

template<FormExpr Form> requires (!std::is_lvalue_reference_v<Form>)
void AsFormTerms(Form&&) = delete;

namespace weak_form_detail {
// Construction owns its result: copy lvalue terms and move temporary terms.
template<FormExpr Form>
constexpr auto OwnFormTerms(Form&& form)
{
   if constexpr (is_sum_form_expr_v<Form>)
      return std::forward<Form>(form).terms;
   else
      return std::tuple<std::remove_cvref_t<Form>>{std::forward<Form>(form)};
}
} // namespace weak_form_detail

template<FormExpr Form, class Fn>
GENDIL_HOST_DEVICE constexpr void ForEachIntegrand(const Form& form, Fn&& fn)
{
   if constexpr (is_sum_form_expr_v<Form>)
      std::apply([&](const auto&... terms) { (fn(terms), ...); }, form.terms);
   else
      fn(form);
}

template<FormExpr LHS, FormExpr RHS>
constexpr auto ConcatForms(LHS&& lhs, RHS&& rhs)
{
   auto terms = std::tuple_cat(weak_form_detail::OwnFormTerms(std::forward<LHS>(lhs)),
                               weak_form_detail::OwnFormTerms(std::forward<RHS>(rhs)));
   using Declarations = typename weak_form_traits::merge_symbols<
      typename form_symbols_t<LHS>::declarations,
      typename form_symbols_t<RHS>::declarations>::type;
   return SumFormExpr<decltype(terms), Declarations>{std::move(terms)};
}

template<FormExpr Form, class Fn>
constexpr auto TransformIntegrands(const Form& form, Fn&& fn)
{
   return std::apply([&](const auto&... term) {
      // Braced initialization evaluates callbacks exactly once in source order.
      // Keep returned forms intact until their carried declarations are merged.
      auto results = std::tuple{fn(term)...};
      using Declarations = typename weak_form_traits::merge_symbols<
         typename form_symbols_t<Form>::declarations,
         typename form_symbols_t<decltype(fn(term))>::declarations...>::type;
      auto transformed = std::apply([](auto&&... result) {
         return std::tuple_cat(weak_form_detail::OwnFormTerms(
            std::forward<decltype(result)>(result))...);
      }, std::move(results));
      return SumFormExpr<decltype(transformed), Declarations>{std::move(transformed)};
   }, AsFormTerms(form));
}

template<FormExpr Form>
constexpr auto MakeSumFormExpr(Form&& form)
{
   auto terms = weak_form_detail::OwnFormTerms(std::forward<Form>(form));
   return SumFormExpr<decltype(terms), typename form_symbols_t<Form>::declarations>{std::move(terms)};
}

template<DomainExpr Domain, FieldExpr LHS, FieldExpr RHS>
constexpr auto Add(const Integrand<Domain, LHS>& lhs, const Integrand<Domain, RHS>& rhs)
{ return ConcatForms(lhs, rhs); }

template<class Terms, class Symbols, FormExpr Form>
constexpr auto AddFormTerm(SumFormExpr<Terms, Symbols> sum, Form&& form)
{ return ConcatForms(std::move(sum), std::forward<Form>(form)); }

template<class LT, class LS, class RT, class RS>
constexpr auto MergeSumFormExpr(SumFormExpr<LT, LS> lhs, const SumFormExpr<RT, RS>& rhs)
{ return ConcatForms(std::move(lhs), rhs); }

template<FormExpr LHS, FormExpr RHS>
constexpr auto operator+(LHS&& lhs, RHS&& rhs)
{ return ConcatForms(std::forward<LHS>(lhs), std::forward<RHS>(rhs)); }

template<class Terms, class Symbols>
std::ostream& operator<<(std::ostream& os, const SumFormExpr<Terms, Symbols>& sum)
{
   bool first = true;
   ForEachIntegrand(sum, [&](const auto& term) {
      os << (first ? "" : " + ") << term;
      first = false;
   });
   return os;
}

} // namespace gendil

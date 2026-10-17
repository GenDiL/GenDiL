// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/signatures.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globalfacefieldbinding.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumformexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"

#include <type_traits>

namespace gendil {

namespace generic_operator_detail {

enum class GenericOperatorDomainKind
{
   Mesh,
   Partition,
   Mixed,
   Invalid
};

constexpr GenericOperatorDomainKind MergeDomainKinds(
   const GenericOperatorDomainKind lhs,
   const GenericOperatorDomainKind rhs)
{
   if (
      lhs == GenericOperatorDomainKind::Invalid ||
      rhs == GenericOperatorDomainKind::Invalid)
   {
      return GenericOperatorDomainKind::Invalid;
   }
   if (lhs == rhs)
   {
      return lhs;
   }
   return GenericOperatorDomainKind::Mixed;
}

template<GenericOperatorDomainKind First, GenericOperatorDomainKind... Rest>
consteval GenericOperatorDomainKind MergeDomainKinds()
{
   auto result = First;
   ((result = MergeDomainKinds(result, Rest)), ...);
   return result;
}

template<class Form, class WeakFormContext>
struct generic_operator_domain_kind
{
   static constexpr auto value = GenericOperatorDomainKind::Invalid;
};

// Dimensions retained by affine transforms still constrain domain admission,
// even if no numerical term remains on that domain.
template<class Domain, class WeakFormContext>
consteval GenericOperatorDomainKind DomainKindFor()
{
   using Context = std::remove_cvref_t<WeakFormContext>;
   if constexpr (!Context::template HasDomain<Domain::name>())
      return GenericOperatorDomainKind::Invalid;
   else
   {
      using IntegrationDomain = std::remove_cvref_t<decltype(
         std::declval<const Context&>().template GetDomain<Domain::name>().domain)>;
      if constexpr (is_mesh_integration_domain_v<IntegrationDomain>)
         return GenericOperatorDomainKind::Mesh;
      else if constexpr (is_partition_integration_domain_v<IntegrationDomain>)
         return GenericOperatorDomainKind::Partition;
      else
         return GenericOperatorDomainKind::Invalid;
   }
}

template<DomainExpr Domain, FieldExpr Expr, class WeakFormContext>
struct generic_operator_domain_kind<Integrand<Domain, Expr>, WeakFormContext>
{
   static constexpr auto value = DomainKindFor<Domain, WeakFormContext>();
};

template<class... Terms, class Context>
struct generic_operator_domain_kind<std::tuple<Terms...>, Context>
{
   static constexpr auto value = [] {
      if constexpr (sizeof...(Terms) == 0) return GenericOperatorDomainKind::Invalid;
      else return MergeDomainKinds<generic_operator_domain_kind<
         std::remove_cvref_t<Terms>, Context>::value...>();
   }();
};
template<class Terms, class Symbols, class Context>
struct generic_operator_domain_kind<SumFormExpr<Terms, Symbols>, Context>
{
   static constexpr auto value = []<class... Relations>(type_list<Relations...>) {
      auto kind = generic_operator_domain_kind<Terms, Context>::value;
      ((kind = MergeDomainKinds(kind,
         DomainKindFor<typename Relations::key_type, Context>())), ...);
      return kind;
   }(typename form_symbols_t<SumFormExpr<Terms, Symbols>>::declarations::retained_callable_input_relations{});
};

template<class WeakForm, class WeakFormContext>
inline constexpr auto generic_operator_domain_kind_v =
   generic_operator_domain_kind<
      std::remove_cvref_t<WeakForm>,
      std::remove_cvref_t<WeakFormContext>>::value;

} // namespace generic_operator_detail

template<class Space>
consteval bool SupportsComponentwiseNonconformingFacetTransform()
{
   using SpaceType = std::remove_cvref_t<Space>;
   using FE = typename SpaceType::finite_element_type;
   using ShapeFunctions = typename FE::shape_functions;
   using Restriction = typename SpaceType::restriction_type;

   if constexpr (is_vector_shape_functions_v<ShapeFunctions>)
   {
      return VectorElementDoFRestrictionForShapeFunctions<
         Restriction,
         ShapeFunctions>;
   }
   else
   {
      return TensorElementDoFRestrictionForShapeFunctions<
         Restriction,
         ShapeFunctions>;
   }
}

template<
   class FaceDomain,
   class TrialMinusSpace,
   class TrialPlusSpace,
   class TestMinusSpace,
   class TestPlusSpace>
consteval void ValidateNonconformingGlobalInteriorFacetTransformSupport()
{
   using FaceMesh = typename std::remove_cvref_t<FaceDomain>::face_mesh_type;
   using FaceInfo = typename FaceMesh::face_info_type;

   if constexpr (
      !FaceInfo::minus_side_type::is_conforming ||
      !FaceInfo::plus_side_type::is_conforming)
   {
      static_assert(
         SupportsComponentwiseNonconformingFacetTransform<TrialMinusSpace>() &&
         SupportsComponentwiseNonconformingFacetTransform<TrialPlusSpace>() &&
         SupportsComponentwiseNonconformingFacetTransform<TestMinusSpace>() &&
         SupportsComponentwiseNonconformingFacetTransform<TestPlusSpace>(),
         "Nonconforming global interior GenericOperator currently supports "
         "scalar and componentwise-vector value/gradient facet semantics on "
         "compatible tensor finite-element restrictions. Piola, H(div), "
         "H(curl), and de Rham-style nonconforming facet transforms require "
         "separate support and tests.");
   }
}

namespace generic_operator_detail {

template<SemanticSide Side, class Context, class Field>
consteval bool SupportsNonconformingFacetFieldTransform()
{
   using Binding = std::remove_cvref_t<decltype(
      std::declval<const Context&>().template GetFiniteElementField<Field::name>().space)>;
   if constexpr (Side == SemanticSide::Minus)
   {
      using Space = std::remove_cvref_t<decltype(
         GetMinusFacetFieldVolumeSpace(std::declval<const Binding&>()))>;
      return SupportsComponentwiseNonconformingFacetTransform<Space>();
   }
   else
   {
      using Space = std::remove_cvref_t<decltype(
         GetPlusFacetFieldVolumeSpace(std::declval<const Binding&>()))>;
      return SupportsComponentwiseNonconformingFacetTransform<Space>();
   }
}

// Keep the pack expansion outside nested lambdas: ROCm 6.3.1's Clang can crash
// while instantiating that pattern during nonconforming-facet validation.
template<SemanticSide Side, class FaceSide, class Context, class... Fields>
consteval void ValidateNonconformingFacetSideTransforms(type_list<Fields...>)
{
   if constexpr (!FaceSide::is_conforming)
   {
      static_assert(
         (SupportsNonconformingFacetFieldTransform<Side, Context, Fields>() && ...),
         "Nonconforming global interior GenericOperator currently supports "
         "scalar and componentwise-vector value/gradient facet semantics on "
         "compatible tensor finite-element restrictions. Piola, H(div), "
         "H(curl), and de Rham-style nonconforming facet transforms require "
         "separate support and tests.");
   }
}

} // namespace generic_operator_detail

// Check only fields actually evaluated on each nonconforming side. No singular
// trial/test pair is needed for loads or coupled residuals.
template<class FaceDomain, class Signature, class Context>
constexpr void ValidateNonconformingGlobalInteriorFacetTransformSupport()
{
   using FaceInfo = typename FaceDomain::face_mesh_type::face_info_type;
   using Minus = concat_many_t<typename Signature::template active_inputs_on<SemanticSide::Minus>,
      typename Signature::template bound_inputs_on<SemanticSide::Minus>,
      typename Signature::template test_outputs_on<SemanticSide::Minus>>;
   using Plus = concat_many_t<typename Signature::template active_inputs_on<SemanticSide::Plus>,
      typename Signature::template bound_inputs_on<SemanticSide::Plus>,
      typename Signature::template test_outputs_on<SemanticSide::Plus>>;
   generic_operator_detail::ValidateNonconformingFacetSideTransforms<
      SemanticSide::Minus, typename FaceInfo::minus_side_type, Context>(Minus{});
   generic_operator_detail::ValidateNonconformingFacetSideTransforms<
      SemanticSide::Plus, typename FaceInfo::plus_side_type, Context>(Plus{});
}

} // namespace gendil

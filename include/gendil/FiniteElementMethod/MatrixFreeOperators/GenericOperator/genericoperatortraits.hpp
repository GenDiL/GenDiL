// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp"
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

template<
   DomainExpr Domain,
   FieldExpr Expr,
   class WeakFormContext>
struct generic_operator_domain_kind<
   Integrand<Domain, Expr>,
   WeakFormContext>
{
private:
   using Context = std::remove_cvref_t<WeakFormContext>;
   static constexpr auto DomainName = Domain::name;

   static consteval GenericOperatorDomainKind Get()
   {
      if constexpr (!Context::template has_domain<DomainName>())
      {
         return GenericOperatorDomainKind::Invalid;
      }
      else
      {
         using IntegrationDomain =
            std::remove_cvref_t<
               decltype(
                  std::declval<const Context&>()
                     .template domain<DomainName>().domain)>;

         if constexpr (
            is_mesh_integration_domain_v<IntegrationDomain>)
         {
            return GenericOperatorDomainKind::Mesh;
         }
         else if constexpr (
            is_partition_integration_domain_v<IntegrationDomain>)
         {
            return GenericOperatorDomainKind::Partition;
         }
         else
         {
            return GenericOperatorDomainKind::Invalid;
         }
      }
   }

public:
   static constexpr auto value = Get();
};

template<class Key, class T, class WeakFormContext>
struct generic_operator_domain_kind<
   Entry<Key, T>,
   WeakFormContext>
   : generic_operator_domain_kind<T, WeakFormContext> {};

template<class WeakFormContext>
struct generic_operator_domain_kind<
   StaticMap<>,
   WeakFormContext>
{
   static constexpr auto value = GenericOperatorDomainKind::Invalid;
};

template<
   class WeakFormContext,
   class FirstEntry,
   class... RestEntries>
struct generic_operator_domain_kind<
   StaticMap<FirstEntry, RestEntries...>,
   WeakFormContext>
{
   static constexpr auto value =
      MergeDomainKinds<
         generic_operator_domain_kind<
            FirstEntry,
            WeakFormContext>::value,
         generic_operator_domain_kind<
            RestEntries,
            WeakFormContext>::value...>();
};

template<class Map, class WeakFormContext>
struct generic_operator_domain_kind<
   SumFormExpr<Map>,
   WeakFormContext>
   : generic_operator_domain_kind<Map, WeakFormContext> {};

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

} // namespace gendil

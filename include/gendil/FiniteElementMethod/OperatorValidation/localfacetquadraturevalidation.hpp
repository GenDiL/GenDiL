// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/domainiteration.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/faceloop.hpp"
#include "gendil/Meshes/Connectivities/Orientations/tensorproductorientation.hpp"
#include "gendil/NumericalIntegration/integrationrule.hpp"
#include "gendil/Utilities/debug.hpp"

#include <array>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil::local_facet_quadrature_detail
{

template<class Points> struct IsSymmetricBuiltin : std::false_type {};
template<Integer N>
struct IsSymmetricBuiltin<GaussLegendrePoints<N>> : std::true_type {};
template<int N>
struct IsSymmetricBuiltin<GaussLobattoLegendrePoints<N>> : std::true_type {};

template<class PointsTuple> struct IsUniformSymmetricBuiltin : std::false_type {};
template<class Head, class... Tail>
struct IsUniformSymmetricBuiltin<std::tuple<Head, Tail...>>
   : std::bool_constant<IsSymmetricBuiltin<Head>::value &&
                        (std::is_same_v<Head, Tail> && ...)> {};

constexpr Real Abs(Real value) { return value < Real{0} ? -value : value; }

constexpr bool SameQuadratureNumber(Real lhs, Real rhs)
{
   const Real a = Abs(lhs), b = Abs(rhs);
   if (!(a <= std::numeric_limits<Real>::max()) ||
       !(b <= std::numeric_limits<Real>::max()))
      return false;
   const Real scale = a > b ? (a > Real{1} ? a : Real{1})
                            : (b > Real{1} ? b : Real{1});
   return Abs(lhs - rhs) <= Real{64} * std::numeric_limits<Real>::epsilon() * scale;
}

// Match weighted point sets, not their C++ types or enumeration order.
template<class NativePoints, class CurrentPoints>
constexpr bool PointSetsMatch(bool reflected)
{
   constexpr Integer n = NativePoints::GetNumPoints();
   if constexpr (n != CurrentPoints::GetNumPoints())
      return false;
   else
   {
      std::array<bool, n> used{};
      for (Integer native = 0; native < n; ++native)
      {
         const Real x = NativePoints::GetCoord(native);
         const Real mapped = reflected ? Real{1} - x : x;
         bool found = false;
         for (Integer current = 0; current < n; ++current)
         {
            if (!used[current] &&
                SameQuadratureNumber(mapped, CurrentPoints::GetCoord(current)) &&
                SameQuadratureNumber(NativePoints::GetWeight(native),
                                     CurrentPoints::GetWeight(current)))
            {
               used[current] = true;
               found = true;
               break;
            }
         }
         if (!found) return false;
      }
      return true;
   }
}

// Compare each axis pair once; connectivity validation then uses table lookups.
template<Integer Dim>
using QuadratureCompatibility =
   std::array<std::array<std::array<bool, 2>, Dim>, Dim>;

template<class Rule>
constexpr auto MakeQuadratureCompatibility()
{
   using PointSets = typename Rule::points::points_1d_tuple;
   QuadratureCompatibility<Rule::space_dim> result{};
   ConstexprLoop<Rule::space_dim>([&](auto native_axis)
   {
      using NativePoints = std::tuple_element_t<decltype(native_axis)::value, PointSets>;
      ConstexprLoop<Rule::space_dim>([&](auto current_axis)
      {
         using CurrentPoints = std::tuple_element_t<decltype(current_axis)::value, PointSets>;
         result[native_axis][current_axis][0] = PointSetsMatch<NativePoints, CurrentPoints>(false);
         result[native_axis][current_axis][1] = PointSetsMatch<NativePoints, CurrentPoints>(true);
      });
   });
   return result;
}

template<auto Value> struct ConstantValue {};

template<class Rule>
inline constexpr bool has_constexpr_quadrature_compatibility_v = requires
{
   typename ConstantValue<MakeQuadratureCompatibility<Rule>()>;
};

template<class Rule, class Function>
void WithQuadratureCompatibility(Function&& function)
{
   if constexpr (has_constexpr_quadrature_compatibility_v<Rule>)
   {
      constexpr auto compatibility = MakeQuadratureCompatibility<Rule>();
      function(compatibility);
   }
   else
   {
      // Custom point providers need not offer constexpr coordinate/weight access.
      // Recompute on each explicit validation call, never cache mutable inputs.
      const auto compatibility = MakeQuadratureCompatibility<Rule>();
      function(compatibility);
   }
}

// Conservatively enumerate the axis maps permitted by an orientation type.
// Product orientations cannot move an axis between independent factors.
template<class Orientation>
constexpr bool OrientationAllowsAxisMap(Integer native, Integer current, bool reflected)
{
   if constexpr (is_static_identity_orientation_v<Orientation>)
      return native == current && !reflected;
   else if constexpr (is_tensor_product_orientation_v<Orientation>)
   {
      bool allowed = false;
      ConstexprLoop<Orientation::num_components>([&](auto component)
      {
         using Component = std::tuple_element_t<decltype(component)::value,
            typename Orientation::component_types>;
         constexpr Integer offset = detail::orientation_component_offset_v<
            decltype(component)::value, typename Orientation::component_types>;
         constexpr Integer end = offset + orientation_dimension_v<Component>;
         if (native >= offset && native < end && current >= offset && current < end)
            allowed = OrientationAllowsAxisMap<Component>(native - offset, current - offset, reflected);
      });
      return allowed;
   }
   else
      return true;
}

template<class Rule, class Face>
constexpr bool FaceTypePreservesQuadrature(const QuadratureCompatibility<Rule::space_dim>& compatibility)
{
   using Side = typename Face::current_side_type;
   using Orientation = typename Face::neighbor_side_type::orientation_type;
   constexpr Integer normal_axis = Side::geometry::GetNormalDimensionIndex(
      Side::local_face_index_type::value);
   for (Integer native = 0; native < Rule::space_dim; ++native)
      for (Integer current = 0; current < Rule::space_dim; ++current)
         if (current != normal_axis)
            for (Integer reflected = 0; reflected < 2; ++reflected)
               if (OrientationAllowsAxisMap<Orientation>(native, current, reflected != 0) &&
                   !compatibility[native][current][reflected])
                  return false;
   return true;
}

template<class Rule, class Face>
constexpr bool FaceQuadratureIsCompatible(const Face& face,
   const QuadratureCompatibility<Rule::space_dim>& compatibility)
{
   using Side = typename Face::current_side_type;
   using Orientation = typename Face::neighbor_side_type::orientation_type;
   using PointSets = typename Rule::points::points_1d_tuple;
   constexpr Integer dim = Rule::space_dim;
   constexpr Integer normal_axis = Side::geometry::GetNormalDimensionIndex(
      Side::local_face_index_type::value);
   if constexpr (is_static_identity_orientation_v<Orientation> ||
                 IsUniformSymmetricBuiltin<PointSets>::value)
      return true;
   else
   {
      const auto& orientation = face.NeighborSide().GetOrientation();
      if (!IsValidOrientation(orientation)) return false;
      bool compatible = true;
      ConstexprLoop<dim>([&](auto native_axis)
      {
         const LocalIndex mapped = detail::GetStructuredOrientationAxis<
            decltype(native_axis)::value>(orientation);
         const Integer current_axis = mapped > 0 ? mapped - 1 : -mapped - 1;
         if (current_axis != normal_axis)
            compatible = compatible && compatibility[native_axis][current_axis][mapped < 0];
      });
      return compatible;
   }
}

template<class Rule, class Face>
constexpr bool FaceQuadratureIsCompatible(const Face& face)
{
   using Orientation = typename Face::neighbor_side_type::orientation_type;
   using PointSets = typename Rule::points::points_1d_tuple;
   if constexpr (is_static_identity_orientation_v<Orientation> ||
                 IsUniformSymmetricBuiltin<PointSets>::value)
      return true;
   else
      return FaceQuadratureIsCompatible<Rule>(face, MakeQuadratureCompatibility<Rule>());
}

template<class Mesh, size_t... Face>
consteval bool MeshHasStaticIdentityOrientations(std::index_sequence<Face...>)
{
   return (is_static_identity_orientation_v<typename std::remove_cvref_t<decltype(
      mesh::GetLocalFaceInfo(std::declval<const Mesh&>(), GlobalIndex{},
         std::integral_constant<Integer, Face>{}))>::neighbor_side_type::orientation_type> && ...);
}

template<class Rule, class Mesh, size_t... Face>
constexpr bool MeshTypesPreserveQuadrature(
   const QuadratureCompatibility<Rule::space_dim>& compatibility, std::index_sequence<Face...>)
{
   return (FaceTypePreservesQuadrature<Rule, std::remove_cvref_t<decltype(
      mesh::GetLocalFaceInfo(std::declval<const Mesh&>(), GlobalIndex{},
         std::integral_constant<Integer, Face>{}))>>(compatibility) && ...);
}

template<class Rule, class Mesh>
void ValidateMesh(const Mesh& mesh, const QuadratureCompatibility<Rule::space_dim>& compatibility)
{
   using PointSets = typename Rule::points::points_1d_tuple;
   using Faces = std::make_index_sequence<mesh::mesh_geometry_t<Mesh>::num_faces>;
   if constexpr (!IsUniformSymmetricBuiltin<PointSets>::value &&
                 !MeshHasStaticIdentityOrientations<Mesh>(Faces{}))
   {
      // A type-level proof avoids reading connectivity, including for custom
      // symmetric rules and product meshes with independent anisotropic factors.
      constexpr bool statically_compatible = []
      {
         if constexpr (has_constexpr_quadrature_compatibility_v<Rule>)
            return MeshTypesPreserveQuadrature<Rule, Mesh>(
               MakeQuadratureCompatibility<Rule>(), Faces{});
         else
            return false;
      }();
      if constexpr (!statically_compatible)
      {
         if constexpr (!has_constexpr_quadrature_compatibility_v<Rule>)
            if (MeshTypesPreserveQuadrature<Rule, Mesh>(compatibility, Faces{}))
               return;

         for (GlobalIndex cell = 0; cell < mesh.GetNumberOfCells(); ++cell)
            InteriorFaceLoop(mesh, cell, [&](const auto& face)
            {
               GENDIL_VERIFY(FaceQuadratureIsCompatible<Rule>(face, compatibility),
                  "Local interior-facet quadrature is incompatible with the face coordinate "
                  "orientation. Use matching weighted point sets on permuted/reversed "
                  "tangential axes, or use global facet execution.");
            });
      }
   }
}

// Host-only: device GENDIL_VERIFY is intentionally disabled. Connectivity must
// be host-readable whenever compatibility cannot be proved from its type.
template<class Rule, class Mesh>
void ValidateMesh(const Mesh& mesh)
{
   using PointSets = typename Rule::points::points_1d_tuple;
   if constexpr (!IsUniformSymmetricBuiltin<PointSets>::value &&
                 !MeshHasStaticIdentityOrientations<Mesh>(
                    std::make_index_sequence<mesh::mesh_geometry_t<Mesh>::num_faces>{}))
      WithQuadratureCompatibility<Rule>([&](const auto& compatibility)
      {
         ValidateMesh<Rule>(mesh, compatibility);
      });
}

template<class Form, class Context, class Rule>
void ValidateLocalFacetQuadrature(const Form& form, const Context& context, const Rule&)
{
   static_assert(FormExpr<Form>, "ValidateOperator requires a weak-form expression.");
   if constexpr (!is_zero<Form> && has_interior_facet_contributions_v<Form>)
   {
      constexpr auto kind = generic_operator_detail::generic_operator_domain_kind_v<Form, Context>;
      static_assert(kind == generic_operator_detail::GenericOperatorDomainKind::Mesh ||
                    kind == generic_operator_detail::GenericOperatorDomainKind::Partition,
         "Local facet quadrature validation requires uniformly mesh or partition integration domains.");
      if constexpr (kind == generic_operator_detail::GenericOperatorDomainKind::Mesh)
      {
         using PointSets = typename Rule::points::points_1d_tuple;
         if constexpr (!IsUniformSymmetricBuiltin<PointSets>::value)
            WithQuadratureCompatibility<Rule>([&](const auto& compatibility)
            {
               ForEachGenericOperatorDomainName(generic_operator_domain_names_t<Form>{}, [&](auto name)
               {
                  constexpr auto Name = decltype(name)::name;
                  using Terms = decltype(weak_form_execution_detail::SelectIntegrands<
                     Name, SemanticEntity::InteriorFacet>(form));
                  if constexpr (std::tuple_size_v<Terms> != 0)
                     ForEachCellExecutionDomain(context, Cells<Name>{}, [&](const auto& batch)
                     {
                        ValidateMesh<Rule>(batch.GetCellMesh(), compatibility);
                     });
               });
            });
      }
   }
}

} // namespace gendil::local_facet_quadrature_detail

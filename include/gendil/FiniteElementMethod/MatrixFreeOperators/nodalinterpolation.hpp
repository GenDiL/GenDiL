// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Analytic interpolation into current tensor-product nodal elements.
 */

#include "gendil/Algebra/accessors.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/Algebra/vectoraccess.hpp"
#include "gendil/Algebra/vectoroperations.hpp"
#include "gendil/FiniteElementMethod/componentfiniteelementspace.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionunitweight.hpp"
#include "gendil/NumericalIntegration/integrationrule.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"
#include "gendil/Utilities/multiindex.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace gendil {

namespace nodal_interpolation_detail {

template<class FiniteElement, class Indices>
struct NodalIntegrationRule;

template<class FiniteElement, size_t... Axis>
struct NodalIntegrationRule<FiniteElement, std::index_sequence<Axis...>>
{
   using ShapeFunctions = typename FiniteElement::shape_functions;
   using ScalarShapeFunctions =
      typename ShapeFunctions::shape_functions_1d_tuple;
   using type = IntegrationRule<
      typename FiniteElement::geometry,
      TensorPoints<
         typename std::tuple_element_t<
            Axis,
            ScalarShapeFunctions>::points...>>;
};

template<class Space>
consteval bool IsVectorSpace()
{
   if constexpr (is_mixed_finite_element_space_v<Space>)
   {
      static_assert(
         std::remove_cvref_t<Space>::num_cell_spaces > 0,
         "Nodal interpolation requires at least one mixed cell space." );
      using First = std::remove_cvref_t<decltype(
         std::declval<const Space&>()
            .template GetCellFiniteElementSpace<0>())>;
      return is_vector_finite_element_space_v<First>;
   }
   else
   {
      return is_vector_finite_element_space_v<Space>;
   }
}

template<class CellSpace, class Visitor>
void ForEachLocalNodalRow(
   const CellSpace& cell_space,
   const GlobalIndex ordinal_offset,
   Visitor&& visitor)
{
   using FiniteElement = typename CellSpace::finite_element_type;
   using ShapeFunctions = typename FiniteElement::shape_functions;
   using Restriction = typename CellSpace::restriction_type;
   static_assert(
      is_tensor_shape_functions_v<ShapeFunctions>,
      "InterpolateNodalField requires scalar TensorShapeFunctions after component selection." );
   static_assert(
      static_restriction_entry_count_v<Restriction> == 1 &&
         restriction_supports_element_reference_view_v<Restriction>,
      "InterpolateNodalField requires one-entry unit-weight nodal restrictions." );

   using DofShape = typename ShapeFunctions::dof_shape;
   constexpr GlobalIndex local_dofs = Product(DofShape{});
   for (GlobalIndex element = 0;
        element < static_cast<GlobalIndex>(
           cell_space.GetNumberOfFiniteElements());
        ++element)
   {
      for (GlobalIndex local = 0; local < local_dofs; ++local)
      {
         const auto local_dof = UnflattenMultiIndex<DofShape>(local);
         const GlobalIndex ordinal = ordinal_offset + element * local_dofs + local;
         ForEachRestrictionEntry(
            GetRestriction(cell_space),
            element,
            local_dof,
            [&] GENDIL_HOST_DEVICE (
               const GlobalIndex global,
               const auto& weight)
            {
               static_assert(
                  std::is_same_v<
                     std::remove_cvref_t<decltype(weight)>,
                     RestrictionUnitWeight>,
                  "Nodal interpolation rejects weighted or oriented restriction rows." );
               visitor(element, local_dof, ordinal, global);
            });
      }
   }
}

template<class CellSpace, class Function>
GENDIL_HOST_DEVICE
Real EvaluateNode(
   const CellSpace& cell_space,
   Function& function,
   const GlobalIndex element,
   const auto& local_dof)
{
   using FiniteElement = typename CellSpace::finite_element_type;
   using ShapeFunctions = typename FiniteElement::shape_functions;
   constexpr Integer Dim = ShapeFunctions::dim;
   using Rule = typename NodalIntegrationRule<
      FiniteElement,
      std::make_index_sequence<Dim>>::type;
   using Cell = typename CellSpace::mesh_type::cell_type;
   using QuadData = typename Cell::template QuadData<Rule>;
   TensorIndex<Dim> nodal_index;
   for (Integer axis = 0; axis < Dim; ++axis)
   {
      nodal_index[axis] = local_dof[axis];
   }
   typename Cell::physical_coordinates coordinates{};
   typename Cell::jacobian jacobian{};
   cell_space.GetCell(element).GetValuesAndJacobian(
      nodal_index,
      QuadData{},
      coordinates,
      jacobian);
   return function(coordinates);
}

template<class Space, size_t... Part>
consteval bool MixedSpaceMayShare(std::index_sequence<Part...>)
{
   return (
      restriction_may_share_global_dofs_v<
         typename std::remove_cvref_t<decltype(
            std::declval<const Space&>()
               .template GetCellFiniteElementSpace<Part>())>::restriction_type> ||
      ...);
}

template<class Space>
consteval bool SpaceMayShare()
{
   if constexpr (is_mixed_finite_element_space_v<Space>)
   {
      return MixedSpaceMayShare<Space>(
         std::make_index_sequence<
            std::remove_cvref_t<Space>::num_cell_spaces>{});
   }
   else
   {
      return restriction_may_share_global_dofs_v<
         typename std::remove_cvref_t<Space>::restriction_type>;
   }
}

template<class KernelPolicy,
         class CellSpace,
         class Function,
         class OutputVector>
void InterpolateNonsharingCellSpace(
   const CellSpace& cell_space,
   Function function,
   OutputVector& output)
{
   using FiniteElement = typename CellSpace::finite_element_type;
   using ShapeFunctions = typename FiniteElement::shape_functions;
   using Restriction = typename CellSpace::restriction_type;
   static_assert(
      !restriction_may_share_global_dofs_v<Restriction>,
      "Direct nodal interpolation requires a nonsharing restriction." );
   static_assert(
      is_tensor_shape_functions_v<ShapeFunctions>,
      "InterpolateNodalField requires scalar TensorShapeFunctions after component selection." );
   static_assert(
      static_restriction_entry_count_v<Restriction> == 1 &&
         restriction_supports_element_reference_view_v<Restriction>,
      "InterpolateNodalField requires one-entry unit-weight nodal restrictions." );

   using DofShape = typename ShapeFunctions::dof_shape;
   constexpr GlobalIndex local_dofs = Product(DofShape{});
   const GlobalIndex num_elements =
      static_cast<GlobalIndex>(cell_space.GetNumberOfFiniteElements());
   const GlobalIndex num_rows = CheckedMultiply(
      num_elements,
      local_dofs,
      "Nodal interpolation row count overflow." );
   constexpr bool OnDevice = is_device_configuration_v<KernelPolicy>;
   auto values = WriteKernelVector<OnDevice>(output);
   const auto cell_space_copy = cell_space;
   KernelLoop<OnDevice>(
      num_rows,
      [=] GENDIL_HOST_DEVICE (const GlobalIndex row) mutable
      {
         const GlobalIndex element = row / local_dofs;
         const auto local_dof =
            UnflattenMultiIndex<DofShape>(row % local_dofs);
         ForEachRestrictionEntry(
            GetRestriction(cell_space_copy),
            element,
            local_dof,
            [&] GENDIL_HOST_DEVICE (
               const GlobalIndex global,
               const auto& weight)
            {
               static_assert(
                  std::is_same_v<
                     std::remove_cvref_t<decltype(weight)>,
                     RestrictionUnitWeight>,
                  "Nodal interpolation rejects weighted or oriented restriction rows." );
               values[global] = EvaluateNode(
                  cell_space_copy, function, element, local_dof);
            });
      });
}

template<class KernelPolicy,
         class Space,
         class Function,
         class OutputVector>
void InterpolateScalar(
   const Space& space,
   Function function,
   OutputVector& output)
{
   if constexpr (!SpaceMayShare<Space>())
   {
      ForEachHomogeneousSpace(
         space,
         [&](const auto& cell_space)
         {
            InterpolateNonsharingCellSpace<KernelPolicy>(
               cell_space, function, output);
         });
      return;
   }

   const size_t extent = GetVectorSize(output);
   std::vector<GlobalIndex> representative(
      extent,
      std::numeric_limits<GlobalIndex>::max());
   GlobalIndex ordinal_offset = 0;
   ForEachHomogeneousSpace(
      space,
      [&](const auto& cell_space)
      {
         ForEachLocalNodalRow(
            cell_space,
            ordinal_offset,
            [&](const GlobalIndex,
                const auto&,
                const GlobalIndex ordinal,
                const GlobalIndex global)
            {
               GENDIL_VERIFY(
                  global < extent,
                  "Nodal interpolation produced an out-of-range global DoF." );
               representative[global] = std::min(
                  representative[global], ordinal);
            });
         ordinal_offset = CheckedAdd(
            ordinal_offset,
            GetNumberOfLocalDofs(cell_space),
            "Nodal interpolation row ordinal overflow." );
      });

   Real* values = ReadWriteHostVector(output);
   ordinal_offset = 0;
   ForEachHomogeneousSpace(
      space,
      [&](const auto& cell_space)
      {
         ForEachLocalNodalRow(
            cell_space,
            ordinal_offset,
            [&](const GlobalIndex element,
                const auto& local_dof,
                const GlobalIndex ordinal,
                const GlobalIndex global)
            {
               if (representative[global] == ordinal)
               {
                  values[global] = EvaluateNode(
                     cell_space, function, element, local_dof);
               }
#ifndef NDEBUG
               else
               {
                  const Real proposal = EvaluateNode(
                     cell_space, function, element, local_dof);
                  GENDIL_VERIFY(
                     std::abs(values[global] - proposal) <=
                        1.0e-12 * (1.0 + std::abs(proposal)),
                     "Repeated nodal interpolation proposals disagree." );
               }
#endif
            });
         ordinal_offset = CheckedAdd(
            ordinal_offset,
            GetNumberOfLocalDofs(cell_space),
            "Nodal interpolation row ordinal overflow." );
      });
}

} // namespace nodal_interpolation_detail

/**
 * @brief Interpolate an analytic scalar or static-vector nodal field.
 *
 * The operation supports homogeneous and mixed spaces. Shared H1 values are
 * written once through a deterministic representative row; debug builds also
 * verify that every duplicate coordinate proposes the same value.
 */
template<class KernelPolicy,
         class Space,
         class Function,
         class OutputVector>
void InterpolateNodalField(
   const Space& space,
   Function function,
   OutputVector& output)
{
   (void)sizeof(KernelPolicy);
   GENDIL_VERIFY(
      GetVectorSize(output) ==
         static_cast<size_t>(GetAlgebraicDofExtent(space)),
      "Nodal-interpolation output extent does not match the field space." );
   Zero<false>(output);
   if constexpr (nodal_interpolation_detail::IsVectorSpace<Space>())
   {
      ForEachComponentFiniteElementSpace(
         space,
         [&]<size_t Component>(
            std::integral_constant<size_t, Component>,
            const auto& component_space)
         {
            auto scalar_function = [function] GENDIL_HOST_DEVICE (
               const auto& coordinates) mutable
            {
               const auto value = function(coordinates);
               return static_cast<Real>(vector_access(value, Component));
            };
            nodal_interpolation_detail::InterpolateScalar<KernelPolicy>(
               component_space,
               scalar_function,
               output);
         });
   }
   else
   {
      nodal_interpolation_detail::InterpolateScalar<KernelPolicy>(
         space,
         std::move(function),
         output);
   }
}

/** @brief Allocate and interpolate an analytic nodal field. */
template<class KernelPolicy, class Space, class Function>
Vector InterpolateNodalField(const Space& space, Function function)
{
   Vector output(GetAlgebraicDofExtent(space));
   InterpolateNodalField<KernelPolicy>(
      space,
      std::move(function),
      output);
   return output;
}

} // namespace gendil

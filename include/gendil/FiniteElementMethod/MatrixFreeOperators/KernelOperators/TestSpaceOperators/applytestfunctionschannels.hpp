// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/weakformtraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applyvaluesandgradienttestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatefields.hpp"

namespace gendil {

// Apply a contribution to output DOFs. Element overlays update the caller's row
// directly; restricted destinations use a zeroed row and one additive scatter.
template<class Kernel, class Endpoint, class Entity, class Fn>
GENDIL_HOST_DEVICE
void ApplyAddToOutputDofs(Kernel& kernel, Endpoint& endpoint, const Entity& entity, Fn&& apply)
{
   if constexpr (std::remove_cvref_t<Endpoint>::is_local)
      apply(endpoint.dofs);
   else
   {
      const auto& space = GetMinusFacetFieldVolumeSpace(endpoint.space);
      using Row = decltype(ReadDofs(kernel, space, entity.element_index, endpoint.dofs));
      Row row{};
      apply(row);
      WriteAddDofs(kernel, space, entity.element_index, row, endpoint.dofs);
   }
}
template<class Kernel, class Endpoint, class Side, class Space, class QData, class Fn>
GENDIL_HOST_DEVICE
void ApplyAddToOutputDofs(Kernel& kernel, Endpoint& endpoint,
   const FacetFieldSideBinding<Side, Space, QData>& binding, Fn&& apply)
{
   if constexpr (std::remove_cvref_t<Endpoint>::is_local)
      apply(endpoint.dofs);
   else
   {
      auto scatter = [&](auto& view)
      {
         using Row = decltype(ReadDofs(kernel, binding.volume_space, binding.side, view));
         Row row{};
         apply(row);
         WriteAddDofs(kernel, binding.volume_space, binding.side, row, view);
      };
      if constexpr (is_face_field_binding_v<decltype(endpoint.space)> &&
                    !execution_context_detail::IsPreboundElementStorage<
                       std::remove_cvref_t<decltype(endpoint.dofs)>>())
      {
         auto view = MakeElementTensorView(binding.volume_space, endpoint.dofs);
         scatter(view);
      }
      else scatter(endpoint.dofs);
   }
}

// Facet contractions are shared by current-row and canonical two-row delivery.
// The caller chooses the side and owns the destination row.
template<class Kernel, class Side, class QData, class Values, class Gradients, class Row>
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(Kernel& kernel, const Side& side, const QData& qdata,
   const InterpolatedField<Values, Gradients>& contributions, Row& row)
{
   constexpr bool has_values = !std::is_same_v<Values, Empty>;
   constexpr bool has_gradients = !std::is_same_v<Gradients, Empty>;
   if constexpr (has_values && has_gradients)
      ApplyValuesAndGradientTestFunctions<true>(kernel, side, qdata,
         contributions.values, contributions.gradients, row);
   else if constexpr (has_values)
      ApplyAddTestFunctions(kernel, side, qdata, contributions.values, row);
   else if constexpr (has_gradients)
      ApplyGradientTestFunctions<true>(kernel, side, qdata, contributions.gradients, row);
}

template<class Kernel, class Context, class OperatorContext, class Entity, class Integrand, class Channels>
   requires CellIntegrand<Integrand>
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(Kernel& kernel, Context& context, const OperatorContext& operator_context,
   const Entity& entity, const Integrand&, const Channels& channels)
{
   using Signature = typename OperatorContext::template signature_for<Integrand>;
   using Output = weak_form_traits::only_field_t<typename Signature::test_outputs>;
   auto& endpoint = context.template GetWriteField<Output::name>();
   const auto& qdata = operator_context.template GetFiniteElementQuadData<Output::name>();
   ApplyAddToOutputDofs(kernel, endpoint, entity, [&](auto& row)
   {
      if constexpr (need_values(Output::mask) && need_gradients(Output::mask))
         ApplyValuesAndGradientTestFunctions<true>(kernel, qdata, channels.values, channels.gradients, row);
      else if constexpr (need_values(Output::mask))
         ApplyAddTestFunctions(kernel, qdata, channels.values, row);
      else
      {
         auto values = ApplyGradientTestFunctionsAtQPoints(kernel, qdata, channels.gradients);
         ApplyAddTestFunctions(kernel, qdata, values, row);
      }
   });
}

template<class Kernel, class Context, class OperatorContext, class Entity, class Face, class Integrand, class Channels>
   requires (BoundaryFacetIntegrand<Integrand> || InteriorFacetIntegrand<Integrand>)
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(Kernel& kernel, Context& context, const OperatorContext& operator_context,
   const Entity& entity, const Face& face, const Integrand&, const Channels& channels)
{
   constexpr auto TestName = weak_form_traits::only_test_field_name_v< Integrand >;
   auto& endpoint = context.template GetWriteField<TestName>();
   const auto binding = MakeMinusFacetFieldBinding(face, endpoint.space,
      operator_context.template GetFiniteElementFacetQuadData<TestName>());
   const auto& qdata = binding.qdata;
   auto apply = [&](auto& row)
   {
      ApplyAddTestFunctions(kernel, binding.side, qdata, channels, row);
   };
   if constexpr (BoundaryFacetIntegrand<Integrand> &&
                 !is_face_field_binding_v<decltype(endpoint.space)>)
      ApplyAddToOutputDofs(kernel, endpoint, entity, apply);
   else ApplyAddToOutputDofs(kernel, endpoint, binding, apply);
}

// Deliver one canonical facet side. Empty contributions require no output
// binding, quadrature data, local row, or scatter.
template <
   SemanticSide Side,
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename FaceInfo,
   typename Integrand,
   typename Values,
   typename Gradients >
   requires ( InteriorFacetIntegrand<Integrand> &&
              (Side == SemanticSide::Minus || Side == SemanticSide::Plus) )
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(
   KernelContext & kernel_context,
   ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const FaceInfo & face_info,
   const Integrand & /*integrand*/,
   const InterpolatedField<Values, Gradients> & contributions )
{
   if constexpr ( !std::is_same_v<Values, Empty> || !std::is_same_v<Gradients, Empty> )
   {
      constexpr auto TestName = weak_form_traits::only_test_field_name_v< Integrand >;
      auto& output_field = execution_context.template GetWriteField<TestName>();
      const auto& finite_element_quad_data =
         operator_context.template GetFiniteElementFacetQuadData<TestName>();

      const auto binding = [&]()
      {
         if constexpr ( Side == SemanticSide::Minus )
         {
            return MakeMinusFacetFieldBinding(
               face_info,
               output_field.space,
               finite_element_quad_data);
         }
         else
         {
            return MakePlusFacetFieldBinding(
               face_info,
               output_field.space,
               finite_element_quad_data);
         }
      }();

      ApplyAddToOutputDofs(
         kernel_context,
         output_field,
         binding,
         [&] (auto& output_row)
         {
            ApplyAddTestFunctions(
               kernel_context,
               binding.side,
               binding.qdata,
               contributions,
               output_row);
         });
   }
}

} // namespace gendil

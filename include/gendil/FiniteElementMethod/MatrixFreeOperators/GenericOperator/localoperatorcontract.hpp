// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/fieldrequirements.hpp"

#include <type_traits>

namespace gendil
{

// Specializations provide requirements, Prepare<Policy>, and Apply. The latter
// two own their numerical preparation and local algorithm. This contract does
// not grant matrix-assembly capability or allocate/rebind field storage.
template<class LocalOperator>
struct local_operator_contract;

template<class LocalOperator>
using local_operator_requirements_t =
   typename local_operator_contract<std::remove_cvref_t<LocalOperator>>::requirements;

// Resource requirements are resolved after binding, because local overlays and
// vector-backed endpoints can require different staging. Native local operators
// supply this contract without any weak-form or quadrature dependency.
template<class LocalOperator, class Execution, class OperatorContext, class Policy>
using local_operator_resources_t = typename local_operator_contract<
   std::remove_cvref_t<LocalOperator>>::template resources<Execution, OperatorContext, Policy>;

// Preparation uses selected spaces/geometry and local operator parameters,
// never the identity of an application's input/output buffers. The operator
// context can be reused across local buffers or assembly probe columns.
template<class Policy, class LocalOperator, class SelectedContext, class Parameters>
   requires requires(const LocalOperator& local_operator, const SelectedContext& context,
                     const Parameters& parameters)
   {
      local_operator_contract<std::remove_cvref_t<LocalOperator>>::template Prepare<Policy>(
         local_operator, context, parameters);
   }
constexpr decltype(auto) PrepareLocalOperator(
   const LocalOperator& local_operator, const SelectedContext& context,
   const Parameters& parameters)
{
   return local_operator_contract<std::remove_cvref_t<LocalOperator>>::template Prepare<Policy>(
      local_operator, context, parameters);
}

// The caller owns traversal, local storage, and output delivery. In particular,
// this dispatch does not insert an implicit interpolation or quadrature loop.
template<class LocalOperator, class Kernel, class OperatorContext, class Entity, class LocalData>
   requires requires(const LocalOperator& local_operator, Kernel& kernel,
                     const OperatorContext& operator_context, const Entity& entity,
                     LocalData& local_data)
   {
      local_operator_contract<std::remove_cvref_t<LocalOperator>>::Apply(
         local_operator, kernel, operator_context, entity, local_data);
   }
GENDIL_HOST_DEVICE
constexpr decltype(auto) ApplyLocalOperator(
   const LocalOperator& local_operator, Kernel& kernel, const OperatorContext& operator_context,
   const Entity& entity, LocalData& local_data)
{
   return local_operator_contract<std::remove_cvref_t<LocalOperator>>::Apply(
      local_operator, kernel, operator_context, entity, local_data);
}

} // namespace gendil

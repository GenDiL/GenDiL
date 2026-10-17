// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/KernelContext/batchingeligibility.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/CellOperator/celloperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/domainiteration.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/localoperatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/LocalFacetOperator/localinteriorfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/BoundaryFacetOperator/boundaryfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/restrictedweakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/readdofs.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/writedofs.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumformexpr.hpp"

namespace gendil {

namespace local_facet_operator_detail {

template<class Field>
consteval auto OnlyFieldName(type_list<Field>) { return Field::name; }

template<class Requirements, class Policy, class Terms, class Execution, class OperatorContext, class Entity>
consteval size_t BatchSharedSlots()
{
   if constexpr (Requirements::can_reuse_current_element_dofs)
   {
      constexpr auto Input = OnlyFieldName(typename Requirements::declared_inputs{});
      constexpr auto Output = OnlyFieldName(typename Requirements::declared_outputs{});
      using Kernel = KernelContext<Policy, 0>;
      using Source = decltype(std::declval<const Execution&>().template GetReadField<Input>());
      using Sink = std::remove_cvref_t<decltype(std::declval<Execution&>().template GetWriteField<Output>())>;
      using InputRow = std::remove_cvref_t<decltype(ReadFieldDofs(std::declval<Kernel&>(),
         std::declval<Source>(), std::declval<const Entity&>()))>;
      using OutputRow = decltype(ReadDofs(std::declval<Kernel&>(),
         std::declval<const Sink&>().space, GlobalIndex{}, std::declval<Sink&>().dofs));
      using Local = decltype(BindElementDofs(std::declval<const Terms&>(),
         std::declval<const Execution&>(), std::declval<const OperatorContext&>(),
         ReadLocal<Input>(std::declval<const InputRow&>()), AddLocal<Output>(std::declval<OutputRow&>())));
      return detail::TermSharedSlots<Policy, Local, OperatorContext, Execution>(static_cast<Terms*>(nullptr));
   }
   else return detail::TermSharedSlots<Policy, Execution, OperatorContext>(static_cast<Terms*>(nullptr));
}

template<class Term, class Kernel, class Context, class NeighborContext, class OperatorContext, class Entity, class Mesh>
GENDIL_HOST_DEVICE
void ApplyTerm(Kernel& kernel, Context& context, const NeighborContext& neighbor_context,
               const OperatorContext& operator_context,
               const Entity& element, const Mesh& mesh, const Term& integrand)
{
   if constexpr (BoundaryFacetIntegrand<Term>)
      BoundaryFaceLoop(mesh, element.element_index, [&](const auto& face) {
         BoundaryFacetIntegrandOperator(kernel, context, operator_context, element, face, integrand);
      });
   else if constexpr (InteriorFacetIntegrand<Term>)
      InteriorFaceLoop(mesh, element.element_index, [&](const auto& face) {
         LocalInteriorFacetIntegrandOperator(kernel, context, neighbor_context,
            operator_context, element, face, integrand);
      });
   else
      GenericCellIntegrandOperator(kernel, context, operator_context, element, integrand);
}

} // namespace local_facet_operator_detail

// A canonical batch owns only its selected original integrands at capture.
// Preparation carries the full form's already-resolved signature as a type.
template<class Policy, SemanticEntity EntityKind = SemanticEntity::Unspecified,
         class Form, class Selected, class Batch, class Rule, class Execution>
void GenericCellBatchOperator(const Form& form, const Selected& selected,
   const Batch& batch, const Rule& rule, const Execution& execution_context)
{
   constexpr auto Name = Batch::domain_name;
   using Signature = form_signature_t<Form>;
   using Signatures = typename weak_form_traits::domain_signatures<typename Signature::integrands, Name, EntityKind>::type;
   using Requirements = weak_form_execution_detail::signature_requirements<Signatures>;
   auto terms = weak_form_execution_detail::SelectIntegrands<Name, EntityKind>(form);
   auto operator_context = MakeOperatorContext(form, selected, rule);
   const auto mesh = batch.GetCellMesh();
   using Entity = ElementContext<std::remove_cvref_t<decltype(mesh.GetCell(GlobalIndex{}))>>;
   constexpr auto slots = local_facet_operator_detail::BatchSharedSlots<Requirements, Policy,
      decltype(terms), Execution, decltype(operator_context), Entity>();
   using Kernel = KernelContext<Policy, slots>;
   mesh::CellIterator<Policy>(mesh,
      [terms = std::move(terms), operator_context = std::move(operator_context),
       execution_context = execution_context, mesh]
      GENDIL_HOST_DEVICE (GlobalIndex index) mutable {
         GENDIL_SHARED Real scratch[Kernel::shared_memory_block_size];
         Kernel kernel(scratch);
         ElementContext element{index, mesh.GetCell(index)};
         if constexpr (Requirements::can_reuse_current_element_dofs)
         {
            constexpr auto Input = local_facet_operator_detail::OnlyFieldName(typename Requirements::declared_inputs{});
            constexpr auto Output = local_facet_operator_detail::OnlyFieldName(typename Requirements::declared_outputs{});
            auto input = ReadFieldDofs(kernel, execution_context.template GetReadField<Input>(), element);
            auto& sink = execution_context.template GetWriteField<Output>();
            using Row = decltype(ReadDofs(kernel, sink.space, index, sink.dofs));
            Row output{};
            auto local = BindElementDofs(terms, execution_context, operator_context,
               ReadLocal<Input>(input), AddLocal<Output>(output));
            std::apply([&](const auto&... term) {
               (local_facet_operator_detail::ApplyTerm(kernel, local, execution_context, operator_context, element, mesh, term), ...);
            }, terms);
            WriteAddDofs(kernel, sink.space, index, output, sink.dofs);
         }
         else
            std::apply([&](const auto&... term) {
               (local_facet_operator_detail::ApplyTerm(kernel, execution_context, execution_context, operator_context, element, mesh, term), ...);
            }, terms);
      });
}

// Original vector access is acquired once by the public invocation and shared
// by every named batch. Binding creates shallow endpoints, never owning data.
template<class Policy, class Form, class Context, class Rule, class... Access>
void GenericLocalFacetDomainOperator(const Form& form, const Context& context,
                                    const Rule& rule, const Access&... access)
{
   ForEachGenericOperatorDomainName(generic_operator_domain_names_t<Form>{}, [&](auto name) {
      constexpr auto Name = decltype(name)::name;
      if constexpr (has_nonzero_integrands<Form, Name>)
      {
         using Signatures = typename weak_form_traits::domain_signatures<
            typename form_signature_t<Form>::integrands, Name>::type;
         using Requirements = typename weak_form_execution_detail::signature_requirements<Signatures>::type;
         ForEachCellExecutionDomain(context, Cells<Name>{}, [&](const auto& batch) {
            auto selected = MakeRestrictedWeakFormContext<Requirements>(context, batch);
            auto execution_context = execution_context_detail::BindExecutionContext<Requirements, Policy>(selected, access...);
            GenericCellBatchOperator<Policy>(form, selected, batch, rule, execution_context);
         });
      }
   });
}

} // namespace gendil

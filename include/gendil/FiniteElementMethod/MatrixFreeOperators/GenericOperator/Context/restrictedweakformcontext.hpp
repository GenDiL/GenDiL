// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/fieldselection.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/fieldrequirements.hpp"
#include "gendil/Algebra/Vector/access.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil
{

namespace restricted_context_detail
{

// Select each space once by name, even when requirements mention both facet
// sides. Storage roles come from requirements, not registration factory names.
template<class List> struct SelectionNames;
template<class... Fields>
struct SelectionNames<type_list<Fields...>>
{
private:
   static auto OrderedFields(type_list<>) { return type_list<>{}; }

   template<class First, class... Rest>
   static auto OrderedFields(type_list<First, Rest...>)
   {
      using Remaining = concat_many_t<std::conditional_t<
         Rest::name == First::name, type_list<>, type_list<Rest>>...>;
      return concat_t<type_list<First>, decltype(OrderedFields(Remaining{}))>{};
   }

   template<class Field, class Context, class Domain>
   static auto MakeField(const Context& context, const Domain& domain)
   {
      constexpr auto Name = Field::name;
      const auto& registered = context.template GetFiniteElementField<Name>();
      auto space = RestrictFiniteElementFieldSpace<FiniteElementFieldKey<Name>,
         restricted_execution_domain_kind_v<Domain>,
         restricted_execution_domain_index_v<Domain>>(registered.space, domain);
      constexpr bool NeedsBound = [] {
         if constexpr (requires { Field::storage_role; })
         { return Field::storage_role == FieldStorageRole::Bound; }
         else { return false; }
      }();
      if constexpr (NeedsBound)
      {
         using Storage = std::remove_cvref_t<decltype(registered.dofs)>;
         if constexpr ((HostReadableVector<Storage> || DeviceReadableVector<Storage>) &&
                       !is_view_v<Storage>)
         {
            return MakeFiniteElementField<Name>(space,
               MakeVectorView(std::as_const(registered.dofs), 0, GetVectorSize(registered.dofs)));
         }
         else { return MakeFiniteElementField<Name>(space, registered.dofs); }
      }
      else { return MakeTrialField<Name>(space); }
   }

public:
   template<class Context, class Domain>
   static auto Make(const Context& context, const Domain& domain)
   {
      // Check every requirement before merging repeated side/name entries.
      ([&] {
         static_assert(Context::template HasFiniteElementField<Fields::name>(),
            "Execution context is missing a required named finite-element space.");
         if constexpr (requires { Fields::storage_role; })
            if constexpr (Fields::storage_role == FieldStorageRole::Bound)
               static_assert(Context::template HasBoundField<Fields::name>(),
                  "Execution context is missing persistent storage for a required bound field.");
      }(), ...);
      return [&]<class... Selected>(type_list<Selected...>) {
         // Braced construction selects each unique field in first-use order,
         // without copying growing map prefixes between reads and writes.
         return StaticMap<decltype(MakeField<Selected>(context, domain))...>{
            std::tuple{MakeField<Selected>(context, domain)...}};
      }(decltype(OrderedFields(type_list<Fields...>{})){});
   }
};

template<class Domain>
auto DomainEntries(const Domain& selected)
{
   constexpr auto Name = Domain::domain_name;
   if constexpr (is_selected_cell_execution_domain_v<Domain>)
   {
      using Cell = CellIntegrationDomain<Domain>;
      return std::tuple{Entry<DomainKey<Name>, Cell>{Cell{selected}}};
   }
   else if constexpr (is_selected_boundary_face_execution_domain_v<Domain>)
   {
      using CellSelection = SelectedCellExecutionDomain<Name,
         Domain::cell_part_index, typename Domain::cell_mesh_type, true>;
      using Cell = CellIntegrationDomain<CellSelection>;
      using Face = BoundaryFaceIntegrationDomain<Domain>;
      return std::tuple{
         Entry<DomainKey<Name>, Cell>{Cell{CellSelection{selected.GetCellMesh()}}},
         Entry<BoundaryFaceDomainKey<Name>, Face>{Face{selected}}};
   }
   else
   {
      using CellSelection = SelectedCellExecutionDomain<Name,
         Domain::minus_cell_part_index, typename Domain::minus_cell_mesh_type, true>;
      using Cell = CellIntegrationDomain<CellSelection>;
      using Face = InteriorFaceIntegrationDomain<Domain>;
      return std::tuple{
         Entry<DomainKey<Name>, Cell>{Cell{CellSelection{selected.GetMinusCellMesh()}}},
         Entry<InteriorFaceDomainKey<Name>, Face>{Face{selected}}};
   }
}

} // namespace restricted_context_detail

/**
 * @brief Select the required spaces and persistent data for one execution batch.
 *
 * Returns a WeakFormContext with each required field restricted to
 * selected_domain, along with that batch's topology and ordinary coefficient
 * registrations. Requirements determine whether a read uses invocation input
 * or persistent data. Missing registrations or conflicting read roles are
 * compile-time errors; complete weak-form validation belongs to the caller.
 *
 * Owning persistent vectors are represented by views borrowing their registered
 * owners, without acquiring vector data. The source context must be an lvalue
 * and its borrowed owners must outlive selection and binding. See
 * BindExecutionContext for the lifetime of the resulting kernel views.
 */
template<class Requirements, class Context, class Domain>
auto MakeRestrictedWeakFormContext(const Context& context, const Domain& selected_domain)
{
   static_assert(field_requirements_detail::ReadRolesCompatible<typename Requirements::reads>::value,
      "A field cannot be read from both active input and bound storage.");
   auto fields = restricted_context_detail::SelectionNames<
      concat_t<typename Requirements::reads, typename Requirements::writes>>::Make(
         context, selected_domain);
   auto entries = std::tuple_cat(
      restricted_context_detail::DomainEntries(selected_domain),
      fields.entries, context.fields.entries);
   auto selected = std::apply([](const auto&... values) {
      return MakeWeakFormContext(values...);
   }, entries);
   return selected;
}

// Selection can borrow owner-backed vector storage from its source registry.
// Require every source registry to be an lvalue so that this lifetime rule
// does not vary with the registrations present in a particular context type.
template<class Requirements, class Context, class Domain>
   requires (!std::is_lvalue_reference_v<Context>)
auto MakeRestrictedWeakFormContext(Context&&, const Domain&) = delete;

} // namespace gendil

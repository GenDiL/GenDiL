// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

// Native field selection and binding must not require the weak-form AST,
// interpolation kernels, or an integration rule.
#include <gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/executioncontext.hpp>
#include <gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/localoperatorcontract.hpp>
#include <gendil/FiniteElementMethod/FiniteElements/lagrangefiniteelement.hpp>
#include <gendil/Meshes/MeshDataStructures/CartesianMesh/cartesian1dmesh.hpp>
#include <gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianintermeshfaceconnectivity.hpp>
#include <gendil/FiniteElementMethod/mixedfiniteelementspace.hpp>

#include <array>
#include <iostream>
#include <type_traits>

namespace native_bindings_test
{
using namespace gendil;
struct LocalOperator {};
struct Parameters { Real multiplier; int* preparations; };
struct OperatorContext { Real multiplier; };
struct Resources { static constexpr size_t shared_slots = 0; };

struct CountedVector
{
   using element_type = Real;
   Vector data;
   mutable int reads = 0;
   int writes = 0;
   int* copies = nullptr;
   explicit CountedVector(size_t n, int* copies_ = nullptr) : data(n), copies(copies_) {}
   CountedVector(const CountedVector& other)
      : data(other.data), reads(other.reads), writes(other.writes), copies(other.copies)
   { if (copies) { ++*copies; } }
   CountedVector(CountedVector&&) = default;
   Real& operator[](GlobalIndex i) { return data[i]; }
   const Real& operator[](GlobalIndex i) const { return data[i]; }
};
size_t GetVectorSize(const CountedVector& v) { return v.data.Size(); }
const Real* ReadHostVector(const CountedVector& v)
{ ++v.reads; return v.data.ReadHostData(); }
Real* ReadWriteHostVector(CountedVector& v)
{ ++v.writes; return v.data.ReadWriteHostData(); }
Real* WriteHostVector(CountedVector& v)
{ ++v.writes; return v.data.WriteHostData(); }
const Real* ReadDeviceVector(const CountedVector& v)
{ ++v.reads; return v.data.ReadDeviceData(); }
Real* ReadWriteDeviceVector(CountedVector& v)
{ ++v.writes; return v.data.ReadWriteDeviceData(); }
Real* WriteDeviceVector(CountedVector& v)
{ ++v.writes; return v.data.WriteDeviceData(); }

struct Entity { GlobalIndex element_index; };

template<class Kernel, class Endpoint>
GENDIL_HOST_DEVICE Real ReadFirstDof(Kernel& kernel, const Endpoint& field, GlobalIndex entity)
{
   if constexpr (Endpoint::is_local) { return field.dofs[0]; }
   else
   {
      const auto local = ReadFieldDofs(kernel, field, Entity{entity});
      return local(0);
   }
}

template<class Endpoint>
GENDIL_HOST_DEVICE void AddFirstDof(Endpoint& field, GlobalIndex entity, Real value)
{
   if constexpr (Endpoint::is_local) { field.dofs[0] += value; }
   else { field.dofs(0, entity) += value; }
}
} // namespace native_bindings_test

namespace gendil
{
template<> struct local_operator_contract<native_bindings_test::LocalOperator>
{
   using requirements = FieldAccessRequirements<type_list<
      FieldRead<"u", FieldStorageRole::Input>,
      FieldRead<"p", FieldStorageRole::Input>,
      FieldRead<"f", FieldStorageRole::Bound>>,
      type_list<FieldWrite<"u">, FieldWrite<"q">>>;

   template<class Execution, class OperatorContext, class Policy>
   using resources = native_bindings_test::Resources;

   template<class Policy, class Selected>
   static auto Prepare(const native_bindings_test::LocalOperator&, const Selected&,
                       const native_bindings_test::Parameters& params)
   {
      ++*params.preparations;
      return native_bindings_test::OperatorContext{params.multiplier};
   }

   template<class Kernel, class Execution>
   GENDIL_HOST_DEVICE static void Apply(const native_bindings_test::LocalOperator&,
      Kernel& kernel, const native_bindings_test::OperatorContext& operator_context,
      const GlobalIndex& entity, Execution& execution_context)
   {
      using namespace native_bindings_test;
      const auto value = operator_context.multiplier * ReadFirstDof(kernel, execution_context.template GetReadField<"u">(), entity)
         + ReadFirstDof(kernel, execution_context.template GetReadField<"p">(), entity)
         + ReadFirstDof(kernel, execution_context.template GetReadField<"f">(), entity);
      AddFirstDof(execution_context.template GetWriteField<"u">(), entity, value);
      AddFirstDof(execution_context.template GetWriteField<"q">(), entity, -value);
   }
};
} // namespace gendil

namespace
{
using namespace gendil;
using namespace native_bindings_test;

template<class Field> concept HasDofs = requires(Field f) { f.dofs; };
template<class T> concept CanReadTemporary = requires { ReadLocal<"u">(T{}); };
template<class T> concept CanAddTemporary = requires { AddLocal<"u">(T{}); };
static_assert(!CanReadTemporary<std::array<Real, 1>>);
static_assert(!CanAddTemporary<std::array<Real, 1>>);

template<class Requirements, class Parent>
concept CanBindTemporaryParent = requires {
   BindElementDofs<Requirements>(std::declval<Parent>());
};
template<class Requirements, class Context, class Domain>
concept CanSelectTemporaryRegistry = requires(const Domain& domain) {
   MakeRestrictedWeakFormContext<Requirements>(std::declval<Context>(), domain);
};

template<class Endpoint> concept HasMutableRead = requires(Endpoint f) { f.dofs(0, 0) = Real(1); };
template<class Storage> concept HasMutableEntry = requires(Storage value) { value(0, 0) = Real(1); };
template<class Endpoint> concept HasMutableTupleRead = requires(Endpoint f) {
   std::get<0>(f.dofs)[0] = Real(1);
};

struct CopyProbe
{
   int* copies;
   explicit CopyProbe(int& count) : copies(&count) {}
   CopyProbe(const CopyProbe& other) : copies(other.copies) { ++*copies; }
};
struct CountedContainer
{
   using element_type = Real;
   Real* data;
   CopyProbe observer;
   Real& operator[](GlobalIndex i) { return data[i]; }
   const Real& operator[](GlobalIndex i) const { return data[i]; }
};

bool CheckBorrowedLocalOverlay()
{
   Cartesian1DMesh mesh(Real(1), 1);
   const auto fe = MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   const auto space = MakeFiniteElementSpace(mesh, fe);
   using Requirements = FieldAccessRequirements<type_list<
      FieldRead<"u", FieldStorageRole::Input>, FieldRead<"f", FieldStorageRole::Bound>>,
      type_list<FieldWrite<"v">>>;
   int registry_copies = 0, source_copies = 0;
   Real bound_value = 7;
   const auto source = MakeElementTensorView(space,
      CountedContainer{&bound_value, CopyProbe{source_copies}});
   const auto context = MakeWeakFormContext(MakeTrialField<"u">(space),
      MakeTestField<"v">(space), MakeFiniteElementField<"f">(space, source),
      MakeField<"copy_probe">(CopyProbe{registry_copies}), MakeIntegrationDomain<"mesh">(mesh));
   const auto selected = MakeRestrictedWeakFormContext<Requirements>(context,
      SelectedCellExecutionDomain<"mesh", 0, Cartesian1DMesh>{mesh});
   registry_copies = 0;
   auto parent = BindExecutionContext<bound_only_requirements_t<Requirements>,
      HostKernelConfiguration>(selected);
   static_assert(!std::is_reference_v<decltype(parent.selected)>);
   static_assert(!decltype(parent.selected)::template HasSpace<"f">());
   static_assert(decltype(parent.selected)::template HasSpace<"u">());
   static_assert(decltype(parent.selected)::template HasSpace<"v">());
   static_assert(decltype(parent)::template HasSpace<"f">());
   using BoundStorage = std::remove_cvref_t<decltype(parent.template GetReadField<"f">().dofs)>;
   static_assert(!HasMutableEntry<BoundStorage>);
   static_assert(!CanBindTemporaryParent<bound_only_requirements_t<Requirements>, decltype(parent)>);
   static_assert(!CanSelectTemporaryRegistry<Requirements, decltype(context),
      SelectedCellExecutionDomain<"mesh", 0, Cartesian1DMesh>>);
   // Vector element rows are tuples and intentionally have no scalar indexing
   // interface. The local field must keep a reference to this exact tuple.
   auto input = std::tuple{std::array<Real, 1>{2}, std::array<Real, 2>{3, 5}};
   auto output = std::tuple{std::array<Real, 1>{0}, std::array<Real, 2>{0, 0}};
   const auto read_handle = ReadLocal<"u">(input);
   const auto add_handle = AddLocal<"v">(output);
   static_assert(std::is_same_v<decltype(read_handle.local), const decltype(input)&>);
   static_assert(std::is_same_v<decltype(add_handle.local), decltype(output)&>);
   registry_copies = source_copies = 0;
   auto local = BindElementDofs<Requirements>(parent, read_handle, add_handle);
   auto again = BindElementDofs<Requirements>(local,
      ReadLocal<"u">(input), AddLocal<"v">(output));
   auto copied = again;
   static_assert(std::is_same_v<decltype(local.selected), const decltype(parent)&>);
   static_assert(std::is_same_v<decltype(copied.selected), const decltype(local)&>);
   static_assert(decltype(again)::template HasSpace<"u">());
   static_assert(decltype(again)::template HasSpace<"f">());
   using LocalRead = std::remove_cvref_t<decltype(local.template GetReadField<"u">())>;
   using LocalWrite = std::remove_cvref_t<decltype(local.template GetWriteField<"v">())>;
   static_assert(std::is_same_v<typename LocalRead::storage_type, const decltype(input)&>);
   static_assert(std::is_same_v<typename LocalWrite::storage_type, decltype(output)&>);
   static_assert(!HasMutableTupleRead<LocalRead>);
   static_assert(std::tuple_size_v<decltype(local.reads.entries)> == 1);
   HostKernelConfiguration kernel;
   const auto& read = ReadFieldDofs(kernel, copied.template GetReadField<"u">(), GlobalIndex(0));
   std::get<0>(copied.template GetWriteField<"v">().dofs)[0] += std::get<0>(read)[0];
   std::get<1>(local.template GetWriteField<"v">().dofs)[1] += std::get<1>(read)[1];
   return registry_copies == 0 && source_copies == 0 &&
      &local.selected == &parent && &copied.selected == &local &&
      &copied.GetRegistry() == &parent.GetRegistry() &&
      &copied.template GetReadField<"f">() == &parent.template GetReadField<"f">() &&
      &again.template GetSpace<"u">() == &parent.template GetSpace<"u">() &&
      &again.template GetSpace<"f">() == &parent.template GetReadField<"f">().space &&
      &local.template GetReadField<"f">() == &parent.template GetReadField<"f">() &&
      &local.template GetReadField<"u">().dofs == &input &&
      &local.template GetWriteField<"v">().dofs == &output &&
      std::get<0>(output)[0] == 2 && std::get<1>(output)[1] == 5 &&
      parent.template GetReadField<"f">().dofs(0, 0) == 7;
}

bool CheckReadOnlyPreboundViews()
{
   Cartesian1DMesh mesh(Real(1), 1);
   const auto space = MakeFiniteElementSpace(mesh,
      MakeLegendreFiniteElement(FiniteElementOrders<1>{}));
   Real values[2]{3, 5};
   auto pointer = MakeElementTensorView(space, PointerContainer<Real>{values});
   auto weighted = MakeWeightedRestrictionView(GetRestriction(space), PointerContainer<Real>{values});
   HostDevicePointer<Real> host_device_pointer;
   host_device_pointer.host_pointer = values;
   const auto host_device = MakeElementTensorView(space, host_device_pointer);
   static_assert(HasMutableEntry<decltype(host_device)>);
   auto host_device_read = MakeReadOnlyElementTensorView(host_device);
   auto host_device_copy = host_device_read;
   static_assert(!HasMutableEntry<decltype(host_device_copy)>);
   static_assert(HasMutableEntry<decltype(pointer)>);
   static_assert(HasMutableEntry<decltype(weighted)>);
   auto reads = MakeReadOnlyElementTensorView(std::tuple{pointer, weighted});
   // Copying either view must retain deep constness, including the weighted
   // proxy path. Const qualification of the endpoint itself is insufficient.
   auto pointer_copy = std::get<0>(reads);
   auto weighted_copy = std::get<1>(reads);
   static_assert(!HasMutableEntry<decltype(pointer_copy)>);
   static_assert(!HasMutableEntry<decltype(weighted_copy)>);
   return host_device_copy(0, 0) == Real(3) && host_device_copy(1, 0) == Real(5) &&
      pointer_copy(0, 0) == Real(3) && pointer_copy(1, 0) == Real(5) &&
      Real(weighted_copy(0, 0)) == Real(3) && Real(weighted_copy(1, 0)) == Real(5);
}

template<class Policy>
bool CheckReadOnlyComponentViews()
{
   Cartesian1DMesh mesh(Real(1), 1);
   const auto space = MakeFiniteElementSpace(mesh,
      MakeLegendreFiniteElement(FiniteElementOrders<1>{}));
   Vector values(2), result(2);
   values.WriteHostData()[0] = Real(3);
   values.ReadWriteHostData()[1] = Real(5);
   const auto data = ReadWriteKernelVector<Policy::is_device_configuration>(values);
   auto output = WriteKernelVector<Policy::is_device_configuration>(result);
   const auto pointer = MakeElementTensorView(space, data);
   const auto weighted = MakeWeightedRestrictionView(GetRestriction(space), data);
   // Convert component views inside the kernel, where HIP cannot use the
   // standard library's host-only tuple deduction guides.
   Policy::BlockLoop(1, [=] GENDIL_HOST_DEVICE (GlobalIndex) mutable {
      const auto components = std::make_tuple(pointer, weighted);
      auto reads = MakeReadOnlyElementTensorView(components);
      static_assert(!HasMutableEntry<std::tuple_element_t<0, decltype(reads)>>);
      static_assert(!HasMutableEntry<std::tuple_element_t<1, decltype(reads)>>);
      const auto empty = MakeReadOnlyElementTensorView(std::tuple<>{});
      static_assert(std::tuple_size_v<decltype(empty)> == 0);
      output[0] = std::get<0>(reads)(0, 0);
      output[1] = Real(std::get<1>(reads)(1, 0));
   });
   if constexpr (Policy::is_device_configuration) { GENDIL_DEVICE_SYNC; }
   return result.ReadHostData()[0] == Real(3) && result.ReadHostData()[1] == Real(5);
}

template<class Policy>
bool CheckOwnerSelection()
{
   Cartesian1DMesh mesh(Real(1), 1);
   auto space = MakeFiniteElementSpace(mesh, MakeLegendreFiniteElement(FiniteElementOrders<0>{}));
   int copies = 0;
   CountedVector storage(1, &copies);
   storage.data = Real(6);
   const auto context = MakeWeakFormContext(MakeFiniteElementField<"f">(space, storage),
      MakeIntegrationDomain<"mesh">(mesh));
   const auto& owner = context.template GetBoundField<"f">();
   using Requirements = FieldAccessRequirements<type_list<FieldRead<"f", FieldStorageRole::Bound>>, type_list<>>;
   copies = 0;
   const auto selected = MakeRestrictedWeakFormContext<Requirements>(context,
      SelectedCellExecutionDomain<"mesh", 0, Cartesian1DMesh>{mesh});
   bool ok = copies == 0 && owner.reads == 0;
   const auto execution_context = BindExecutionContext<Requirements, Policy>(selected);
   static_assert(!decltype(execution_context.selected)::template HasBoundField<"f">());
   return ok && copies == 0 && owner.reads == 1;
}

// Side requirements remain numerical metadata: one named binding serves both
// sides, even when their basis orders and restriction shapes differ.
template<class Policy>
bool CheckFacetBindings()
{
   Cartesian1DMesh minus_mesh(Real(1), 1), plus_mesh(Real(1), 1);
   CartesianIntermeshFaceConnectivity<1, 1> faces({1}, {1});
   auto partition = MakePartition(MakeCellPart(minus_mesh), MakeCellPart(plus_mesh),
      MakeInteriorFacePart<0, 1>(faces));
   auto mixed = MakeMixedFiniteElementSpace(partition,
      std::tuple{MakeLegendreFiniteElement(FiniteElementOrders<0>{}),
                 MakeLobattoFiniteElement(FiniteElementOrders<1>{})},
      std::tuple{ContiguousL2RestrictionSpecification{0, 3},
                 ContiguousL2RestrictionSpecification{1, 3}});
   int owner_copies = 0;
   CountedVector storage(3, &owner_copies);
   storage.data = Real(7);
   auto context = MakeWeakFormContext(MakeTrialField<"u">(mixed), MakeTestField<"u">(mixed),
      MakeFiniteElementField<"f">(mixed, storage), MakeIntegrationDomain<"mesh">(partition));
   const auto& owner = context.template GetBoundField<"f">();
   using Requirements = FieldAccessRequirements<type_list<
      FieldRead<"u", FieldStorageRole::Input, FieldSide::Minus>,
      FieldRead<"f", FieldStorageRole::Bound, FieldSide::Minus>,
      FieldRead<"u", FieldStorageRole::Input, FieldSide::Plus>,
      FieldRead<"f", FieldStorageRole::Bound, FieldSide::Plus>>,
      type_list<FieldWrite<"u", FieldSide::Minus>, FieldWrite<"u", FieldSide::Plus>>>;
   const auto part = std::get<0>(partition.interior_face_parts);
   using Batch = SelectedInteriorFaceExecutionDomain<"mesh", 0, decltype(part),
      Cartesian1DMesh, Cartesian1DMesh>;
   owner_copies = 0;
   auto selected = MakeRestrictedWeakFormContext<Requirements>(context,
      Batch{part, minus_mesh, plus_mesh});
   using SelectedEntries = decltype(selected.fe_fields.entries);
   static_assert(std::tuple_size_v<SelectedEntries> == 2);
   static_assert(std::is_same_v<typename std::tuple_element_t<0, SelectedEntries>::key_type,
      FiniteElementFieldKey<"u">>);
   static_assert(std::is_same_v<typename std::tuple_element_t<1, SelectedEntries>::key_type,
      FiniteElementFieldKey<"f">>);
   CountedVector x(3), y(3);
   x.data = Real(2); y.data = Real(0);
   auto execution_context = BindExecutionContext<Requirements, Policy>(selected, x, y);
   static_assert(std::tuple_size_v<decltype(execution_context.reads.entries)> == 2);
   static_assert(std::tuple_size_v<decltype(execution_context.writes.entries)> == 1);
   static_assert(std::tuple_size_v<decltype(execution_context.selected.fe_fields.entries)> == 0);
   static_assert(decltype(execution_context)::template HasSpace<"u">());
   static_assert(decltype(execution_context)::template HasSpace<"f">());
   static_assert(!decltype(execution_context)::template HasSpace<"absent">());
   using Read = std::remove_cvref_t<decltype(execution_context.template GetReadField<"u">())>;
   static_assert(is_two_space_interior_face_field_binding_v<typename Read::space_type>);
   static_assert(std::remove_cvref_t<typename Read::storage_type>::rank == 1);
   using ReadStorage = std::remove_cvref_t<typename Read::storage_type>;
   static_assert(!requires(ReadStorage value) { value[0] = Real(1); });
   bool ok = owner_copies == 0 && owner.reads == 1 && x.reads == 1 && y.writes == 1 &&
      &execution_context.template GetSpace<"u">() == &execution_context.template GetReadField<"u">().space &&
      &execution_context.template GetSpace<"f">() == &execution_context.template GetReadField<"f">().space;
   using LocalRequirements = FieldAccessRequirements<
      type_list<FieldRead<"u", FieldStorageRole::Input>, FieldRead<"f", FieldStorageRole::Bound>>,
      type_list<FieldWrite<"u">>>;
   Policy::BlockLoop(1, [=] GENDIL_HOST_DEVICE (GlobalIndex) mutable {
      std::array<Real, 1> minus_input{3}, minus_output{0};
      std::array<Real, 2> plus_input{5, 11}, plus_output{0, 0};
      auto minus = BindElementDofs<LocalRequirements>(execution_context,
         ReadLocal<"u">(minus_input), AddLocal<"u">(minus_output));
      auto plus = BindElementDofs<LocalRequirements>(execution_context,
         ReadLocal<"u">(plus_input), AddLocal<"u">(plus_output));
      minus.template GetWriteField<"u">().dofs[0] += minus.template GetReadField<"u">().dofs[0];
      plus.template GetWriteField<"u">().dofs[0] += plus.template GetReadField<"u">().dofs[0];
      plus.template GetWriteField<"u">().dofs[1] += plus.template GetReadField<"u">().dofs[1];
      auto& output = execution_context.template GetWriteField<"u">().dofs;
      output[0] = minus_output[0];
      output[1] = plus_output[0];
      output[2] = plus_output[1] + plus.template GetReadField<"f">().dofs[2];
   });
   if constexpr (Policy::is_device_configuration) { GENDIL_DEVICE_SYNC; }
   return ok && y.data.ReadHostData()[0] == 3 && y.data.ReadHostData()[1] == 5 &&
      y.data.ReadHostData()[2] == 18;
}

template<class Policy>
bool CheckBindings()
{
   Cartesian1DMesh mesh(Real(0.5), 2);
   auto p0 = MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   auto p1 = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   auto u = MakeFiniteElementSpace(mesh, p0, ContiguousL2RestrictionSpecification{3, 9});
   auto p = MakeFiniteElementSpace(mesh, p1, ContiguousL2RestrictionSpecification{8, 15});
   auto q = MakeFiniteElementSpace(mesh, p0, ContiguousL2RestrictionSpecification{12, 17});
   auto f = MakeFiniteElementSpace(mesh, p0, ContiguousL2RestrictionSpecification{1, 4});
   Vector bound(4); bound = Real(2);
   auto context = MakeWeakFormContext(
      MakeTrialField<"u">(u), MakeTestField<"u">(u),
      MakeTrialField<"p">(p), MakeTestField<"q">(q),
      MakeFiniteElementField<"f">(f, bound),
      MakeTrialField<"unused">(p), MakeIntegrationDomain<"mesh">(mesh));
   static_assert(!HasDofs<std::remove_cvref_t<decltype(context.template GetFiniteElementField<"u">())>>);
   static_assert(decltype(context)::template HasBoundField<"f">());
   static_assert(!decltype(context)::template HasBoundField<"u">());
   using Requirements = local_operator_requirements_t<LocalOperator>;
   auto selected = MakeRestrictedWeakFormContext<Requirements>(context,
      SelectedCellExecutionDomain<"mesh", 0, Cartesian1DMesh>{mesh});
   static_assert(!decltype(selected)::template HasFiniteElementField<"unused">());
   static_assert(std::tuple_size_v<decltype(selected.fe_fields.entries)> == 4);
   using SelectedEntries = decltype(selected.fe_fields.entries);
   static_assert(std::is_same_v<typename std::tuple_element_t<0, SelectedEntries>::key_type,
      FiniteElementFieldKey<"u">>);
   static_assert(std::is_same_v<typename std::tuple_element_t<1, SelectedEntries>::key_type,
      FiniteElementFieldKey<"p">>);
   static_assert(std::is_same_v<typename std::tuple_element_t<2, SelectedEntries>::key_type,
      FiniteElementFieldKey<"f">>);
   static_assert(std::is_same_v<typename std::tuple_element_t<3, SelectedEntries>::key_type,
      FiniteElementFieldKey<"q">>);
   using SelectedStorage = std::remove_cvref_t<decltype(selected.template GetBoundField<"f">())>;
   static_assert(is_view_v<SelectedStorage>);
   static_assert(SelectedStorage::rank == 1);
   // Factory ownership remains unchanged; selection must borrow that stable
   // registration rather than create another owning vector in a kernel handle.
   auto input_registration = MakeWeakFormContext(MakeFiniteElementField<"u">(u, bound),
      MakeIntegrationDomain<"mesh">(mesh));
   using InputOnly = FieldAccessRequirements<type_list<FieldRead<"u", FieldStorageRole::Input>>,
      type_list<>>;
   auto active_selected = MakeRestrictedWeakFormContext<InputOnly>(input_registration,
      SelectedCellExecutionDomain<"mesh", 0, Cartesian1DMesh>{mesh});
   static_assert(!decltype(active_selected)::template HasBoundField<"u">());
   using DuplicateReads = FieldAccessRequirements<type_list<
      FieldRead<"u", FieldStorageRole::Input>, FieldRead<"u", FieldStorageRole::Input>>,
      type_list<FieldWrite<"u">>>;
   auto duplicate_selected = MakeRestrictedWeakFormContext<DuplicateReads>(context,
      SelectedCellExecutionDomain<"mesh", 0, Cartesian1DMesh>{mesh});
   static_assert(std::tuple_size_v<decltype(duplicate_selected.fe_fields.entries)> == 1);
   int preparations = 0;
   const LocalOperator local_operator;
   const auto operator_context = PrepareLocalOperator<Policy>(local_operator, selected, Parameters{Real(3), &preparations});
   CountedVector x(15), y(19), x2(15), y2(19);
   x.data = Real(4); x2.data = Real(7); y.data = Real(1); y2.data = Real(-1);
   auto duplicate_binding = BindExecutionContext<DuplicateReads, Policy>(duplicate_selected, x, y);
   static_assert(std::tuple_size_v<decltype(duplicate_binding.reads.entries)> == 1);
   static_assert(std::tuple_size_v<decltype(duplicate_binding.writes.entries)> == 1);
   x.reads = 0; y.writes = 0;
   auto first = BindExecutionContext<Requirements, Policy>(selected, x, y);
   auto second = [&] {
      auto temporary_selected = MakeRestrictedWeakFormContext<Requirements>(context,
         SelectedCellExecutionDomain<"mesh", 0, Cartesian1DMesh>{mesh});
      return BindExecutionContext<Requirements, Policy>(temporary_selected, x2, y2);
   }();
   using KernelStorage = std::remove_cvref_t<decltype(first.template GetReadField<"f">().dofs)>;
   static_assert(is_view_v<KernelStorage>);
   static_assert(!decltype(first.selected)::template HasBoundField<"f">());
   static_assert(std::tuple_size_v<decltype(first.selected.fe_fields.entries)> == 0);
   static_assert(decltype(first)::template HasSpace<"q">());
   static_assert(!decltype(first)::template HasSpace<"unused">());
   auto persistent = BindExecutionContext<bound_only_requirements_t<Requirements>, Policy>(selected);
   static_assert(decltype(persistent)::template HasSpace<"u">());
   static_assert(std::tuple_size_v<decltype(persistent.reads.entries)> == 1);
   static_assert(std::tuple_size_v<decltype(persistent.writes.entries)> == 0);
   using Read = std::remove_cvref_t<decltype(first.template GetReadField<"u">())>;
   static_assert(!HasMutableRead<Read>);
   using Resource = local_operator_resources_t<LocalOperator, decltype(first), decltype(operator_context), Policy>;
   static_assert(Resource::shared_slots == 0);
   Policy::BlockLoop(2, [=] GENDIL_HOST_DEVICE (GlobalIndex entity) mutable {
      Policy kernel;
      ApplyLocalOperator(local_operator, kernel, operator_context, entity, first);
      ApplyLocalOperator(local_operator, kernel, operator_context, entity, second);
   });
   if constexpr (Policy::is_device_configuration) { GENDIL_DEVICE_SYNC; }
   bool ok = preparations == 1 && x.reads == 1 && y.writes == 1 && x2.reads == 1 && y2.writes == 1 &&
      &first.template GetSpace<"q">() == &first.template GetWriteField<"q">().space;
   // These overlays are constructed inside each kernel. The tensors remain
   // local references, and bound field reads keep their selected restrictions.
   Policy::BlockLoop(2, [=] GENDIL_HOST_DEVICE (GlobalIndex entity) mutable {
      Policy kernel;
      std::array<Real, 1> local_u{Real(2)}, local_out{Real(5)}, local_q{Real(-1)};
      std::array<Real, 2> local_p{Real(3), Real(8)};
      auto bound = BindElementDofs<bound_only_requirements_t<Requirements>>(persistent);
      auto probe = BindElementDofs<Requirements>(bound,
         ReadLocal<"u">(local_u), ReadLocal<"p">(local_p),
         AddLocal<"u">(local_out), AddLocal<"q">(local_q));
      ApplyLocalOperator(local_operator, kernel, operator_context, entity, probe);
      auto& output = first.template GetWriteField<"u">();
      output.dofs(0, entity) = local_out[0];
      first.template GetWriteField<"q">().dofs(0, entity) = local_q[0];
   });
   if constexpr (Policy::is_device_configuration) { GENDIL_DEVICE_SYNC; }
   for (size_t i = 0; i < 19; ++i)
   {
      const Real contribution2 = i == 3 || i == 4 ? Real(30) : i == 12 || i == 13 ? Real(-30) : Real(0);
      const Real expected = i == 3 || i == 4 ? Real(16) :
         i == 12 || i == 13 ? Real(-12) : Real(1);
      ok = y.data.ReadHostData()[i] == expected && ok;
      ok = y2.data.ReadHostData()[i] == Real(-1) + contribution2 && ok;
   }

   using Load = FieldAccessRequirements<type_list<FieldRead<"f", FieldStorageRole::Bound>>,
      type_list<FieldWrite<"q">>>;
   auto load_selected = MakeRestrictedWeakFormContext<Load>(context,
      SelectedCellExecutionDomain<"mesh", 0, Cartesian1DMesh>{mesh});
   auto load = BindExecutionContext<Load, Policy>(load_selected, y);
   static_assert(std::tuple_size_v<decltype(load.reads.entries)> == 1);
   static_assert(!decltype(load.reads)::template Contains<NameTag<"u">>());
   return ok;
}
} // namespace

int main()
{
   bool ok = CheckBorrowedLocalOverlay() && CheckReadOnlyPreboundViews() &&
      CheckReadOnlyComponentViews<gendil::HostKernelConfiguration>() &&
      CheckOwnerSelection<gendil::HostKernelConfiguration>() &&
      CheckFacetBindings<gendil::HostKernelConfiguration>() &&
      CheckBindings<gendil::HostKernelConfiguration>();
#ifdef GENDIL_USE_DEVICE
   // This native local operator processes a complete local row in one device thread.
   using Device = gendil::DeviceKernelConfiguration<gendil::ThreadBlockLayout<>, 1, 4>;
   ok = CheckBindings<Device>() && ok;
   ok = CheckReadOnlyComponentViews<Device>() && ok;
   ok = CheckOwnerSelection<Device>() && ok;
   ok = CheckFacetBindings<Device>() && ok;
#endif
   if (!ok) { std::cerr << "Execution context binding/native local operator contract failed.\n"; }
   return ok ? 0 : 1;
}

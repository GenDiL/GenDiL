// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Connect named finite-element fields to the vectors used by one application.
 *
 * Numerical routines read and write fields by name. Each field's restriction
 * determines which entries of the common input or output vector it uses; field
 * order does not determine storage offsets. Residual operators arrange this
 * binding internally. Use this header when implementing numerical routines,
 * native local operators, or assembly probes.
 *
 * Given selected spaces and requirements that read "u" and write "v":
 * @code
 * auto execution_context = BindExecutionContext<Requirements, HostKernelConfiguration>(
 *    selected, x, y);
 * const auto& input_field = execution_context.GetReadField<"u">();  // Reads from x.
 * auto& output_field = execution_context.GetWriteField<"v">();     // Writes into y.
 * @endcode
 * These field views carry a space and access to its storage. Binding alone
 * neither evaluates the operator nor clears y.
 *
 * The registered WeakFormContext supplies spaces and persistent field data.
 * MakeRestrictedWeakFormContext selects them for one mesh or partition batch.
 * Requirements describe named reads and writes: active reads use x, bound
 * reads use registered data, and writes use y. Read and write bindings are
 * separate, so one name may refer to different input and output buffers.
 * Facet sides select restrictions later; field access remains name-only.
 *
 * An operator_context holds reusable basis and geometry preparation separately;
 * it is not part of the execution_context. Rebinding vectors therefore does not
 * require rebuilding that preparation. BindElementDofs provides the same named
 * access to borrowed element rows for cached execution and assembly probes.
 */

#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/fieldrequirements.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/restrictedweakformcontext.hpp"
#include "gendil/Algebra/Vector/access.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/readdofs.hpp"
#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil
{

namespace execution_context_detail
{
template<class Base> struct RegistryType { using type = Base; };
template<class Parent> struct RegistryType<const Parent&>
{
   using type = typename Parent::selected_context_type;
};
}

/**
 * @brief Supply named field storage to a numerical routine for one application.
 *
 * Construct with BindExecutionContext or BindElementDofs. A field returned by
 * GetReadField() or GetWriteField() combines its selected space with its storage.
 * Read bindings expose const access; write bindings identify the destination.
 *
 * An invocation context owns its shallow space/domain handles and field views,
 * so the temporary selected context used to build it may go out of scope.
 * It does not own the underlying vector or mesh allocations: those must remain
 * valid, at stable addresses, until all kernels using the bindings complete.
 *
 * A local overlay borrows a parent ExecutionContext and element rows by
 * reference. It stores only local overrides and inherits bound reads from its
 * parent. Copy construction preserves these borrows without copying the parent
 * or rows. Construct overlays inside the executing scope, as described by
 * BindElementDofs; capturing a host-local overlay into a device kernel is invalid.
 */
template<class Base, class Reads, class Writes>
struct ExecutionContext
{
   using execution_context_tag = void;
   using selected_context_type = typename execution_context_detail::RegistryType<Base>::type;
   using reads_type = Reads;
   using writes_type = Writes;
   static constexpr bool is_borrowed = std::is_lvalue_reference_v<Base>;
   Base selected;
   Reads reads;
   Writes writes;

   /// Remaining registrations; use GetSpace() to also find spaces stored in field views.
   GENDIL_HOST_DEVICE constexpr const auto& GetRegistry() const
   {
      if constexpr (is_borrowed)
         return selected.GetRegistry();
      else return selected;
   }

   /// Obtain the named cell-domain entry of the selected batch.
   template<StaticString Name>
   GENDIL_HOST_DEVICE constexpr decltype(auto) GetDomain() const
   { return GetRegistry().template GetDomain<Name>(); }

   /// Obtain a selected space without requiring a read or write binding for it.
   template<StaticString Name>
   GENDIL_HOST_DEVICE constexpr decltype(auto) GetSpace() const
   {
      if constexpr (is_borrowed)
         return selected.template GetSpace<Name>();
      else if constexpr (Reads::template Contains<NameTag<Name>>())
         return (reads.template Get<NameTag<Name>>().space);
      else if constexpr (Writes::template Contains<NameTag<Name>>())
         return (writes.template Get<NameTag<Name>>().space);
      else return selected.template GetSpace<Name>();
   }

   template<StaticString Name>
   static consteval bool HasSpace()
   {
      if constexpr (is_borrowed)
         return std::remove_cvref_t<Base>::template HasSpace<Name>();
      else return Reads::template Contains<NameTag<Name>>() ||
                  Writes::template Contains<NameTag<Name>>() ||
                  selected_context_type::template HasSpace<Name>();
   }

   /// Return the read binding, including inherited persistent data in local overlays.
   template<StaticString Name>
   GENDIL_HOST_DEVICE constexpr const auto& GetReadField() const
   {
      if constexpr (Reads::template Contains<NameTag<Name>>())
         return reads.template Get<NameTag<Name>>();
      else
      {
         static_assert(is_borrowed && requires { selected.template GetReadField<Name>(); },
            "Execution context has no binding for the required read field.");
         return selected.template GetReadField<Name>();
      }
   }

   /// Return this context's output binding; local outputs never fall back to the parent.
   template<StaticString Name>
   GENDIL_HOST_DEVICE constexpr auto& GetWriteField()
   { return writes.template Get<NameTag<Name>>(); }

   template<StaticString Name>
   GENDIL_HOST_DEVICE constexpr const auto& GetWriteField() const
   { return writes.template Get<NameTag<Name>>(); }
};

/// Named const row reference created by ReadLocal for BindElementDofs.
template<StaticString Name, class Tensor>
struct LocalReadHandle
{
   static constexpr auto name = Name;
   static constexpr bool is_read = true;
   const Tensor& local;
};

/// Named mutable row reference created by AddLocal for BindElementDofs.
template<StaticString Name, class Tensor>
struct LocalAddHandle
{
   static constexpr auto name = Name;
   static constexpr bool is_read = false;
   Tensor& local;
};

/** Borrow an existing input row; no tensor is copied or gathered.
 * The row must outlive every local overlay using the handle. Temporaries are rejected.
 */
template<StaticString Name, class Tensor>
GENDIL_HOST_DEVICE constexpr auto ReadLocal(const Tensor& local)
{ return LocalReadHandle<Name, Tensor>{local}; }

template<StaticString Name, class Tensor>
   requires (!std::is_lvalue_reference_v<Tensor>)
auto ReadLocal(Tensor&&) = delete;

/** Borrow an existing output row for accumulation by the numerical routine.
 * The caller initializes the row and keeps it alive; this handle neither clears
 * it nor scatters it to a global vector. Const rows and temporaries are rejected.
 */
template<StaticString Name, class Tensor>
   requires (!std::is_const_v<Tensor>)
GENDIL_HOST_DEVICE constexpr auto AddLocal(Tensor& local)
{ return LocalAddHandle<Name, Tensor>{local}; }

namespace execution_context_detail
{

template<class T> struct HasActiveReads;
template<class... Fields>
struct HasActiveReads<type_list<Fields...>>
   : std::bool_constant<((Fields::storage_role == FieldStorageRole::Input) || ...)> {};

template<class Space, class Storage>
GENDIL_HOST_DEVICE auto BoundElementAccess(const Space& space, const Storage& storage)
{
   if constexpr (is_face_field_binding_v<Space>)
   {
      // Algebraic storage is restricted only after the facet selects its side.
      // Prebound element/component views retain their existing interpretation.
      if constexpr (std::is_pointer_v<Storage>)
         return static_cast<const std::remove_pointer_t<Storage>*>(storage);
      else if constexpr (requires { typename Storage::index_type; storage.data; })
         return PointerContainer<const typename Storage::element_type, typename Storage::index_type>{storage.data};
      else if constexpr (requires { storage.host_pointer; })
      {
         HostDevicePointer<const typename Storage::element_type> access;
         access.host_pointer = storage.host_pointer;
#ifdef GENDIL_USE_DEVICE
         access.device_pointer = storage.device_pointer;
#endif
         return access;
      }
      else return MakeReadOnlyElementTensorView(storage);
   }
   else if constexpr (requires { Storage::rank; })
   {
      if constexpr (Storage::rank > 1) { return MakeReadOnlyElementTensorView(storage); }
      else { return MakeReadOnlyElementTensorView(MakeElementTensorView(space, storage)); }
   }
   else if constexpr (requires { std::tuple_size<Storage>::value; })
   { return MakeReadOnlyElementTensorView(storage); }
   else { return MakeReadOnlyElementTensorView(MakeElementTensorView(space, storage)); }
}

// Resolve persistent owners to access in the policy's memory space before
// kernel capture. Existing element views retain their local indexing layout.
template<class Policy, class Space, class Storage>
auto BoundAccess(const Space& space, const Storage& storage)
{
   if constexpr (KernelReadableVector<is_device_configuration_v<Policy>, Storage>)
   {
      auto access = ReadKernelVector<is_device_configuration_v<Policy>>(storage);
      if constexpr (is_face_field_binding_v<Space>) return access;
      else return MakeElementTensorView(space, access.data);
   }
   else { return BoundElementAccess(space, storage); }
}

template<class List> struct BindReads;
template<class... Fields>
struct BindReads<type_list<Fields...>>
{
   template<class Policy, class Selected, class Input>
   static auto Make(const Selected& selected, const Input& input)
   { return Add<Policy>(MakeStaticMap(), selected, input, type_list<Fields...>{}); }

   template<class Policy, class Map, class Selected, class Input>
   static auto Add(Map map, const Selected&, const Input&, type_list<>)
   { return map; }

   template<class Policy, class Map, class Selected, class Input, class Field, class... Rest>
   static auto Add(Map map, const Selected& selected, const Input& input,
                   type_list<Field, Rest...>)
   {
      constexpr auto Name = Field::name;
      if constexpr (Map::template Contains<NameTag<Name>>())
      { return Add<Policy>(std::move(map), selected, input, type_list<Rest...>{}); }
      else
      {
         auto space = selected.template GetFiniteElementField<Name>().space;
         auto access = [&]() {
            if constexpr (Field::storage_role == FieldStorageRole::Input)
            {
               if constexpr (is_face_field_binding_v<decltype(space)>)
                  return MakeReadOnlyElementTensorView(input);
               else return MakeReadOnlyElementTensorView(MakeElementTensorView(space, input.data));
            }
            else { return BoundAccess<Policy>(space, selected.template GetBoundField<Name>()); }
         }();
         auto endpoint = FiniteElementFieldView<decltype(space), decltype(access)>{space, access};
         return Add<Policy>(MapAppend(std::move(map), MakeEntry<NameTag<Name>>(endpoint)),
            selected, input, type_list<Rest...>{});
      }
   }
};

template<class List> struct BindWrites;
template<class... Fields>
struct BindWrites<type_list<Fields...>>
{
   template<class Selected, class Output>
   static auto Make(const Selected& selected, const Output& output)
   { return Add(MakeStaticMap(), selected, output, type_list<Fields...>{}); }

   template<class Map, class Selected, class Output>
   static auto Add(Map map, const Selected&, const Output&, type_list<>)
   { return map; }

   template<class Map, class Selected, class Output, class Field, class... Rest>
   static auto Add(Map map, const Selected& selected, const Output& output,
                   type_list<Field, Rest...>)
   {
      constexpr auto Name = Field::name;
      if constexpr (Map::template Contains<NameTag<Name>>())
      { return Add(std::move(map), selected, output, type_list<Rest...>{}); }
      else
      {
         auto space = selected.template GetFiniteElementField<Name>().space;
         auto dofs = [&]() {
            if constexpr (is_face_field_binding_v<decltype(space)>) return output;
            else return MakeElementTensorView(space, output.data);
         }();
         auto endpoint = FiniteElementFieldView<decltype(space), decltype(dofs)>{space, dofs};
         return Add(MapAppend(std::move(map), MakeEntry<NameTag<Name>>(endpoint)),
            selected, output, type_list<Rest...>{});
      }
   }
};

template<FieldSide Side, class Space>
constexpr decltype(auto) RequiredVolumeSpace(const Space& space)
{
   if constexpr (Side == FieldSide::Plus) return GetPlusFacetFieldVolumeSpace(space);
   else return GetMinusFacetFieldVolumeSpace(space);
}

template<class Selected, class... Fields>
void ValidateInputExtent(const Selected& selected, size_t extent, type_list<Fields...>)
{
   ([&] {
      if constexpr (Fields::storage_role == FieldStorageRole::Input)
      {
         GENDIL_VERIFY(extent >= static_cast<size_t>(GetAlgebraicDofExtent(
            RequiredVolumeSpace<Fields::side>(selected.template GetFiniteElementField<Fields::name>().space))),
            "Execution input vector is smaller than a required field's algebraic extent.");
      }
   }(), ...);
}

template<class Selected, class... Fields>
void ValidateOutputExtent(const Selected& selected, size_t extent, type_list<Fields...>)
{
   auto validate_output = [&]<class Field>()
   {
      GENDIL_VERIFY(extent >= static_cast<size_t>(GetAlgebraicDofExtent(
         RequiredVolumeSpace<Field::side>(selected.template GetFiniteElementField<Field::name>().space))),
         "Execution output vector is smaller than a required field's algebraic extent.");
   };
   (validate_output.template operator()<Fields>(), ...);
}

// Validate available logical layout extents without assuming a layout's number
// of entries is the size of its backing allocation.
template<class Restriction, class Storage>
void ValidateElementStorageExtent(const Restriction& restriction, const Storage& storage, GlobalIndex elements)
{
   if constexpr (requires { std::tuple_size<Storage>::value; })
   {
      static_assert(VectorElementDoFRestriction<Restriction>,
         "Component storage requires a vector finite element restriction.");
      []<size_t... I>(const auto& r, const auto& values, GlobalIndex count, std::index_sequence<I...>) {
         (ValidateElementStorageExtent(GetComponentRestriction<I>(r), std::get<I>(values), count), ...);
      }(restriction, storage, elements, std::make_index_sequence<std::tuple_size_v<Storage>>{});
   }
   else if constexpr (requires { Storage::rank; })
   {
      using R = std::remove_cvref_t<Restriction>;
      static_assert(Storage::rank == static_cast<size_t>(R::tensor_dim) + 1,
         "Persistent element storage rank does not match its selected restriction.");
      []<size_t... I>(const auto& values, GlobalIndex count, std::index_sequence<I...>) {
         ([&] {
            if constexpr (requires { values.Get(AxisTag<I>{}); })
               GENDIL_VERIFY((values.Get(AxisTag<I>{}) >= seq_get_v<I, typename R::dof_shape_type>),
                  "Persistent field element storage has an insufficient local extent.");
         }(), ...);
         if constexpr (requires { values.Get(AxisTag<Storage::rank - 1>{}); })
            GENDIL_VERIFY(values.Get(AxisTag<Storage::rank - 1>{}) >= count,
               "Persistent field element storage has an insufficient element extent.");
      }(storage, elements, std::make_index_sequence<Storage::rank - 1>{});
   }
}

template<class Space, class Storage>
void ValidateBoundStorageExtent(const Space& space, const Storage& storage)
{
   if constexpr (requires { GetVectorSize(storage); })
   {
      const auto extent = GetVectorSize(storage);
      GENDIL_VERIFY(extent >= static_cast<size_t>(GetAlgebraicDofExtent(space)),
         "Persistent field storage is smaller than its required algebraic extent.");
      if constexpr (requires { GetVectorSize(storage.data.Get()); storage.layout.BaseOffset(); })
      {
         // Owner-bound views retain their logical range when the owner resizes.
         // Check that range against current allocation metadata before access.
         const auto owner_extent = GetVectorSize(storage.data.Get());
         const auto offset = static_cast<size_t>(storage.layout.BaseOffset());
         GENDIL_VERIFY(offset <= owner_extent && extent <= owner_extent - offset,
            "Persistent field storage is smaller than its required algebraic extent.");
      }
   }
   else if constexpr (requires { Storage::rank; })
   {
      if constexpr (Storage::rank > 1)
         ValidateElementStorageExtent(GetRestriction(space), storage, space.GetNumberOfFiniteElements());
      // A rank-one view without complete extents is unbounded algebraic storage.
   }
   else if constexpr (requires { std::tuple_size<Storage>::value; })
      ValidateElementStorageExtent(GetRestriction(space), storage, space.GetNumberOfFiniteElements());
   // Raw storage has no allocation length; its existing size precondition is
   // the caller's responsibility. Do not resolve/synchronize owners here.
}

template<class Selected, class... Fields>
void ValidateBoundExtents(const Selected& selected, type_list<Fields...>)
{
   ([&] {
      if constexpr (Fields::storage_role == FieldStorageRole::Bound)
         ValidateBoundStorageExtent(
            RequiredVolumeSpace<Fields::side>(selected.template GetFiniteElementField<Fields::name>().space),
            selected.template GetBoundField<Fields::name>());
   }(), ...);
}

// Endpoints already own their selected spaces. Retain only registrations that
// have no endpoint, such as the active/output spaces of an assembly parent.
// Persistent storage lives in reads alone, without host owner references.
template<class Reads, class Writes, class Selected>
GENDIL_HOST_DEVICE auto KernelSelection(const Selected& selected)
{
   auto fields = std::apply([](const auto&... entries) {
      return ToStaticMap(std::tuple_cat([&] {
         constexpr auto Name = std::remove_cvref_t<decltype(entries)>::key_type::name;
         if constexpr (Reads::template Contains<NameTag<Name>>() ||
                       Writes::template Contains<NameTag<Name>>())
            return std::make_tuple();
         else return std::make_tuple(MakeTrialField<Name>(entries.value.space));
      }()...));
   }, selected.fe_fields.entries);
   return WeakFormContext{selected.domains, selected.interior_face_domains,
      selected.boundary_face_domains, std::move(fields), selected.fields};
}

template<class Storage>
consteval bool IsPreboundElementStorage()
{
   if constexpr (is_view_v<Storage>) { return Storage::rank > 1; }
   else if constexpr (requires { std::tuple_size<Storage>::value; })
   {
      return []<size_t... I>(std::index_sequence<I...>) {
         return (IsPreboundElementStorage<std::tuple_element_t<I, Storage>>() && ...);
      }(std::make_index_sequence<std::tuple_size_v<Storage>>{});
   }
   else { return false; }
}

struct NoInputStorage {};

// Tuple entries are shallow handles that reference the original tensors;
// storing the tensors themselves would create owning row copies.
template<StaticString Name, bool Read, class... Handles>
GENDIL_HOST_DEVICE constexpr auto FindLocalHandle(const std::tuple<Handles...>& handles)
{
   constexpr size_t matches = ((Handles::name == Name && Handles::is_read == Read) + ... + 0);
   static_assert(matches == 1,
      "BindElementDofs requires exactly one local handle for each active read and output.");
   return std::apply([](const auto&... values) {
      return std::tuple_cat([&]() {
         using Handle = std::remove_cvref_t<decltype(values)>;
         if constexpr (Handle::name == Name && Handle::is_read == Read)
         { return std::make_tuple(values); }
         else { return std::make_tuple(); }
      }()...);
   }, handles);
}

template<class List> struct BindLocalReads;
template<class... Fields>
struct BindLocalReads<type_list<Fields...>>
{
   template<class Parent, class Handles>
   GENDIL_HOST_DEVICE static auto Make(const Parent& parent, const Handles& handles)
   { return Add(MakeStaticMap(), parent, handles, type_list<Fields...>{}); }

   template<class Map, class Parent, class Handles>
   GENDIL_HOST_DEVICE static auto Add(Map map, const Parent&, const Handles&, type_list<>)
   { return map; }

   template<class Map, class Parent, class Handles, class Field, class... Rest>
   GENDIL_HOST_DEVICE static auto Add(Map map, const Parent& parent, const Handles& handles,
                                     type_list<Field, Rest...>)
   {
      constexpr auto Name = Field::name;
      if constexpr (Field::storage_role == FieldStorageRole::Bound)
      {
         // The host binding resolved persistent sources before kernel capture.
         // Inspect availability but do not copy that endpoint into an overlay.
         (void)parent.template GetReadField<Name>();
         return Add(std::move(map), parent, handles, type_list<Rest...>{});
      }
      else if constexpr (Map::template Contains<NameTag<Name>>())
      { return Add(std::move(map), parent, handles, type_list<Rest...>{}); }
      else
      {
         const auto& space = parent.template GetSpace<Name>();
         const auto handle = std::get<0>(FindLocalHandle<Name, true>(handles));
         using Space = std::remove_cvref_t<decltype(space)>;
         using Tensor = std::remove_cvref_t<decltype(handle.local)>;
         auto endpoint = FiniteElementFieldView<const Space&, const Tensor&>{space, handle.local};
         return Add(MapAppend(std::move(map), MakeEntry<NameTag<Name>>(endpoint)),
            parent, handles, type_list<Rest...>{});
      }
   }
};

template<class List> struct BindLocalWrites;
template<class... Fields>
struct BindLocalWrites<type_list<Fields...>>
{
   template<class Parent, class Handles>
   GENDIL_HOST_DEVICE static auto Make(const Parent& parent, const Handles& handles)
   { return Add(MakeStaticMap(), parent, handles, type_list<Fields...>{}); }

   template<class Map, class Parent, class Handles>
   GENDIL_HOST_DEVICE static auto Add(Map map, const Parent&, const Handles&, type_list<>)
   { return map; }

   template<class Map, class Parent, class Handles, class Field, class... Rest>
   GENDIL_HOST_DEVICE static auto Add(Map map, const Parent& parent, const Handles& handles,
                                     type_list<Field, Rest...>)
   {
      constexpr auto Name = Field::name;
      if constexpr (Map::template Contains<NameTag<Name>>())
      { return Add(std::move(map), parent, handles, type_list<Rest...>{}); }
      else
      {
         const auto& space = parent.template GetSpace<Name>();
         const auto handle = std::get<0>(FindLocalHandle<Name, false>(handles));
         using Space = std::remove_cvref_t<decltype(space)>;
         using Tensor = std::remove_reference_t<decltype(handle.local)>;
         auto endpoint = FiniteElementFieldView<const Space&, Tensor&>{space, handle.local};
         return Add(MapAppend(std::move(map), MakeEntry<NameTag<Name>>(endpoint)),
            parent, handles, type_list<Rest...>{});
      }
   }
};
} // namespace execution_context_detail

/** Check available storage bounds for the required persistent fields.
 * Sized storage is checked against algebraic extents, including holes and
 * offsets. Prebound element storage is checked against available logical
 * layout extents; raw or unbounded storage retains the caller's allocation-size
 * precondition. This check does not acquire or synchronize vector data.
 */
template<class Requirements, class Selected>
void ValidateBoundFieldExtents(const Selected& selected)
{
   execution_context_detail::ValidateBoundExtents(selected, typename Requirements::reads{});
}

namespace execution_context_detail
{

/**
 * @brief Bind previously acquired input/output access for use across batches.
 *
 * Internal entry for callers that have already validated the selected fields
 * and vector extents. The supplied access objects must address Policy's memory
 * space; their backing storage must remain valid through kernel completion.
 * This overload does not reacquire the common vectors or clear output; persistent
 * owners are still resolved here. Use gendil::BindExecutionContext to validate
 * storage and acquire vector data.
 */
template<class Requirements, class Policy, class Selected, class InputAccess, class OutputAccess>
auto BindExecutionContext(const Selected& selected, const InputAccess& input,
                          const OutputAccess& output)
{
   static_assert(field_requirements_detail::ReadRolesCompatible<typename Requirements::reads>::value,
      "A field cannot be read from both active input and bound storage.");
   auto reads = execution_context_detail::BindReads<typename Requirements::reads>::template
      Make<Policy>(selected, input);
   auto writes = execution_context_detail::BindWrites<typename Requirements::writes>::Make(selected, output);
   auto kernel_selected = execution_context_detail::KernelSelection<decltype(reads), decltype(writes)>(selected);
   return ExecutionContext<decltype(kernel_selected), decltype(reads), decltype(writes)>{
      std::move(kernel_selected), std::move(reads), std::move(writes)};
}

/// Bind already acquired output access for requirements with no active reads.
template<class Requirements, class Policy, class Selected, class OutputAccess>
auto BindExecutionContext(const Selected& selected, const OutputAccess& output)
{
   static_assert(!execution_context_detail::HasActiveReads<typename Requirements::reads>::value,
      "Input-free binding cannot satisfy an active input requirement.");
   return execution_context_detail::BindExecutionContext<Requirements, Policy>(
      selected, execution_context_detail::NoInputStorage{}, output);
}

/** Bind only persistent reads after the caller has validated their storage.
 * Requirements must have no active reads or writes. Assembly uses this parent
 * context before traversal, then supplies local input/output rows per probe.
 * Unbound spaces remain available for those later local bindings.
 */
template<class Requirements, class Policy, class Selected>
   requires (!execution_context_detail::HasActiveReads<typename Requirements::reads>::value &&
             std::is_same_v<typename Requirements::writes, type_list<>>)
auto BindExecutionContext(const Selected& selected)
{
   static_assert(field_requirements_detail::ReadRolesCompatible<typename Requirements::reads>::value,
      "A field cannot be read from both active input and bound storage.");
   auto reads = execution_context_detail::BindReads<typename Requirements::reads>::template
      Make<Policy>(selected, execution_context_detail::NoInputStorage{});
   auto writes = MakeStaticMap();
   auto spaces = execution_context_detail::KernelSelection<decltype(reads), decltype(writes)>(selected);
   return ExecutionContext<decltype(spaces), decltype(reads), decltype(writes)>{
      std::move(spaces), std::move(reads), std::move(writes)};
}

} // namespace execution_context_detail

/**
 * @brief Bind a common input and output vector to the selected named fields.
 *
 * Checks required storage extents with GENDIL_VERIFY before acquiring access.
 * Input and output extents are independent and must cover every corresponding
 * restriction, including offsets and holes. This is storage validation, not a
 * replacement for complete form, shape, domain, or invocation-arity validation.
 *
 * Acquires input once if active reads exist, and read/write output once, in
 * Policy's memory space. With no active reads, input is neither checked nor
 * accessed. Vector access may synchronize data according to the vector type;
 * no numerical evaluation or output clearing is performed here. Keep the
 * underlying allocations stable until kernel use ends (see ExecutionContext).
 */
template<class Requirements, class Policy, class Selected, class Input, class Output>
auto BindExecutionContext(const Selected& selected, const Input& input, Output& output)
{
   ValidateBoundFieldExtents<Requirements>(selected);
   execution_context_detail::ValidateOutputExtent(selected, GetVectorSize(output),
      typename Requirements::writes{});
   if constexpr (execution_context_detail::HasActiveReads<typename Requirements::reads>::value)
   {
      execution_context_detail::ValidateInputExtent(selected, GetVectorSize(input),
         typename Requirements::reads{});
      auto input_access = ReadKernelVector<is_device_configuration_v<Policy>>(input);
      auto output_access = ReadWriteKernelVector<is_device_configuration_v<Policy>>(output);
      return execution_context_detail::BindExecutionContext<Requirements, Policy>(
         selected, input_access, output_access);
   }
   else
   {
      auto output_access = ReadWriteKernelVector<is_device_configuration_v<Policy>>(output);
      return execution_context_detail::BindExecutionContext<Requirements, Policy>(
         selected, output_access);
   }
}

/** Bind an input-free load's output and any persistent reads.
 * Requirements must contain no active reads. Storage checks and output access
 * follow the two-vector overload; no dummy input vector is needed.
 */
template<class Requirements, class Policy, class Selected, class Output>
auto BindExecutionContext(const Selected& selected, Output& output)
{
   ValidateBoundFieldExtents<Requirements>(selected);
   execution_context_detail::ValidateOutputExtent(selected, GetVectorSize(output),
      typename Requirements::writes{});
   auto output_access = ReadWriteKernelVector<is_device_configuration_v<Policy>>(output);
   return execution_context_detail::BindExecutionContext<Requirements, Policy>(
      selected, output_access);
}

/** Bind persistent data for local probes without input or output vectors.
 * Validates bound storage, then delegates to the internal binding overload.
 * Select spaces with the full Requirements first, then bind with
 * bound_only_requirements_t<Requirements>. This retains the spaces needed by
 * subsequent BindElementDofs calls while resolving only persistent storage.
 */
template<class Requirements, class Policy, class Selected>
   requires (!execution_context_detail::HasActiveReads<typename Requirements::reads>::value &&
             std::is_same_v<typename Requirements::writes, type_list<>>)
auto BindExecutionContext(const Selected& selected)
{
   ValidateBoundFieldExtents<Requirements>(selected);
   return execution_context_detail::BindExecutionContext<Requirements, Policy>(selected);
}

/**
 * @brief Use element rows in place of global input/output storage.
 *
 * For requirements that read "u" and write "v", inside the executing scope:
 * @code
 * auto local = BindElementDofs<Requirements>(execution_context,
 *    ReadLocal<"u">(input_row), AddLocal<"v">(output_row));
 * auto& destination = local.GetWriteField<"v">().dofs; // Refers to output_row.
 * @endcode
 * Numerical routines using local now read input_row and add into output_row;
 * persistent reads still use the parent. Every required active read and output
 * needs exactly one matching handle. Rows may also be tuples of component rows.
 * They must already have the layout and orientation expected by the routine:
 * local reads bypass restriction gathering and additional orientation.
 *
 * The parent and rows must be lvalues and outlive the overlay, including any
 * copies or nested overlays. Construct it inside the host/device scope where
 * the rows are used, not on the host for later device capture. No registry or
 * row is copied, and this function performs no initialization or global scatter.
 *
 * Weak-form callers can instead pass the form and operator_context to the
 * overloads in operatorcontext.hpp, which infer Requirements from
 * the already-resolved signatures.
 */
template<class Requirements, class Parent, class... Handles>
   requires requires { typename Parent::execution_context_tag; }
GENDIL_HOST_DEVICE auto BindElementDofs(const Parent& parent, Handles... handles)
{
   static_assert(field_requirements_detail::ReadRolesCompatible<typename Requirements::reads>::value,
      "A field cannot be read from both active input and bound storage.");
   // make_tuple avoids host-only deduction guides in HIP device code.
   const auto local_handles = std::make_tuple(handles...);
   auto reads = execution_context_detail::BindLocalReads<typename Requirements::reads>::Make(parent, local_handles);
   auto writes = execution_context_detail::BindLocalWrites<typename Requirements::writes>::Make(parent, local_handles);
   return ExecutionContext<const Parent&, decltype(reads), decltype(writes)>{parent, reads, writes};
}

template<class Requirements, class Parent, class... Handles>
   requires (!std::is_lvalue_reference_v<Parent>)
auto BindElementDofs(Parent&&, Handles...) = delete;

/**
 * @brief Obtain a field's element DOFs for interpolation or a local operator.
 *
 * A restricted source gathers the entity's row through ReadDofs; the caller
 * must provide the scratch required by that gather. A local source returns a
 * const reference to its original row without gathering. Use decltype(auto)
 * when the caller must preserve that reference instead of copying the tensor.
 * This function does not interpolate values or gradients.
 *
 * This overload uses the entity's current/parent cell index. The facet overload
 * in KernelOperators/TrialSpaceOperators/interpolatefields.hpp uses a selected
 * FacetFieldSideBinding for restriction and orientation, with the same named source.
 */
template<class Kernel, class Source, class Entity>
GENDIL_HOST_DEVICE decltype(auto) ReadFieldDofs(Kernel& kernel, const Source& source, const Entity& entity)
{
   if constexpr (Source::is_local)
      return static_cast<const std::remove_reference_t<decltype(source.dofs)>&>(source.dofs);
   else
   {
      const auto index = [&]() {
         if constexpr (requires { entity.element_context.element_index; })
            return entity.element_context.element_index;
         else return entity.element_index;
      }();
      const auto& space = GetMinusFacetFieldVolumeSpace(source.space);
      if constexpr (is_face_field_binding_v<std::remove_cvref_t<decltype(source.space)>> &&
                    !execution_context_detail::IsPreboundElementStorage<
                       std::remove_cvref_t<decltype(source.dofs)>>())
         return ReadDofs(kernel, space, index, MakeElementTensorView(space, source.dofs));
      else return ReadDofs(kernel, space, index, source.dofs);
   }
}

} // namespace gendil

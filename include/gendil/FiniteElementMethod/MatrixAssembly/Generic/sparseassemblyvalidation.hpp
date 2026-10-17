// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrbackendconfiguration.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/matrixassemblytype.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/localrestrictionentrylayout.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/SGBSR/sgbsrgatherscatter.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/weakformvalidation.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/localdofoperations.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/signatures.hpp"
#include "gendil/Utilities/KernelContext/batchingeligibility.hpp"
#include "gendil/Utilities/KernelContext/kernelcontext.hpp"
#include "gendil/Utilities/KernelContext/kernelcontexttraits.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

namespace details {
template<class Signature, class Context>
consteval bool SparseAffinitySupported()
{
   if constexpr (Signature::is_zero || Signature::active_affinity == ActiveAffinity::Linear)
      return true;
   else if constexpr (Signature::active_affinity == ActiveAffinity::Affine &&
                      is_boundary_facets_domain<typename Signature::domain_type>::value)
   {
      constexpr auto Name = Signature::domain_type::name;
      if constexpr (Context::template HasDomain<Name>())
      {
         using Domain = std::remove_cvref_t<decltype(
            std::declval<const Context&>().template GetDomain<Name>().domain)>;
         // Existing local boundary assembly explicitly subtracts F(0).
         // Cell and canonical partition assembly do not implement that step.
         return is_mesh_integration_domain_v<Domain>;
      }
      else return false;
   }
   else return false;
}

template<class Context, class... Signatures>
consteval void ValidateSparseLinearity(type_list<Signatures...>)
{
   static_assert((SparseAffinitySupported<Signatures, Context>() && ...),
      "Sparse linear assembly requires joint linear dependence on its active "
      "trial; only existing mesh-boundary assembly supports subtracting an "
      "affine load. Separate other loads or freeze nonlinear coefficient inputs.");
}
} // namespace details

template<class Form, class WFContext>
struct weak_form_uses_partition_integration_domain : std::false_type {};

template<class Domain, FieldExpr Expr, class WFContext>
struct weak_form_uses_partition_integration_domain<
   Integrand<Domain, Expr>,
   WFContext>
{
   static constexpr auto Name = Domain::name;
   using Context = std::remove_cvref_t<WFContext>;

   static consteval bool Get()
   {
      if constexpr (!Context::template HasDomain<Name>())
      {
         return false;
      }
      else
      {
         using NormalizedDomain = std::remove_cvref_t<
            decltype(std::declval<const Context&>().template GetDomain<Name>())>;
         using IntegrationDomain = std::remove_cvref_t<
            decltype(std::declval<NormalizedDomain>().domain)>;
         return is_partition_integration_domain_v<IntegrationDomain>;
      }
   }

   static constexpr bool value = Get();
};

template<class... Terms, class Context>
struct weak_form_uses_partition_integration_domain<std::tuple<Terms...>, Context>
   : std::bool_constant<(weak_form_uses_partition_integration_domain<std::remove_cvref_t<Terms>, Context>::value || ...)> {};
template<class Terms, class Symbols, class Context>
struct weak_form_uses_partition_integration_domain<SumFormExpr<Terms, Symbols>, Context>
   : weak_form_uses_partition_integration_domain<Terms, Context> {};

template<class WeakForm, class WFContext>
inline constexpr bool weak_form_uses_partition_integration_domain_v =
   weak_form_uses_partition_integration_domain<
      std::remove_cvref_t<WeakForm>,
      std::remove_cvref_t<WFContext>>::value;

namespace details {

template<class Space>
concept RawCOOCoordinateAssemblySpace =
   requires { typename std::remove_cvref_t<Space>::restriction_type; } &&
   ElementDoFRestriction<
      typename std::remove_cvref_t<Space>::restriction_type> &&
   has_static_expanded_restriction_rows_v<std::remove_cvref_t<Space>>;

template<class CellSpacesTuple, size_t... I>
consteval bool IsRawCOOPartitionCellAssemblySpace(
   std::index_sequence<I...>)
{
   return (
      RawCOOCoordinateAssemblySpace<
         std::tuple_element_t<I, CellSpacesTuple>> && ...);
}

} // namespace details

template < typename FESpace >
struct IsRawCOOCellAssemblySpace
{
   using Space = std::remove_cvref_t< FESpace >;
   static constexpr bool value = []
   {
      if constexpr ( is_mixed_finite_element_space_v<Space> )
      {
         using CellSpaces = typename Space::cell_spaces_type;
         return details::IsRawCOOPartitionCellAssemblySpace<CellSpaces>(
            std::make_index_sequence<std::tuple_size_v<CellSpaces>>{});
      }
      else
      {
         return details::RawCOOCoordinateAssemblySpace<Space>;
      }
   }();
};

template < typename FESpace >
struct IsRawCOOFaceAssemblySpace
{
   using Space = std::remove_cvref_t< FESpace >;
   static constexpr bool value = []
   {
      if constexpr ( is_mixed_finite_element_space_v<Space> )
      {
         using CellSpaces = typename Space::cell_spaces_type;
         return details::IsRawCOOPartitionCellAssemblySpace<CellSpaces>(
            std::make_index_sequence<std::tuple_size_v<CellSpaces>>{});
      }
      else
      {
         return details::RawCOOCoordinateAssemblySpace<Space>;
      }
   }();
};

namespace details {

template<class Signature>
consteval bool SparseAssemblyPairSupported()
{
   if constexpr (weak_form_traits::list_size<typename Signature::formal_inputs>::value != 1 ||
                 weak_form_traits::list_size<typename Signature::formal_outputs>::value != 1 ||
                 weak_form_traits::list_size<typename Signature::callable_inputs>::value != 1 ||
                 weak_form_traits::list_size<typename Signature::logical_outputs>::value != 1)
      return false;
   else
   {
      using Input = weak_form_traits::only_field_t<typename Signature::formal_inputs>;
      using Output = weak_form_traits::only_field_t<typename Signature::formal_outputs>;
      using Callable = weak_form_traits::only_field_t<typename Signature::callable_inputs>;
      using Logical = weak_form_traits::only_field_t<typename Signature::logical_outputs>;
      if constexpr (Input::name != Callable::name || Output::name != Logical::name)
         return false;
      else return []<class... Terms>(type_list<Terms...>) {
         return ([]<class Term>() {
            if constexpr (Term::is_zero) return true;
            else if constexpr (!weak_form_traits::has_single_field_pair<Term>::value)
               return false;
            else
            {
               using Trial = weak_form_traits::only_field_t<typename Term::declared_inputs>;
               using Test = weak_form_traits::only_field_t<typename Term::declared_outputs>;
               return Trial::name == Input::name && Test::name == Output::name &&
                  Trial::shape == Input::shape && Test::shape == Output::shape;
            }
         }.template operator()<Terms>() && ...);
      }(typename Signature::integrands{});
   }
}

template<class WeakForm>
struct SparseAssemblyFormTraits
{
   using form_type = std::remove_cvref_t<WeakForm>;
   using signature = form_signature_t<form_type>;
   using inputs = typename signature::callable_inputs;
   using outputs = typename signature::logical_outputs;
   static constexpr bool has_active_callback = []<class... Reads>(type_list<Reads...>) {
      return (has_provenance(Reads::provenance, NamedFieldProvenance::CoefficientInput) || ...);
   }(typename signature::active_inputs{});
   static constexpr bool is_zero = signature::is_zero;
   static constexpr bool has_trial = weak_form_traits::list_size<inputs>::value == 1;
   static constexpr bool has_test = weak_form_traits::list_size<outputs>::value == 1;
   static constexpr bool has_single_pair = SparseAssemblyPairSupported<signature>();
   static constexpr bool compatible_zero_pair = !is_zero || has_single_pair;
   static constexpr bool has_active_domain = has_cell_contributions_v<form_type> ||
      has_boundary_facet_contributions_v<form_type> || has_interior_facet_contributions_v<form_type>;
   static constexpr bool has_face_terms = has_boundary_facet_contributions_v<form_type> ||
      has_interior_facet_contributions_v<form_type>;
   static constexpr bool value = signature::valid && has_single_pair && has_active_domain;
};

template<class Form>
consteval void ValidateSparseLinearAssemblyCoefficientInputs()
{
   static_assert(!SparseAssemblyFormTraits<Form>::has_active_callback,
      "Coefficient expression depends on active trial field during sparse "
      "linear assembly. This is nonlinear or ambiguous. Use a supplied frozen "
      "field with a distinct name, for example \"u_lagged\".");
}

template<
   MatrixAssemblyType Format,
   SparseAssemblyMode Mode,
   class WeakForm,
   class WFContext>
consteval void ValidateSparseAssemblyDomainSupport()
{
   if constexpr (
      weak_form_uses_partition_integration_domain_v<WeakForm, WFContext>)
   {
      static_assert(
         Format == MatrixAssemblyType::RawCOO &&
            Mode == SparseAssemblyMode::Full,
         "Partition sparse assembly is supported only by full RawCOO "
         "assembly and its derived COO/CSR/CSC/HypreCSR conversions.");
   }
}

template<
   MatrixAssemblyType Format,
   SparseAssemblyMode Mode,
   class WeakForm>
consteval void ValidateSparseAssemblyFormContract()
{
   using Traits = SparseAssemblyFormTraits<WeakForm>;
   static_assert(Traits::signature::valid, "Weak form has invalid result, output, field, or entity semantics.");
   static_assert(Traits::is_zero || Traits::has_single_pair,
      "Sparse assembly requires one explicit trial/test pair in every nonzero original integrand and the same pair throughout the form.");
   static_assert(Traits::compatible_zero_pair,
      "Zero sparse assembly requires one retained compatible trial/test pair and a declared column space.");
   if constexpr (Traits::has_active_domain)
   {
      // Removing numerical zeros must not widen the single-domain assembly contract.
      [[maybe_unused]] constexpr auto DomainName = local_facet_assembly_domain_name_v<WeakForm>;
   }

   if constexpr (Format == MatrixAssemblyType::RawCOO)
   {
      if constexpr (Mode == SparseAssemblyMode::Full)
      {
         static_assert(
            Traits::has_trial,
            "GenericAssembly<RawCOO>: missing TrialSpace in integrand.");
         static_assert(
            Traits::has_test,
            "GenericAssembly<RawCOO>: missing TestSpace in integrand.");
         static_assert(
            Traits::has_active_domain,
            "GenericAssembly<RawCOO> requires at least one active weak-form domain.");
      }
      else
      {
         static_assert(
            Traits::has_trial,
            "GenericRawCOOElementBlockDiagonalAssembly: missing TrialSpace in integrand.");
         static_assert(
            Traits::has_test,
            "GenericRawCOOElementBlockDiagonalAssembly: missing TestSpace in integrand.");
         static_assert(
            Traits::has_active_domain,
            "GenericRawCOOElementBlockDiagonalAssembly requires at least one active weak-form domain.");
      }
   }
   else if constexpr (Format == MatrixAssemblyType::BSR)
   {
      if constexpr (Mode == SparseAssemblyMode::Full)
      {
         static_assert(
            Traits::has_trial,
            "GenericBSRAssembly: missing TrialSpace in integrand.");
         static_assert(
            Traits::has_test,
            "GenericBSRAssembly: missing TestSpace in integrand.");
      }
      else
      {
         static_assert(
            Traits::has_trial,
            "GenericBSRElementBlockDiagonalAssembly: missing TrialSpace in integrand.");
         static_assert(
            Traits::has_test,
            "GenericBSRElementBlockDiagonalAssembly: missing TestSpace in integrand.");
      }
      static_assert(
         Traits::has_active_domain,
         "BSR assembly requires at least one active weak-form domain.");
   }
   else if constexpr (Format == MatrixAssemblyType::SGBSR)
   {
      if constexpr (Mode == SparseAssemblyMode::Full)
      {
         static_assert(
            Traits::has_trial,
            "GenericAssembly<SGBSR>: missing TrialSpace in integrand.");
         static_assert(
            Traits::has_test,
            "GenericAssembly<SGBSR>: missing TestSpace in integrand.");
      }
      else
      {
         static_assert(
            Traits::has_trial,
            "GenericSGBSRElementBlockDiagonalAssembly: missing TrialSpace in integrand.");
         static_assert(
            Traits::has_test,
            "GenericSGBSRElementBlockDiagonalAssembly: missing TestSpace in integrand.");
      }
      static_assert(
         Traits::has_active_domain,
         "SGBSR assembly requires at least one active weak-form domain.");
   }
}

template<
   SparseAssemblyMode Mode,
   class WeakForm>
consteval void ValidateSparseAssemblyExecutionFormContract()
{
   using Traits = SparseAssemblyFormTraits<WeakForm>;
   static_assert(Traits::signature::valid, "Weak form has invalid result, output, field, or entity semantics.");
   static_assert(Traits::is_zero || Traits::has_single_pair,
      "Sparse assembly requires one explicit trial/test pair in every nonzero original integrand and the same pair throughout the form.");
   static_assert(Traits::compatible_zero_pair,
      "Zero sparse assembly requires one retained compatible trial/test pair and a declared column space.");
   if constexpr (Traits::has_active_domain)
   {
      // Removing numerical zeros must not widen the single-domain assembly contract.
      [[maybe_unused]] constexpr auto DomainName = local_facet_assembly_domain_name_v<WeakForm>;
   }
   if constexpr (Mode == SparseAssemblyMode::Full)
   {
      static_assert(
         Traits::has_trial,
         "GenericAssembly: missing TrialSpace in integrand.");
      static_assert(
         Traits::has_test,
         "GenericAssembly: missing TestSpace in integrand.");
   }
   else
   {
      static_assert(
         Traits::has_trial,
         "GenericElementBlockDiagonalAssembly: missing TrialSpace in integrand.");
      static_assert(
         Traits::has_test,
         "GenericElementBlockDiagonalAssembly: missing TestSpace in integrand.");
   }
   static_assert(
      Traits::has_active_domain,
      "Sparse assembly requires at least one active weak-form domain.");
}

template<class WeakForm, class WeakFormContext>
consteval bool SparseAssemblyContextBindingsAvailable()
{
   using Traits = SparseAssemblyFormTraits<WeakForm>;
   using Context = std::remove_cvref_t<WeakFormContext>;
   if constexpr (!Traits::has_trial || !Traits::has_test)
   {
      return false;
   }
   else
   {
      using Input = weak_form_traits::only_field_t<typename Traits::inputs>;
      using Output = weak_form_traits::only_field_t<typename Traits::outputs>;
      return Context::template HasFiniteElementField<Input::name>() &&
             Context::template HasFiniteElementField<Output::name>();
   }
}

template<class WeakForm, class WeakFormContext>
consteval bool SparseAssemblyUsesMeshDomain()
{
   using namespace generic_operator_detail;
   if constexpr (!SparseAssemblyFormTraits<WeakForm>::has_active_domain)
   {
      return false;
   }
   else
   {
      return generic_operator_domain_kind_v<WeakForm, WeakFormContext> ==
             GenericOperatorDomainKind::Mesh;
   }
}

template<
   MatrixAssemblyType Format,
   SparseAssemblyMode Mode,
   class WeakForm,
   class TrialSpace,
   class TestSpace>
struct SparseAssemblySpaceContract
{
   static constexpr bool value = true;
};

template<
   SparseAssemblyMode Mode,
   class WeakForm,
   class TrialSpace,
   class TestSpace>
struct SparseAssemblySpaceContract<
   MatrixAssemblyType::RawCOO,
   Mode,
   WeakForm,
   TrialSpace,
   TestSpace>
{
   static constexpr bool has_face_terms =
      SparseAssemblyFormTraits<WeakForm>::has_face_terms;
   static constexpr bool raw_coo_supported =
      (!has_face_terms &&
       IsRawCOOCellAssemblySpace<TrialSpace>::value &&
       IsRawCOOCellAssemblySpace<TestSpace>::value) ||
      (has_face_terms &&
       IsRawCOOFaceAssemblySpace<TrialSpace>::value &&
       IsRawCOOFaceAssemblySpace<TestSpace>::value);
   static constexpr bool value = raw_coo_supported;
};

template<
   SparseAssemblyMode Mode,
   class WeakForm,
   class TrialSpace,
   class TestSpace>
struct SparseAssemblySpaceContract<
   MatrixAssemblyType::SGBSR,
   Mode,
   WeakForm,
   TrialSpace,
   TestSpace>
{
   static constexpr bool sgbsr_mappings_supported =
      DefaultBsrMappingSpace< TrialSpace > &&
      DefaultBsrMappingSpace< TestSpace >;
   static constexpr bool sgbsr_space_pair_supported =
      sgbsr_mappings_supported;
   static constexpr bool value = sgbsr_space_pair_supported;
};

template<
   MatrixAssemblyType Format,
   SparseAssemblyMode Mode,
   class WeakForm,
   class TrialSpace,
   class TestSpace>
consteval void ValidateSparseAssemblySpaceContract()
{
   using Contract =
      SparseAssemblySpaceContract<
         Format,
         Mode,
         WeakForm,
         TrialSpace,
         TestSpace>;

   if constexpr (Format == MatrixAssemblyType::RawCOO)
   {
      if constexpr (Mode == SparseAssemblyMode::Full)
      {
         static_assert(
            Contract::raw_coo_supported,
            "GenericAssembly<RawCOO> requires an exact compile-time entry "
            "count for every native trial and test restriction row. Dynamic-"
            "cardinality coordinate planning and unsupported nonconforming FE "
            "transforms remain unsupported.");
      }
      else
      {
         static_assert(
            Contract::raw_coo_supported,
            "GenericRawCOOElementBlockDiagonalAssembly requires exact compile-"
            "time entry counts for all native restriction rows. Mixed-space "
            "faces, unsupported nonconforming FE transforms, global face "
            "traversal, dynamic-cardinality rows, and variable-size hp "
            "emission remain unsupported.");
      }
   }
   else if constexpr (Format == MatrixAssemblyType::SGBSR)
   {
      if constexpr (Mode == SparseAssemblyMode::Full)
      {
         static_assert(
            Contract::sgbsr_mappings_supported,
            "SGBSR GenericAssembly requires completed trial and test "
            "restrictions with backend-access validation." );
         if constexpr (Contract::sgbsr_mappings_supported)
         {
            static_assert(
               Contract::sgbsr_space_pair_supported,
               "SGBSR GenericAssembly requires independently supported trial "
               "gather and test scatter mappings." );
         }
      }
      else
      {
         static_assert(
            Contract::sgbsr_mappings_supported,
            "SGBSR element block-diagonal assembly requires independently "
            "supported trial gather and test scatter restrictions with "
            "backend-access validation." );
         if constexpr (Contract::sgbsr_mappings_supported)
         {
            static_assert(
               Contract::sgbsr_space_pair_supported,
               "SGBSR element block-diagonal assembly requires independently "
               "supported trial gather and test scatter mappings." );
         }
      }
   }
}

template<MatrixAssemblyType Format, class KernelPolicy>
struct SparseAssemblyKernelContract
{
   static constexpr bool placement_supported =
      is_host_configuration_v<KernelPolicy> !=
      is_device_configuration_v<KernelPolicy>;
   static constexpr bool batching_supported =
      Format == MatrixAssemblyType::RawCOO ||
      is_unbatched_operator_configuration_allowed_v<KernelPolicy>;
   static constexpr bool value =
      placement_supported && batching_supported;
};

template<MatrixAssemblyType Format, class KernelPolicy>
consteval void ValidateSparseAssemblyKernelContract()
{
   using Contract = SparseAssemblyKernelContract<Format, KernelPolicy>;
   if constexpr (Format == MatrixAssemblyType::RawCOO)
   {
      static_assert(
         Contract::placement_supported,
         "RawCOO assembly requires a host or device kernel policy.");
   }
   else if constexpr (Format == MatrixAssemblyType::BSR)
   {
      static_assert(
         Contract::placement_supported,
         "BSR assembly requires a host or device kernel policy.");
   }
   else
   {
      static_assert(
         Contract::placement_supported,
         "SGBSR assembly requires a host or device kernel policy.");
   }
   static_assert(
      Contract::batching_supported,
      "This operator has not been audited for batched device execution. "
      "Use BatchSize == 1 or audit this operator before enabling "
      "BatchSize > 1.");
}

template<class KernelPolicy>
consteval void ValidateSparseAssemblyExecutionKernelContract()
{
   static_assert(
      is_host_configuration_v<KernelPolicy> !=
         is_device_configuration_v<KernelPolicy>,
      "Sparse assembly execution requires a host or device kernel policy.");
}

template<
   MatrixAssemblyType Format,
   class Backend,
   class TrialSpace,
   class TestSpace>
struct SparseAssemblyBackendContract
{
   using TrialShapeFunctions =
      typename TrialSpace::finite_element_type::shape_functions;
   using TestShapeFunctions =
      typename TestSpace::finite_element_type::shape_functions;
   static constexpr GlobalIndex ntrial =
      LocalDofCount<TrialShapeFunctions>();
   static constexpr GlobalIndex ntest =
      LocalDofCount<TestShapeFunctions>();
   static constexpr bool value = [] {
      if constexpr (
         Format == MatrixAssemblyType::BSR ||
         Format == MatrixAssemblyType::SGBSR)
      {
         return is_bsr_assembly_backend_compatible_v<
            Backend,
            ntest,
            ntrial>;
      }
      else
      {
         return true;
      }
   }();
};

template<class Backend, class TrialSpace, class TestSpace>
struct SparseAssemblyBackendContract<
   MatrixAssemblyType::RawCOO,
   Backend,
   TrialSpace,
   TestSpace>
{
   static constexpr bool value = true;
};

template<
   MatrixAssemblyType Format,
   class Backend,
   class TrialSpace,
   class TestSpace>
consteval void ValidateSparseAssemblyBackendContract()
{
   if constexpr (
      Format == MatrixAssemblyType::BSR ||
      Format == MatrixAssemblyType::SGBSR)
   {
      using Contract =
         SparseAssemblyBackendContract<
            Format,
            Backend,
            TrialSpace,
            TestSpace>;
      ValidateBSRAssemblyBackendCompatibility<
         Backend,
         Contract::ntest,
         Contract::ntrial>();
   }
}

template<
   MatrixAssemblyType Format,
   SparseAssemblyMode Mode,
   class KernelPolicy,
   class Backend,
   class WeakForm,
   class WeakFormContext>
consteval bool SparseAssemblyCanInstantiate()
{
   using FormTraits = SparseAssemblyFormTraits<WeakForm>;
   using Context = std::remove_cvref_t<WeakFormContext>;
   constexpr bool uses_partition =
      weak_form_uses_partition_integration_domain_v<WeakForm, Context>;
   constexpr bool domain_supported =
      SparseAssemblyUsesMeshDomain<WeakForm, Context>() ||
      (Format == MatrixAssemblyType::RawCOO &&
       Mode == SparseAssemblyMode::Full &&
       uses_partition);
   if constexpr (
      !FormTraits::value ||
      FormTraits::has_active_callback ||
      !domain_supported ||
      !SparseAssemblyContextBindingsAvailable<WeakForm, Context>())
   {
      return false;
   }
   else
   {
      constexpr auto TrialName = weak_form_traits::only_trial_field_name_v< WeakForm >;
      constexpr auto TestName = weak_form_traits::only_test_field_name_v< WeakForm >;
      using TrialSpace = std::remove_cvref_t<
         decltype(
            std::declval<const Context&>()
               .template GetFiniteElementField<TrialName>().space)>;
      using TestSpace = std::remove_cvref_t<
         decltype(
            std::declval<const Context&>()
               .template GetFiniteElementField<TestName>().space)>;
      using SpaceContract =
         SparseAssemblySpaceContract<
            Format,
            Mode,
            WeakForm,
            TrialSpace,
            TestSpace>;
      using BackendContract =
         SparseAssemblyBackendContract<
            Format,
            Backend,
            TrialSpace,
            TestSpace>;
      using KernelContract =
         SparseAssemblyKernelContract<Format, KernelPolicy>;
      return SpaceContract::value &&
             BackendContract::value &&
             KernelContract::value;
   }
}

template<
   MatrixAssemblyType Format,
   SparseAssemblyMode Mode,
   class KernelPolicy,
   class Backend = Empty,
   class WeakForm,
   class WeakFormContext>
auto ValidateSparseAssemblyInputs(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx)
{
   using FormTraits = SparseAssemblyFormTraits<WeakForm>;

   ValidateSparseAssemblyFormContract<Format, Mode, WeakForm>();
   if constexpr (FormTraits::value)
   {
      ValidateSparseLinearity<WeakFormContext>(
         typename form_signature_t<WeakForm>::integrands{});
      ValidateSparseLinearAssemblyCoefficientInputs<WeakForm>();
      if constexpr (!FormTraits::has_active_callback)
      {
         ValidateSparseAssemblyDomainSupport<
            Format,
            Mode,
            WeakForm,
            WeakFormContext>();
         constexpr bool domain_supported =
            SparseAssemblyUsesMeshDomain<
               WeakForm,
               WeakFormContext>() ||
            (Format == MatrixAssemblyType::RawCOO &&
             Mode == SparseAssemblyMode::Full &&
             weak_form_uses_partition_integration_domain_v<
                WeakForm,
                WeakFormContext>);
         if constexpr (domain_supported)
         {
            ValidateWeakFormContext(weak_form, wf_ctx);

            if constexpr (
               SparseAssemblyContextBindingsAvailable<
                  WeakForm,
                  WeakFormContext>())
            {
               constexpr auto TrialName = weak_form_traits::only_trial_field_name_v< WeakForm >;
               constexpr auto TestName = weak_form_traits::only_test_field_name_v< WeakForm >;
               const auto& trial_space =
                  wf_ctx.template GetFiniteElementField<TrialName>().space;
               const auto& test_space =
                  wf_ctx.template GetFiniteElementField<TestName>().space;
               using TrialSpace =
                  std::remove_cvref_t<decltype(trial_space)>;
               using TestSpace =
                  std::remove_cvref_t<decltype(test_space)>;
               using SpaceContract =
                  SparseAssemblySpaceContract<
                     Format,
                     Mode,
                     WeakForm,
                     TrialSpace,
                     TestSpace>;

               ValidateSparseAssemblySpaceContract<
                  Format,
                  Mode,
                  WeakForm,
                  TrialSpace,
                  TestSpace>();
               if constexpr (SpaceContract::value)
               {
                  ValidateSparseAssemblyBackendContract<
                     Format,
                     Backend,
                     TrialSpace,
                     TestSpace>();
               }
               ValidateSparseAssemblyKernelContract<Format, KernelPolicy>();
            }
         }
      }
   }

   return std::bool_constant<
      SparseAssemblyCanInstantiate<
         Format,
         Mode,
         KernelPolicy,
         Backend,
         WeakForm,
         WeakFormContext>()>{};
}

template<
   class KernelPolicy,
   SparseAssemblyMode Mode = SparseAssemblyMode::Full,
   class WeakForm,
   class WeakFormContext>
void ValidateSparseAssemblyExecutionInputs(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx)
{
   using FormTraits = SparseAssemblyFormTraits<WeakForm>;
   ValidateSparseAssemblyExecutionFormContract<Mode, WeakForm>();
   if constexpr (FormTraits::value)
   {
      ValidateSparseLinearity<WeakFormContext>(
         typename form_signature_t<WeakForm>::integrands{});
      ValidateSparseLinearAssemblyCoefficientInputs<WeakForm>();
      if constexpr (!FormTraits::has_active_callback)
      {
         ValidateSparseAssemblyDomainSupport<
            MatrixAssemblyType::RawCOO,
            Mode,
            WeakForm,
            WeakFormContext>();
         ValidateWeakFormContext(weak_form, wf_ctx);
         ValidateSparseAssemblyExecutionKernelContract<KernelPolicy>();
      }
   }
}

// A numerical zero still requires a supported insertion target. Check its
// interface using row types only; no local vector or numerical probe is made.
template<class KernelPolicy, SparseAssemblyMode Mode = SparseAssemblyMode::Full,
         class Form, class Context, class Target>
void ValidateSparseAssemblyExecutionInputs(
   const Form& form, const Context& context, const Target&)
{
   ValidateSparseAssemblyExecutionInputs<KernelPolicy, Mode>(form, context);
   using Traits = SparseAssemblyFormTraits<Form>;
   if constexpr (is_zero<Form> && Traits::value &&
                 SparseAssemblyContextBindingsAvailable<Form, Context>())
   {
      constexpr auto TrialName = weak_form_traits::only_trial_field_name_v< Form >;
      constexpr auto TestName = weak_form_traits::only_test_field_name_v< Form >;
      using Trial = std::remove_cvref_t<decltype(context.template GetFiniteElementField<TrialName>().space)>;
      using Test = std::remove_cvref_t<decltype(context.template GetFiniteElementField<TestName>().space)>;
      using ShapeFunctions = finite_element_space_shape_functions_t<Trial>;
      using Kernel = KernelContext<KernelPolicy, 0>;
      using Row = decltype(MakeZeroElementVector(std::declval<const Kernel&>(), std::declval<const Test&>()));
      constexpr size_t components = [] {
         if constexpr (is_vector_shape_functions_v<ShapeFunctions>) return ShapeFunctions::vector_dim;
         else return 1;
      }();
      []<size_t... Component>(std::index_sequence<Component...>) {
         ([]<size_t C>() {
            using Shape = component_dof_shape_t<ShapeFunctions, C>;
            using Threads = subsequence_t<Shape, typename Kernel::template threaded_dimensions<Shape::size()>>;
            using Registers = subsequence_t<Shape, typename Kernel::template register_dimensions<Shape::size()>>;
            using Dof = LocalDofDescriptor<C, is_vector_shape_functions_v<ShapeFunctions>,
               Threads, Registers, Threads::size(), Registers::size()>;
            static_assert(requires(const Kernel& kernel, const Trial& trial, const Test& test,
                                   const Dof& dof, const Row& row, Target& target) {
               AddSparseMatrixEntry(kernel, trial, test, GlobalIndex{}, dof, row, target);
            }, "Zero sparse assembly requires a supported sparse insertion target.");
         }.template operator()<Component>(), ...);
      }(std::make_index_sequence<components>{});
   }
}

template<class WeakForm, class WeakFormContext>
auto ValidateDefaultBSRBackendSelectionInputs()
{
   using Form = std::remove_cvref_t<WeakForm>;
   using Context = std::remove_cvref_t<WeakFormContext>;
   using Traits = SparseAssemblyFormTraits<Form>;
   static_assert(Traits::compatible_zero_pair,
      "Zero sparse assembly requires one retained compatible trial/test pair and a declared column space.");

   static_assert(
      Traits::has_trial,
      "Default BSR backend selection requires a TrialSpace in the integrand.");
   static_assert(
      Traits::has_test,
      "Default BSR backend selection requires a TestSpace in the integrand.");
   if constexpr (Traits::has_trial && Traits::has_test)
   {
      using Input = weak_form_traits::only_field_t<typename Traits::inputs>;
      using Output = weak_form_traits::only_field_t<typename Traits::outputs>;
      static_assert(
         Context::template HasFiniteElementField<Input::name>(),
         "Default BSR backend selection requires the trial field in the WeakFormContext.");
      static_assert(
         Context::template HasFiniteElementField<Output::name>(),
         "Default BSR backend selection requires the test field in the WeakFormContext.");
   }

   // Query declared field names only when they agree with the assembly pair.
   return std::bool_constant<
      Traits::has_single_pair &&
      SparseAssemblyContextBindingsAvailable<Form, Context>()>{};
}

} // namespace details

} // namespace gendil

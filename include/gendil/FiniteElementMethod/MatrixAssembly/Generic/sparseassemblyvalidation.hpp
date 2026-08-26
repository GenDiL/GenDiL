// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrbackendconfiguration.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/matrixassemblytype.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/SGBSR/sgbsrgatherscatter.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/restrictedweakformcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperatortraits.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"
#include "gendil/Utilities/KernelContext/batchingeligibility.hpp"
#include "gendil/Utilities/KernelContext/kernelcontexttraits.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

template<class WeakForm>
consteval void ValidateSparseLinearAssemblyCoefficientInputs()
{
   static_assert(
      !has_active_trial_coefficient_dependency_v<WeakForm>,
      "Coefficient expression depends on active trial field during sparse "
      "linear assembly. This is nonlinear or ambiguous. Use a supplied frozen "
      "field with a distinct name, for example \"u_lagged\".");
}

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
      if constexpr (!Context::template has_domain<Name>())
      {
         return false;
      }
      else
      {
         using NormalizedDomain = std::remove_cvref_t<
            decltype(std::declval<const Context&>().template domain<Name>())>;
         using IntegrationDomain = std::remove_cvref_t<
            decltype(std::declval<NormalizedDomain>().domain)>;
         return is_partition_integration_domain_v<IntegrationDomain>;
      }
   }

   static constexpr bool value = Get();
};

template<class Key, class T, class WFContext>
struct weak_form_uses_partition_integration_domain<
   Entry<Key, T>,
   WFContext>
   : weak_form_uses_partition_integration_domain<T, WFContext> {};

template<class... Entries, class WFContext>
struct weak_form_uses_partition_integration_domain<
   StaticMap<Entries...>,
   WFContext>
   : std::bool_constant<
        (
           weak_form_uses_partition_integration_domain<
              Entries,
              WFContext>::value ||
           ...)> {};

template<class Map, class WFContext>
struct weak_form_uses_partition_integration_domain<
   SumFormExpr<Map>,
   WFContext>
   : weak_form_uses_partition_integration_domain<Map, WFContext> {};

template<class WeakForm, class WFContext>
inline constexpr bool weak_form_uses_partition_integration_domain_v =
   weak_form_uses_partition_integration_domain<
      std::remove_cvref_t<WeakForm>,
      std::remove_cvref_t<WFContext>>::value;

namespace details {

template<class Restriction>
concept RawCOOCoordinateRestriction =
   ElementDoFRestriction<Restriction> &&
   static_restriction_entry_count_v<Restriction> == 1 &&
   restriction_supports_element_reference_view_v<Restriction>;

template<class CellSpacesTuple, size_t... I>
consteval bool IsRawCOOPartitionCellAssemblySpace(
   std::index_sequence<I...>)
{
   return (
      RawCOOCoordinateRestriction<
          typename std::tuple_element_t<
             I,
             CellSpacesTuple>::restriction_type> && ...);
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
         using Restriction = typename Space::restriction_type;
         return details::RawCOOCoordinateRestriction<Restriction>;
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
         using Restriction = typename Space::restriction_type;
         return details::RawCOOCoordinateRestriction<Restriction>;
      }
   }();
};

namespace details {

enum class SparseAssemblyMode
{
   Full,
   ElementBlockDiagonal
};

template<class WeakForm>
struct SparseAssemblyFormTraits
{
   using form_type = std::remove_cvref_t<WeakForm>;
   static constexpr auto trial_name = requirements<form_type>::trial_name;
   static constexpr auto test_name = requirements<form_type>::test_name;
   static constexpr bool has_trial =
      trial_name != StaticString{"Error"};
   static constexpr bool has_test =
      test_name != StaticString{"Error"};
   static constexpr bool has_active_domain =
      has_cell_contributions_v<form_type> ||
      has_boundary_facet_contributions_v<form_type> ||
      has_interior_facet_contributions_v<form_type>;
   static constexpr bool has_face_terms =
      has_boundary_facet_contributions_v<form_type> ||
      has_interior_facet_contributions_v<form_type>;
   static constexpr bool value =
      has_trial && has_test && has_active_domain;
};

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
      return Context::template has_fe_field<Traits::trial_name>() &&
             Context::template has_fe_field<Traits::test_name>();
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
            "GenericAssembly<RawCOO> requires statically one-entry, unit-weight "
            "reference-addressable completed restrictions. Unsupported "
            "nonconforming FE transforms, multi-entry rows, and variable-size "
            "runtime emission remain unsupported.");
      }
      else
      {
         static_assert(
            Contract::raw_coo_supported,
            "GenericRawCOOElementBlockDiagonalAssembly requires statically "
            "one-entry, unit-weight reference-addressable completed "
            "restrictions. Mixed-space faces, unsupported "
            "nonconforming FE transforms, global face traversal, multi-entry "
            "rows, and variable-size hp emission remain unsupported.");
      }
   }
   else if constexpr (Format == MatrixAssemblyType::SGBSR)
   {
      if constexpr (Mode == SparseAssemblyMode::Full)
      {
         static_assert(
            Contract::sgbsr_mappings_supported,
            "SGBSR GenericAssembly requires statically one-entry, unit-weight "
            "trial and test restrictions with backend-access validation." );
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
            "supported, statically one-entry, unit-weight trial gather and "
            "test scatter mappings with backend-access validation." );
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
      has_active_trial_coefficient_dependency_v<WeakForm> ||
      !domain_supported ||
      !SparseAssemblyContextBindingsAvailable<WeakForm, Context>())
   {
      return false;
   }
   else
   {
      using TrialSpace = std::remove_cvref_t<
         decltype(
            std::declval<const Context&>()
               .template fe_field<FormTraits::trial_name>().space)>;
      using TestSpace = std::remove_cvref_t<
         decltype(
            std::declval<const Context&>()
               .template fe_field<FormTraits::test_name>().space)>;
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
      ValidateSparseLinearAssemblyCoefficientInputs<WeakForm>();
      if constexpr (!has_active_trial_coefficient_dependency_v<WeakForm>)
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
               const auto& trial_space =
                  wf_ctx.template fe_field<FormTraits::trial_name>().space;
               const auto& test_space =
                  wf_ctx.template fe_field<FormTraits::test_name>().space;
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
   SparseAssemblyMode Mode,
   class KernelPolicy,
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
      ValidateSparseLinearAssemblyCoefficientInputs<WeakForm>();
      if constexpr (!has_active_trial_coefficient_dependency_v<WeakForm>)
      {
         ValidateSparseAssemblyDomainSupport<
            MatrixAssemblyType::RawCOO,
            Mode,
            WeakForm,
            WeakFormContext>();
         if constexpr (
            !weak_form_uses_partition_integration_domain_v<
               WeakForm,
               WeakFormContext>)
         {
            ValidateWeakFormContext(weak_form, wf_ctx);
            ValidateSparseAssemblyExecutionKernelContract<KernelPolicy>();
         }
      }
   }
}

template<class WeakForm, class WeakFormContext>
auto ValidateDefaultBSRBackendSelectionInputs()
{
   using Form = std::remove_cvref_t<WeakForm>;
   using Context = std::remove_cvref_t<WeakFormContext>;
   using Traits = SparseAssemblyFormTraits<Form>;

   static_assert(
      Traits::has_trial,
      "Default BSR backend selection requires a TrialSpace in the integrand.");
   static_assert(
      Traits::has_test,
      "Default BSR backend selection requires a TestSpace in the integrand.");
   if constexpr (Traits::has_trial && Traits::has_test)
   {
      static_assert(
         Context::template has_fe_field<Traits::trial_name>(),
         "Default BSR backend selection requires the trial field in the WeakFormContext.");
      static_assert(
         Context::template has_fe_field<Traits::test_name>(),
         "Default BSR backend selection requires the test field in the WeakFormContext.");
   }

   return std::bool_constant<
      SparseAssemblyContextBindingsAvailable<Form, Context>()>{};
}

} // namespace details

} // namespace gendil

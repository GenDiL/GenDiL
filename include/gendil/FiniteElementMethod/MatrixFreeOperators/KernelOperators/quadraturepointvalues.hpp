// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/RecursiveArray/instantiatearray.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Utilities/MathHelperFunctions/min.hpp"
#include "elementdof.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"

namespace gendil {

/**
 * @brief A helper structure that provides a container type for quadrature point data.
 * 
 * @tparam IntegrationRule The integration rule associated to the quadrature data.
 * @tparam extra_dims Extra dimensions per quadrature point.
 */
template < typename IntegrationRule, Integer... extra_dims >
struct get_quad_tensor_type_t
{
   using points = typename IntegrationRule::points;
   using Orders = typename points::num_points_tensor;
   using type = typename instantiate_array< Orders, extra_dims... >::type;
};

template < typename IntegrationRule, Integer... extra_dims >
using get_quad_tensor_type = typename get_quad_tensor_type_t< IntegrationRule, extra_dims... >::type;

/**
 * @brief A helper structure to store data at quadrature points, behaves like a multi-dimension array.
 * 
 * @tparam IntegrationRule The integration rule associated with the quadrature data.
 * @tparam extra_dims Extra dimensions per quadrature point.
 */
template < typename IntegrationRule, Integer... extra_dims >
struct QuadraturePointValues
{
   using Data = get_quad_tensor_type< IntegrationRule, extra_dims... >;
   static constexpr Integer Dim = Data::Dim;
   Data data;

   template < typename... Args >
   GENDIL_HOST_DEVICE
   Real & operator()( Args... args )
   {
      return data( args... );
   }

   template < typename... Args >
   GENDIL_HOST_DEVICE
   const Real & operator()( Args... args ) const
   {
      return data( args... );
   }

   template < typename FESpace >
   GENDIL_HOST_DEVICE
   explicit operator ElementDoF< FESpace >() const
   {
      // TODO: Add static check that the basis functions have the same points as the integration rule?
      return ElementDoF< FESpace >{ data };
   }
};

/**
 * @brief Container holding interpolated data associated with a single field.
 *
 * This structure stores the pointwise representation of a field at the
 * quadrature points relevant to the current integration context.
 *
 * @tparam ValuesType    Type used to store interpolated values.
 * @tparam GradientType  Type used to store interpolated gradients.
 *
 * In the common GenDiL pattern, either member may be `Empty` when that
 * quantity is not requested by the operator mask.
 */
template<typename ValuesType, typename GradientType>
struct InterpolatedField
{
   /// Interpolated field values at quadrature points, or `Empty`.
   ValuesType values;

   /// Interpolated field gradients at quadrature points, or `Empty`.
   GradientType gradients;
};

// template < Integer Index, typename IntegrationRule, size_t ... extra_dims >
// struct GetTensorSize< Index, QuadraturePointValues< IntegrationRule, extra_dims ... > > : GetTensorSize< Index, typename QuadraturePointValues< IntegrationRule, extra_dims ... >::Data > {};

template <
   size_t ... Dims,
   typename KernelContext,
   typename IntegrationRule >
GENDIL_HOST_DEVICE
auto MakeQuadraturePointValuesContainer( const KernelContext & kernel_conf, IntegrationRule )
{
   using quad_shape = typename IntegrationRule::points::num_points_tensor;
   using rdims = typename KernelContext::template register_dimensions< IntegrationRule::space_dim >;
   using rshape = subsequence_t< quad_shape, rdims >;
   using shape = cat_t< rshape, std::index_sequence< Dims... > >;
   return MakeStaticFIFOView< Real >( shape{} );
}

template <
   size_t ... Dims,
   typename KernelContext,
   typename ... DofToQuads >
GENDIL_HOST_DEVICE
auto MakeQuadraturePointValuesContainer(
   const KernelContext & kernel_conf,
   const std::tuple<DofToQuads...> & element_quad_data )
{
   using quad_shape = std::index_sequence< DofToQuads::num_quads... >;
   static constexpr Integer Dim = sizeof...( DofToQuads );
   using rdims = typename KernelContext::template register_dimensions< Dim >;
   using rshape = subsequence_t< quad_shape, rdims >;
   using shape = cat_t< rshape, std::index_sequence< Dims... > >;
   return MakeStaticFIFOView< Real >( shape{} );
}

template <
   size_t ... Dims,
   typename KernelContext,
   typename IntegrationRule >
GENDIL_HOST_DEVICE
auto MakeSharedQuadraturePointValuesContainer( const KernelContext & kernel_conf, IntegrationRule )
{
   using quad_shape = typename IntegrationRule::points::num_points_tensor;
   using shape = cat_t< quad_shape, std::index_sequence< Dims... > >;
   constexpr size_t shared_size = Product( shape{} );
   Real * buffer = kernel_conf.SharedAllocator.allocate( shared_size );
   return MakeFixedFIFOView( buffer, shape{} );
}

// ============================================================================
// Tuple helpers for vector field quadrature storage
// ============================================================================
//
// These helpers create tuple-per-component quadrature storage for vector FEs,
// matching the tuple-of-DOFs architecture in ReadVectorDofs/WriteVectorDofs.
//
// **Current support:** Vector values (cell mass, cell grad-grad)
// **TODO (vector grad-grad):** Tuple gradient read/write for grad-grad operators
//   will need to return/accept SerialRecursiveArray<Real, NumComp, Dim>.
// **TODO (vector facets):** Interior/boundary facet operators may need tuple-aware
//   facet ApplyAddTestFunctions overloads.
//
// ============================================================================

/**
 * @brief Create tuple of scalar value containers for vector fields.
 *
 * For vector FE with NumComp components, creates:
 *   tuple<scalar_container_0, scalar_container_1, ...>
 *
 * Follows the pattern from MakeVectorDofs in readdofs.hpp.
 */
template<
   size_t NumComp,
   typename KernelContext,
   typename IntegrationRule,
   size_t... I>
GENDIL_HOST_DEVICE constexpr
auto MakeVectorValuesTupleImpl(
   const KernelContext& kernel,
   const IntegrationRule& integration_rule,
   std::index_sequence<I...>)
{
   // Create NumComp identical scalar value containers
   return std::make_tuple(
      (static_cast<void>(I),
       MakeQuadraturePointValuesContainer(kernel, integration_rule))...
   );
}

template<
   size_t NumComp,
   typename KernelContext,
   typename IntegrationRule>
GENDIL_HOST_DEVICE constexpr
auto MakeVectorValuesTuple(
   const KernelContext& kernel,
   const IntegrationRule& integration_rule)
{
   return MakeVectorValuesTupleImpl<NumComp>(
      kernel,
      integration_rule,
      std::make_index_sequence<NumComp>{});
}

/**
 * @brief Create tuple of scalar gradient containers for vector fields.
 *
 * For vector FE with NumComp components in Dim-dimensional space, creates:
 *   tuple<scalar_grad_container_0, scalar_grad_container_1, ...>
 * where each scalar gradient container has shape [quad_points..., Dim].
 */
template<
   size_t NumComp,
   size_t Dim,
   typename KernelContext,
   typename IntegrationRule,
   size_t... I>
GENDIL_HOST_DEVICE constexpr
auto MakeVectorGradientsTupleImpl(
   const KernelContext& kernel,
   const IntegrationRule& integration_rule,
   std::index_sequence<I...>)
{
   // Create NumComp identical scalar gradient containers
   return std::make_tuple(
      (static_cast<void>(I),
       MakeQuadraturePointValuesContainer<Dim>(kernel, integration_rule))...
   );
}

template<
   size_t NumComp,
   size_t Dim,
   typename KernelContext,
   typename IntegrationRule>
GENDIL_HOST_DEVICE constexpr
auto MakeVectorGradientsTuple(
   const KernelContext& kernel,
   const IntegrationRule& integration_rule)
{
   return MakeVectorGradientsTupleImpl<NumComp, Dim>(
      kernel,
      integration_rule,
      std::make_index_sequence<NumComp>{});
}

// ============================================================================

template<
   typename KernelContext,
   typename WeakFormContext,
   typename Integrand,
   typename IntegrationRule >
GENDIL_HOST_DEVICE
auto MakeQuadraturePointContainer(
   const KernelContext & kernel,
   const WeakFormContext & wf_ctx,
   const Integrand & integrand,
   const IntegrationRule & integration_rule)
{
   using I  = std::remove_cvref_t<Integrand>;
   using IR = std::remove_cvref_t<decltype(integration_rule)>;

   constexpr auto TestName = requirements<I>::test_name;
   constexpr auto TestMask = requirements<I>::test_mask;

   static_assert(TestName != StaticString("Error"),
      "MakeQuadraturePointContainer: test_name == \"Error\". Integrand must contain a TestSpace.");

   constexpr bool need_vals  = need_values(TestMask);
   constexpr bool need_grads = need_gradients(TestMask);

   static_assert(need_vals || need_grads,
      "MakeQuadraturePointContainer: neither test values nor test gradients are required.");

   // Test space comes from wf_ctx via MakeTestField<TestName>(space)
   const auto& test_fev   = wf_ctx.template fe_field<TestName>();
   const auto& test_space = test_fev.space;

   constexpr size_t NumComp = num_comp_v<decltype(test_space)>;
   constexpr size_t Dim     = static_cast<size_t>(IR::space_dim);

   if constexpr (NumComp == 1)
   {
      using ValuesC = std::conditional_t<
         need_vals,
         decltype(MakeQuadraturePointValuesContainer(kernel, integration_rule)),
         Empty>;

      using GradsC = std::conditional_t<
         need_grads,
         decltype(MakeQuadraturePointValuesContainer<Dim>(kernel, integration_rule)),
         Empty>;

      using Container = InterpolatedField<ValuesC, GradsC>;

      return Container{
         // values container (or Empty)
         [] (const KernelContext & kernel, const IR & integration_rule) -> ValuesC {
            if constexpr (need_vals)  return MakeQuadraturePointValuesContainer(kernel, integration_rule);
            else                      return Empty{};
         }(kernel, integration_rule),

         // gradients container (or Empty)
         [] (const KernelContext & kernel, const IR & integration_rule) -> GradsC {
            if constexpr (need_grads) return MakeQuadraturePointValuesContainer<Dim>(kernel, integration_rule);
            else                      return Empty{};
         }(kernel, integration_rule)
      };
   }
   else
   {
      // Vector branch - use tuple-per-component storage
      //
      // **Why tuple-per-component?**
      // Vector FE components may have different scalar shapes/orders (e.g., H1 × L2,
      // different polynomial degrees per component). Tuple storage ensures each
      // component's quadrature container has its own type, matching the tuple-of-DOFs
      // pattern in ReadVectorDofs/WriteVectorDofs.
      //
      // **Storage layout:**
      //   - Values: tuple<scalar_QP_container_0, ..., scalar_QP_container_N-1>
      //   - Gradients: tuple<scalar_grad_QP_container_0, ..., scalar_grad_QP_container_N-1>
      //
      // **Single quadrature point:**
      //   - Values: SerialRecursiveArray<Real, NumComp>
      //   - Gradients: SerialRecursiveArray<Real, NumComp, Dim> with grad(comp, dir)
      //
      // **Component-wise operations:**
      // Tuple storage allows reuse of scalar kernel operators via std::get<I>(...)
      // in InterpolateValues, ApplyAddTestFunctions, etc. Each component is processed
      // independently using existing scalar infrastructure.
      //
      // TODO (vector grad-grad): Tuple gradient Read/Write will need to handle
      //   SerialRecursiveArray<Real, NumComp, Dim> for vector grad-grad operators.
      // TODO (vector facets): Facet tuple ApplyAddTestFunctions may be needed for
      //   vector interior/boundary facet operators.

      using ValuesC = std::conditional_t<
         need_vals,
         decltype(MakeVectorValuesTuple<NumComp>(kernel, integration_rule)),
         Empty>;

      using GradsC = std::conditional_t<
         need_grads,
         decltype(MakeVectorGradientsTuple<NumComp, Dim>(kernel, integration_rule)),
         Empty>;

      using Container = InterpolatedField<ValuesC, GradsC>;

      return Container{
         // values container (or Empty)
         [&]() -> ValuesC {
            if constexpr (need_vals)  return MakeVectorValuesTuple<NumComp>(kernel, integration_rule);
            else                      return Empty{};
         }(),

         // gradients container (or Empty)
         [&]() -> GradsC {
            if constexpr (need_grads) return MakeVectorGradientsTuple<NumComp, Dim>(kernel, integration_rule);
            else                      return Empty{};
         }()
      };
   }
}

/**
 * @brief Make quadrature point container with explicit test mask
 *
 * This overload accepts an explicit OperatorMask instead of deriving it from
 * requirements<Integrand>::test_mask. This enables the pullback-based operator path
 * to allocate Du containers based on PullbackResult channel presence rather than
 * integrand requirements.
 *
 * **Use case:** GenericCellIntegrandOperatorPullback computes:
 *    auto channels = pullback(integrand, ScaleExpr{1.0});
 *    constexpr auto ChannelTestMask =
 *       (channels.contains<ValueChannel>() ? OperatorMask::Values : OperatorMask::None) +
 *       (channels.contains<GradientChannel>() ? OperatorMask::Gradients : OperatorMask::None);
 *    auto Du = MakeQuadraturePointContainerFromTestMask(
 *       kernel, wf_ctx, ChannelTestMask, TestName, integration_rule);
 *
 * This allows mixed-channel integrands like u*v + dot(grad(u), grad(v)) to allocate
 * both Du.values and Du.gradients, which the old requirements-based path rejects due
 * to mutual-exclusivity constraints.
 *
 * @tparam KernelContext Kernel execution context
 * @tparam WeakFormContext Weak-form context (provides test field via MakeTestField<TestName>)
 * @tparam IntegrationRule Integration rule type
 * @tparam TestName Test space name (StaticString)
 *
 * @param kernel Kernel execution context
 * @param wf_ctx Weak-form context
 * @param explicit_mask OperatorMask with Values/Gradients bits set explicitly
 * @param test_name Test space name (for accessing test_fev from wf_ctx)
 * @param integration_rule Integration rule
 * @return InterpolatedField with .values and/or .gradients allocated per mask
 */
template<
   StaticString TestName,
   OperatorMask ExplicitMask,
   typename KernelContext,
   typename WeakFormContext,
   typename IntegrationRule>
GENDIL_HOST_DEVICE
auto MakeQuadraturePointContainerFromTestMask(
   const KernelContext & kernel,
   const WeakFormContext & wf_ctx,
   const IntegrationRule & integration_rule)
{
   using IR = std::remove_cvref_t<decltype(integration_rule)>;

   static constexpr bool need_vals  = need_values(ExplicitMask);
   static constexpr bool need_grads = need_gradients(ExplicitMask);

   static_assert(need_vals || need_grads,
      "MakeQuadraturePointContainerFromTestMask: explicit_mask requires at least Values or Gradients.");

   // Test space comes from wf_ctx via MakeTestField<TestName>(space)
   const auto& test_fev   = wf_ctx.template fe_field<TestName>();
   const auto& test_space = test_fev.space;

   constexpr size_t NumComp = num_comp_v<decltype(test_space)>;
   constexpr size_t Dim     = static_cast<size_t>(IR::space_dim);

   if constexpr (NumComp == 1)
   {
      // Scalar test space
      using ValuesC = std::conditional_t<
         need_vals,
         decltype(MakeQuadraturePointValuesContainer(kernel, integration_rule)),
         Empty>;

      using GradsC = std::conditional_t<
         need_grads,
         decltype(MakeQuadraturePointValuesContainer<Dim>(kernel, integration_rule)),
         Empty>;

      using Container = InterpolatedField<ValuesC, GradsC>;

      ValuesC values_c;
      if constexpr (need_vals) {
         values_c = MakeQuadraturePointValuesContainer(kernel, integration_rule);
      } else {
         values_c = Empty{};
      }

      GradsC grads_c;
      if constexpr (need_grads) {
         grads_c = MakeQuadraturePointValuesContainer<Dim>(kernel, integration_rule);
      } else {
         grads_c = Empty{};
      }

      return Container{values_c, grads_c};
   }
   else
   {
      // Vector test space - use tuple-per-component storage
      using ValuesC = std::conditional_t<
         need_vals,
         decltype(MakeVectorValuesTuple<NumComp>(kernel, integration_rule)),
         Empty>;

      using GradsC = std::conditional_t<
         need_grads,
         decltype(MakeVectorGradientsTuple<NumComp, Dim>(kernel, integration_rule)),
         Empty>;

      using Container = InterpolatedField<ValuesC, GradsC>;

      ValuesC values_c;
      if constexpr (need_vals) {
         values_c = MakeVectorValuesTuple<NumComp>(kernel, integration_rule);
      } else {
         values_c = Empty{};
      }

      GradsC grads_c;
      if constexpr (need_grads) {
         grads_c = MakeVectorGradientsTuple<NumComp, Dim>(kernel, integration_rule);
      } else {
         grads_c = Empty{};
      }

      return Container{values_c, grads_c};
   }
}

template<
   typename KernelContext,
   typename WeakFormContext,
   typename Integrand,
   typename IntegrationRule >
GENDIL_HOST_DEVICE
auto MakeQuadraturePointContainerFromIntegrand(
   const KernelContext & kernel,
   const WeakFormContext & wf_ctx,
   const Integrand & integrand,
   const IntegrationRule & integration_rule)
{
   constexpr auto TestName = requirements<Integrand>::test_name;
   constexpr auto TestMask = requirements<Integrand>::test_mask;

   return MakeQuadraturePointContainerFromTestMask<TestName, TestMask>(
      kernel,
      wf_ctx,
      integration_rule);
}

}

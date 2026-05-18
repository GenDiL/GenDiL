// ApplyAddTestFunctions.hpp

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"

// kernel ops (the "low-level" ones you already use in AdvectionOperator):
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/elementcontext.hpp"

namespace gendil
{

template<
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class Cell,
   class Integrand,
   class DuType,
   class ElementDofsOut>
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(
   const KernelContext&   kernel,
   const WeakFormContext& /*wf_ctx*/,
   const OperatorContext& op_ctx,
   const ElementContext<Cell>&  /*ec*/,
   const Integrand&       /*integrand*/,
   const DuType&          Du,
   ElementDofsOut&        dofs_out)
{
   using I = std::remove_cvref_t<Integrand>;

   constexpr auto TestName = requirements<I>::test_name;
   constexpr auto TestMask = requirements<I>::test_mask;

   static_assert(TestName != StaticString("Error"),
      "ApplyAddTestFunctions: test_name == \"Error\". Integrand must contain a TestSpace.");

   constexpr bool need_vals  = need_values(TestMask);
   constexpr bool need_grads = need_gradients(TestMask);

   static_assert(need_vals || need_grads,
      "ApplyAddTestFunctions: neither test values nor test gradients are required.");

   // Test-space quad data is generated in OperatorContext from MakeTestField<TestName>(...)
   const auto& test_qd = op_ctx.template finite_element_quad_data<TestName>();

   // --- Values test: v(x_q) ---
   if constexpr (need_vals)
   {
      using ValuesT = std::remove_cvref_t<decltype(Du.values)>;
      static_assert(!std::is_same_v<ValuesT, Empty>,
         "ApplyAddTestFunctions: test values required but Du.values is Empty.");

      // Low-level: adds B^T * Du.values into dofs_out
      ApplyAddTestFunctions(kernel, test_qd, Du.values, dofs_out);
   }

   // --- Gradient test: ∇v(x_q) ---
   if constexpr (need_grads)
   {
      using GradsT = std::remove_cvref_t<decltype(Du.gradients)>;
      static_assert(!std::is_same_v<GradsT, Empty>,
         "ApplyAddTestFunctions: test gradients required but Du.gradients is Empty.");

      // Low-level pattern (same as your advection volume example):
      //   tmp_q = G^T * Du.gradients   (still in quadrature space)
      //   add B^T * tmp_q into dofs_out
      const auto tmp_q = ApplyGradientTestFunctionsAtQPoints(kernel, test_qd, Du.gradients);

      ApplyAddTestFunctions(kernel, test_qd, tmp_q, dofs_out);
   }
}

template<
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class MinusFace,
   class PlusFace,
   class Integrand,
   class DuType,
   class ElementDofsOut>
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(
   const KernelContext&   kernel,
   const WeakFormContext& /*wf_ctx*/,
   const OperatorContext& op_ctx,
   const GlobalFaceInfo<MinusFace,PlusFace>&  fc,
   const Integrand&       /*integrand*/,
   const DuType&          Du,
   ElementDofsOut&        dofs_out)
{
   using I = std::remove_cvref_t<Integrand>;

   constexpr auto TestName = requirements<I>::test_name;
   constexpr auto TestMask = requirements<I>::test_mask;

   static_assert(TestName != StaticString("Error"),
      "ApplyAddTestFunctions: test_name == \"Error\". Integrand must contain a TestSpace.");

   constexpr bool need_vals  = need_values(TestMask);
   constexpr bool need_grads = need_gradients(TestMask);

   static_assert(need_vals || need_grads,
      "ApplyAddTestFunctions: neither test values nor test gradients are required.");

   // Test-space quad data is generated in OperatorContext from MakeTestField<TestName>(...)
   const auto& test_qd = op_ctx.template finite_element_facet_quad_data<TestName>();
   // constexpr Integer local_face_index = Face::local_face_index_type::value;
   // const auto & local_face_quad_data = std::get< local_face_index >( test_qd );

   // --- Values test: v(x_q) ---
   if constexpr (need_vals)
   {
      using ValuesT = std::remove_cvref_t<decltype(Du.values)>;
      static_assert(!std::is_same_v<ValuesT, Empty>,
         "ApplyAddTestFunctions: test values required but Du.values is Empty.");

      // Low-level: adds B^T * Du.values into dofs_out
      ApplyAddTestFunctions(kernel, fc.MinusSide(), test_qd, Du.values, dofs_out);
   }

   // --- Gradient test: ∇v(x_q) ---
   if constexpr (need_grads)
   {
      // static_assert(false, "ApplyAddTestFunctions: gradient test not yet implemented for facet integrands.");
      using GradsT = std::remove_cvref_t<decltype(Du.gradients)>;
      static_assert(!std::is_same_v<GradsT, Empty>,
         "ApplyAddTestFunctions: test gradients required but Du.gradients is Empty.");

      // // Low-level pattern (same as your advection volume example):
      // //   tmp_q = G^T * Du.gradients   (still in quadrature space)
      // //   add B^T * tmp_q into dofs_out
      // const auto tmp_q = ApplyGradientTestFunctionsAtQPoints(kernel, test_qd, Du.gradients);

      // ApplyAddTestFunctions(kernel, fc.MinusSide(), test_qd, tmp_q, dofs_out);

      ApplyGradientTestFunctions<true>(kernel, fc.MinusSide(), test_qd, Du.gradients, dofs_out);
   }
}

} // namespace gendil

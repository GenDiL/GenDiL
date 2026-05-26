// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp>

#include <cmath>
#include <iostream>
#include <type_traits>

using namespace gendil;

namespace
{

struct FakeValueTensor2D
{
   Real data[2][2]{};

   Real operator()(GlobalIndex i, GlobalIndex j) const
   {
      return data[i][j];
   }
};

struct FakeGradientTensor2D
{
   Real data[2][2][2]{};

   Real operator()(GlobalIndex i, GlobalIndex j, GlobalIndex d) const
   {
      return data[i][j][d];
   }
};

struct FakeQuadPointContext2D
{
   TensorIndex<2> quad_index;
   std::array<std::array<Real, 2>, 2> inv_J_mesh;
};

bool Near(Real a, Real b, Real tol = 1e-12)
{
   return std::abs(a - b) <= tol;
}

} // namespace

template<class Req, OperatorMask ExpectedMask, NamedFieldProvenance ExpectedProvenance>
constexpr bool requirement_has()
{
   return need_values(ExpectedMask) == need_values(Req::mask) &&
          need_gradients(ExpectedMask) == need_gradients(Req::mask) &&
          has_provenance(Req::provenance, ExpectedProvenance);
}

int main()
{
   std::cout << "Testing named-field requirement traits...\n";

   {
      auto coeff = MakeCoefficient<"k", FieldValue<"w">>(
         [] (const auto&) { return 1.0; });

      using Reqs = coefficient_named_field_requirements_t<decltype(coeff)>;
      static_assert(contains_named_field_requirement_v<Reqs, "w">);

      using WReq = find_named_field_requirement_t<Reqs, "w">;
      static_assert(requirement_has<WReq, OperatorMask::Values, NamedFieldProvenance::CoefficientInput>());
      static_assert(has_side_dependent_named_field_inputs_v<decltype(coeff)>);

      std::cout << "  [PASS] coefficient FieldValue requires named field values\n";
   }

   {
      auto coeff = MakeCoefficient<"k", FieldGradient<"w">>(
         [] (const auto&) { return 1.0; });

      using Reqs = coefficient_named_field_requirements_t<decltype(coeff)>;
      static_assert(contains_named_field_requirement_v<Reqs, "w">);

      using WReq = find_named_field_requirement_t<Reqs, "w">;
      static_assert(requirement_has<WReq, OperatorMask::Gradients, NamedFieldProvenance::CoefficientInput>());
      static_assert(has_side_dependent_named_field_inputs_v<decltype(coeff)>);

      std::cout << "  [PASS] coefficient FieldGradient requires named field gradients\n";
   }

   {
      FiniteElementField<"rho"> rho;

      using ValueReqs = finite_element_expr_named_field_requirements_t<decltype(rho)>;
      static_assert(contains_named_field_requirement_v<ValueReqs, "rho">);

      using RhoValueReq = find_named_field_requirement_t<ValueReqs, "rho">;
      static_assert(requirement_has<RhoValueReq, OperatorMask::Values, NamedFieldProvenance::FiniteElementExpression>());

      auto grad_rho = grad(rho);
      using GradientReqs = finite_element_expr_named_field_requirements_t<decltype(grad_rho)>;
      static_assert(contains_named_field_requirement_v<GradientReqs, "rho">);

      using RhoGradientReq = find_named_field_requirement_t<GradientReqs, "rho">;
      static_assert(requirement_has<RhoGradientReq, OperatorMask::Gradients, NamedFieldProvenance::FiniteElementExpression>());

      std::cout << "  [PASS] FiniteElementField and grad(FiniteElementField) requirements\n";
   }

   {
      FiniteElementField<"rho"> rho;
      auto coeff = MakeCoefficient<"k", FieldGradient<"rho">>(
         [] (const auto&) { return 1.0; });

      auto expr = rho * coeff;
      using Reqs = input_named_field_requirements_t<decltype(expr)>;
      static_assert(contains_named_field_requirement_v<Reqs, "rho">);

      using RhoReq = find_named_field_requirement_t<Reqs, "rho">;
      static_assert(need_values(RhoReq::mask));
      static_assert(need_gradients(RhoReq::mask));
      static_assert(has_provenance(RhoReq::provenance, NamedFieldProvenance::FiniteElementExpression));
      static_assert(has_provenance(RhoReq::provenance, NamedFieldProvenance::CoefficientInput));

      std::cout << "  [PASS] same-name value and gradient requirements are unioned\n";
   }

   {
      auto coeff = MakeCoefficient<"k", FieldValue<"rho">, FieldGradient<"eta">>(
         [] (const auto&, const auto&) { return 1.0; });

      using Reqs = coefficient_named_field_requirements_t<decltype(coeff)>;
      static_assert(contains_named_field_requirement_v<Reqs, "rho">);
      static_assert(contains_named_field_requirement_v<Reqs, "eta">);

      using RhoReq = find_named_field_requirement_t<Reqs, "rho">;
      using EtaReq = find_named_field_requirement_t<Reqs, "eta">;
      static_assert(need_values(RhoReq::mask));
      static_assert(!need_gradients(RhoReq::mask));
      static_assert(!need_values(EtaReq::mask));
      static_assert(need_gradients(EtaReq::mask));

      std::cout << "  [PASS] multiple coefficient named fields are collected\n";
   }

   {
      TrialSpace<"u"> u;
      TestSpace<"v"> v;
      Cells<"mesh"> cells;

      auto coeff = MakeCoefficient<"k", FieldValue<"u">>(
         [] (const auto&) { return 1.0; });
      auto form = integrate(cells, coeff * u * v);
      using Form = decltype(form);

      static_assert(has_active_trial_coefficient_dependency_v<Form>);

      using ActiveReqs = active_trial_named_field_requirements_t<Form>;
      static_assert(contains_named_field_requirement_v<ActiveReqs, "u">);
      using UReq = find_named_field_requirement_t<ActiveReqs, "u">;
      static_assert(need_values(UReq::mask));
      static_assert(has_provenance(UReq::provenance, NamedFieldProvenance::ActiveTrial));

      std::cout << "  [PASS] active-trial coefficient overlap is detected\n";
   }

   {
      TrialSpace<"u"> u;
      TestSpace<"v"> v;
      Cells<"mesh"> cells;

      auto coeff = MakeCoefficient<"k", FieldGradient<"u">>(
         [] (const auto&) { return 1.0; });
      auto form = integrate(cells, coeff * u * v);

      static_assert(has_active_trial_coefficient_dependency_v<decltype(form)>);

      std::cout << "  [PASS] active-trial FieldGradient overlap is detected\n";
   }

   {
      TrialSpace<"u"> u;
      TestSpace<"v"> v;
      Cells<"mesh"> cells;

      auto coeff = MakeCoefficient<"k", FieldValue<"u_lagged">>(
         [] (const auto&) { return 1.0; });
      auto form = integrate(cells, coeff * u * v);

      static_assert(!has_active_trial_coefficient_dependency_v<decltype(form)>);

      std::cout << "  [PASS] lagged coefficient field avoids active-trial overlap\n";
   }

   {
      auto geom_coeff = MakeCoefficient<"k", PhysicalCoordinate>(
         [] (const auto&) { return 1.0; });

      static_assert(!has_side_dependent_named_field_inputs_v<decltype(geom_coeff)>);
      static_assert(has_test_space_dependency_v<TestSpace<"v">>);
      static_assert(!has_test_space_dependency_v<TrialSpace<"u">>);
      static_assert(is_side_evaluable_v<decltype(geom_coeff)>);

      std::cout << "  [PASS] side-dependent named fields are distinct from geometry-only coefficients\n";
   }

   {
      Real shared[1]{};
      KernelContext<SerialKernelConfiguration, 1> kernel(shared);

      FakeValueTensor2D values{};
      values.data[1][0] = 7.0;

      FakeGradientTensor2D gradients{};
      gradients.data[1][0][0] = 3.0;
      gradients.data[1][0][1] = 8.0;

      using Field = InterpolatedField<FakeValueTensor2D, FakeGradientTensor2D>;
      auto fields = make_map(
         Entry<NameTag<"w">, Field>{ Field{ values, gradients } }
      );

      FakeQuadPointContext2D qctx{
         TensorIndex<2>{GlobalIndex{1}, GlobalIndex{0}},
         std::array<std::array<Real, 2>, 2>{
            std::array<Real, 2>{2.0, 0.0},
            std::array<Real, 2>{0.0, 0.25}
         }
      };

      auto coeff = MakeCoefficient<"k", FieldValue<"w">, FieldGradient<"w">>(
         [] (const Real value, const auto& physical_gradient)
         {
            return value + physical_gradient[0] + physical_gradient[1];
         });

      const Real result = coeff(kernel, Empty{}, Empty{}, Empty{}, qctx, fields);

      // Raw reference gradient is [3, 8]; inv_J maps it to [6, 2].
      if (!Near(result, 15.0))
      {
         std::cerr << "FAILED: expected coefficient value 15, got " << result << "\n";
         return 1;
      }

      std::cout << "  [PASS] coefficient FieldValue/FieldGradient readers return plain physical data\n";
   }

   std::cout << "All named-field requirement trait tests passed.\n";
   return 0;
}

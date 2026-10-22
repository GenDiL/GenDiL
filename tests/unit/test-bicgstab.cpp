// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/Solvers/bicgstab.hpp>
#include <gendil/Algebra/Vector/vector.hpp>

#include <array>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string_view>

namespace
{
using namespace gendil;

bool Check(bool condition, const char* label)
{
   if (!condition) std::cerr << label << '\n';
   return condition;
}

Vector Values(std::initializer_list<Real> values)
{
   Vector result(values.size());
   auto* data = result.WriteHostData();
   size_t i = 0;
   for (const Real value : values) data[i++] = value;
   return result;
}

bool Same(const Vector& left, const Vector& right)
{
   if (left.Size() != right.Size()) return false;
   const auto* a = left.ReadHostData();
   const auto* b = right.ReadHostData();
   for (size_t i = 0; i < left.Size(); ++i)
      if (!std::isfinite(a[i]) || a[i] != b[i]) return false;
   return true;
}

Real HostDot(const Vector& left, const Vector& right)
{
   Real result = 0;
   const auto* a = left.ReadHostData();
   const auto* b = right.ReadHostData();
   for (size_t i = 0; i < left.Size(); ++i) result += a[i]*b[i];
   return result;
}

struct Matrix
{
   size_t size;
   std::array<Real, 9> entries;
   mutable Integer calls = 0;

   void operator()(const Vector& input, Vector& output) const
   {
      ++calls;
      const auto* x = input.ReadHostData();
      auto* y = output.WriteHostData();
      for (size_t i = 0; i < size; ++i)
      {
         y[i] = 0;
         for (size_t j = 0; j < size; ++j) y[i] += entries[size*i+j]*x[j];
      }
   }
};

struct Workspace
{
   Vector r, r_hat, p, v, s, t;
   explicit Workspace(size_t size)
      : r(size), r_hat(size), p(size), v(size), s(size), t(size)
   {
      r = 13; r_hat = -17; p = 19; v = -23; s = 29; t = -31;
   }
};

IterativeSolverResult Solve(const Matrix& op, const Vector& rhs, Vector& x,
                           Integer budget, Real tolerance)
{
   Workspace work(x.Size());
   return BiCGSTAB(op, rhs, HostDot, budget, tolerance, x,
      work.r, work.r_hat, work.p, work.v, work.s, work.t);
}

IterativeSolverResult SolveRestarted(const Matrix& op, const Vector& rhs, Vector& x,
                                    Integer budget, Real tolerance, Integer restarts)
{
   Workspace work(x.Size());
   return RestartedBiCGSTAB_no_alloc(op, rhs, HostDot, budget, tolerance, x,
      work.r, work.r_hat, work.p, work.v, work.s, work.t, restarts);
}

// Independent matrix evaluation and hypot-based norm do not use the solver's
// reduction callback, recurrence residual, or instrumented operator call count.
Real TrueResidual(const Matrix& op, const Vector& rhs, const Vector& x)
{
   const auto* b = rhs.ReadHostData();
   const auto* solution = x.ReadHostData();
   Real residual = 0, rhs_norm = 0;
   for (size_t i = 0; i < op.size; ++i)
   {
      Real applied = 0;
      for (size_t j = 0; j < op.size; ++j)
         applied += op.entries[op.size*i+j]*solution[j];
      residual = std::hypot(residual, b[i]-applied);
      rhs_norm = std::hypot(rhs_norm, b[i]);
   }
   return rhs_norm == 0 ? residual : residual/rhs_norm;
}

bool CheckNumerics(const IterativeSolverResult& result, const Matrix& op,
                   const Vector& rhs, const Vector& saved_rhs, const Vector& x,
                   Real tolerance, const char* label)
{
   const Real actual = TrueResidual(op, rhs, x);
   constexpr Real slack = 64*std::numeric_limits<Real>::epsilon();
   bool ok = Check(Same(rhs, saved_rhs), "BiCGSTAB changed its RHS");
   ok = Check(std::isfinite(actual) && std::isfinite(result.relative_error) &&
      result.relative_error >= 0 &&
      std::abs(result.relative_error-actual) <= slack*actual, label) && ok;
   if (result.success)
      ok = Check(actual <= tolerance*(1+slack), label) && ok;
   return ok;
}

bool CheckBasicCases()
{
   bool ok = true;
   const Matrix identity{2, {1, 0, 0, 1}};
   const auto run = [&](const Matrix& op, const Vector& rhs, Vector& x,
                        Integer budget, Real tolerance, bool success,
                        Integer iterations, bool preserve, const char* label)
   {
      const Vector saved_rhs(rhs), saved_x(x);
      const auto result = Solve(op, rhs, x, budget, tolerance);
      bool passed = Check(result.success == success && result.iterations == iterations, label);
      passed = CheckNumerics(result, op, rhs, saved_rhs, x, tolerance, label) && passed;
      if (preserve) passed = Check(Same(x, saved_x), label) && passed;
      return passed;
   };

   const Vector rhs = Values({1, -2});
   Vector x(2); x = 0;
   ok = run(identity, rhs, x, 10, 1e-12, true, 1, false, "identity from zero") && ok;
   x = Values({-.5, .25});
   ok = run(identity, rhs, x, 10, 0, true, 1, false, "identity nonzero guess, exact tolerance") && ok;
   x = rhs;
   ok = run(identity, rhs, x, 0, 0, true, 0, true, "exact initial guess preserved") && ok;
   x = Values({1.125, -2.25});
   ok = run(identity, rhs, x, 0, .125, true, 0, true, "relative tolerance equality") && ok;
   x = Values({1.0625, -2.125});
   ok = run(identity, rhs, x, 10, .125, true, 0, true, "already within tolerance") && ok;
   x = Values({.25, -.5});
   ok = run(identity, rhs, x, 0, 1e-12, false, 0, true, "zero budget preserves nonconverged guess") && ok;

   Vector zero(2); zero = 0; x = 0;
   ok = run(identity, zero, x, 0, 0, true, 0, true, "zero RHS and zero guess") && ok;
   x = Values({1, -2});
   ok = run(identity, zero, x, 0, 1e-12, false, 0, true, "zero RHS, nonzero guess, zero budget") && ok;
   ok = run(identity, zero, x, 10, 0, true, 1, false, "zero RHS solves nonzero guess") && ok;
   x = Values({.125, 0});
   ok = run(identity, zero, x, 0, .125, true, 0, true, "absolute tolerance equality") && ok;
   const Vector tiny = Values({1e-31, 2e-31}); x = 0;
   ok = run(identity, tiny, x, 10, 0, true, 1, false, "tiny nonzero RHS") && ok;

   const Matrix nonsymmetric{2, {4, 1, 2, 3}};
   const Vector nonsymmetric_rhs = Values({1, 2}); x = 0;
   ok = run(nonsymmetric, nonsymmetric_rhs, x, 10, 1e-12, true, 2, false, "nonsymmetric solve") && ok;
   const auto* solution = x.ReadHostData();
   ok = Check(std::abs(solution[0]-.1) < 1e-14 && std::abs(solution[1]-.6) < 1e-14,
      "nonsymmetric analytic solution") && ok;
   x = 0;
   ok = run(nonsymmetric, nonsymmetric_rhs, x, 1, 1e-12, false, 1, false, "limited iteration budget") && ok;

   const Matrix negative_identity{2, {-1, 0, 0, -1}}; x = 0;
   ok = run(negative_identity, rhs, x, 10, 0, true, 1, false, "negative alpha is valid") && ok;
   const Matrix negative_diagonal{2, {-1, 0, 0, -2}};
   const Vector ones = Values({1, 1}); x = 0;
   ok = run(negative_diagonal, ones, x, 10, 1e-12, true, 2, false, "negative omega is valid") && ok;
   for (const Real scale : {Real(1e-40), Real(1e40)})
   {
      const Matrix scaled{2, {4*scale, scale, 2*scale, 3*scale}}; x = 0;
      ok = run(scaled, nonsymmetric_rhs, x, 10, 1e-12, true, 2, false,
         "operator scaling must not trigger an absolute breakdown cutoff") && ok;
      const Vector scaled_rhs = Values({scale, 2*scale}); x = 0;
      ok = run(nonsymmetric, scaled_rhs, x, 10, 1e-12, true, 2, false,
         "RHS scaling must not trigger an absolute breakdown cutoff") && ok;
   }
   return ok;
}

bool CheckRhoRecovery()
{
   const Matrix base{3, {1, 0, 0, 2e-10, 1, 0, 0, 1, 1}};
   const Vector rhs = Values({1, 0, 0}), saved_rhs(rhs);
   Vector baseline(3);
   IterativeSolverResult baseline_result{};
   Integer baseline_calls = 0;
   bool ok = true;
   for (Integer scenario = 0; scenario < 6; ++scenario)
   {
      Matrix op = base;
      Vector x(3); x = 0;
      const Integer budget = scenario == 4 ? 1 : scenario == 5 ? 0 : 2000;
      const auto result = scenario == 0 ? Solve(op, rhs, x, budget, 1e-10) :
         SolveRestarted(op, rhs, x, budget, 1e-10, scenario == 1 ? 0 : scenario == 2 ? 1 : 10);
      const bool success = scenario == 2 || scenario == 3;
      const Integer iterations = scenario == 5 ? 0 : success ? 2 : 1;
      ok = Check(result.success == success && result.iterations == iterations,
         "rho breakdown direct/restarted cap and budget behavior") && ok;
      ok = CheckNumerics(result, op, rhs, saved_rhs, x, 1e-10, "rho recovery residual") && ok;
      if (scenario == 0 || scenario == 1 || scenario == 4)
         ok = Check(std::abs(result.relative_error-std::sqrt(Real(2))*1e-10) < 1e-24,
            "unrecovered rho residual") && ok;
      if (scenario == 0)
      {
         baseline = x; baseline_result = result; baseline_calls = op.calls;
      }
      if (scenario == 1)
         ok = Check(result.success == baseline_result.success &&
            result.iterations == baseline_result.iterations &&
            result.relative_error == baseline_result.relative_error && Same(x, baseline) &&
            op.calls == baseline_calls, "Direct BiCGSTAB must match restarted cap zero") && ok;
      if (scenario == 5)
         ok = Check(Same(x, Values({0, 0, 0})), "rho fixture zero budget changed x") && ok;
   }
   return ok;
}

bool CheckFiniteBreakdowns()
{
   // These exact zeros exercise different failure stages. The latter two
   // retain an alpha step; restarting then encounters an alpha denominator zero.
   const std::array<Matrix, 3> matrices{{
      {2, {0, -1, 1, 0}},
      {2, {1, 0, 1, 0}},
      {2, {1, 1, 1, 0}}
   }};
   const Vector rhs = Values({1, 0}), saved_rhs(rhs);
   bool ok = true;
   for (size_t stage = 0; stage < matrices.size(); ++stage)
   {
      for (const Integer cap : {Integer(0), Integer(2), Integer(10)})
      {
         Matrix op = matrices[stage];
         Vector x(2); x = 0;
         const auto result = SolveRestarted(op, rhs, x, 100, 1e-12, cap);
         ok = Check(!result.success && result.iterations == (stage == 0 ? 0 : 1),
            "finite breakdown update count") && ok;
         ok = Check(Same(x, stage == 0 ? Values({0, 0}) : Values({1, 0})),
            "finite breakdown must preserve exactly one valid alpha step") && ok;
         // Each fresh attempt evaluates its initial residual, applies the
         // search direction, and verifies its terminal residual. The retained
         // alpha step adds one application in the initial attempt.
         ok = Check(op.calls == 3*cap+(stage == 0 ? 3 : 4),
            "breakdown recovery must use exactly the bounded fresh attempts") && ok;
         ok = CheckNumerics(result, op, rhs, saved_rhs, x, 1e-12,
            "finite breakdown true residual") && ok;
         if (cap == 0)
         {
            Matrix direct = matrices[stage];
            Vector y(2); y = 0;
            const auto direct_result = Solve(direct, rhs, y, 100, 1e-12);
            ok = Check(direct_result.success == result.success &&
               direct_result.iterations == result.iterations &&
               direct_result.relative_error == result.relative_error && Same(x, y) &&
               direct.calls == op.calls, "Finite breakdown direct/cap-zero parity") && ok;
         }
      }
      if (stage != 0)
      {
         Matrix op = matrices[stage];
         Vector x(2); x = 0;
         const auto result = SolveRestarted(op, rhs, x, 1, 1e-12, 10);
         ok = Check(!result.success && result.iterations == 1 && op.calls <= 4,
            "alpha-only step exhausting budget must not restart") && ok;
         ok = CheckNumerics(result, op, rhs, saved_rhs, x, 1e-12,
            "alpha-only final residual") && ok;
      }
   }

   return ok;
}

struct FalseSNorm
{
   const Vector* s;
   Integer false_norms;
   mutable Integer injected = 0;

   Real operator()(const Vector& a, const Vector& b) const
   {
      if (&a == s && &b == s && injected < false_norms)
      {
         ++injected;
         return 0;
      }
      return HostDot(a, b);
   }
};

bool CheckRejectedVerification()
{
   const Matrix op{2, {1, 0, 0, 2}};
   const Vector rhs = Values({1, 1}), saved_rhs(rhs);
   bool ok = true;
   for (Integer scenario = 0; scenario < 4; ++scenario)
   {
      Vector x(2); x = 0;
      Workspace work(2);
      const FalseSNorm dot{&work.s, scenario == 2 ? Integer(2) : Integer(1)};
      const auto result = RestartedBiCGSTAB_no_alloc(op, rhs, dot,
         scenario == 3 ? Integer(2) : Integer(10), Real(1e-12), x,
         work.r, work.r_hat, work.p, work.v, work.s, work.t,
         scenario == 0 ? Integer(0) : Integer(1));
      ok = Check(dot.injected == dot.false_norms, "verification fault was not exercised") && ok;
      if (scenario == 1)
         ok = Check(result.success && result.iterations > 1 && result.iterations <= 4,
            "failed verification must recover within the remaining update budget") && ok;
      else if (scenario == 3)
         ok = Check(!result.success && result.iterations == 2,
            "recovered attempt must receive only the remaining total iteration budget") && ok;
      else
         ok = Check(!result.success && result.iterations == (scenario == 0 ? 1 : 2),
            "failed verification must consume the global restart allowance") && ok;
      ok = CheckNumerics(result, op, rhs, saved_rhs, x, 1e-12,
         "rejected verification must report the true residual") && ok;
   }
   return ok;
}
} // namespace

int main(int argc, char** argv)
{
   if (argc == 2)
   {
      std::string_view mode(argv[1]);
      const bool single_attempt = mode.starts_with("single-");
      if (single_attempt) mode.remove_prefix(7);
      const Real tolerance = mode == "negative-tolerance" ? -1 :
         mode == "nan-tolerance" ? std::numeric_limits<Real>::quiet_NaN() :
         mode == "infinite-tolerance" ? std::numeric_limits<Real>::infinity() : 0;
      if (mode != "negative-tolerance" && mode != "nan-tolerance" && mode != "infinite-tolerance")
         return 2;
      const Matrix op{2, {1, 0, 0, 1}};
      const Vector rhs = Values({1, 2});
      Vector x(2); x = 0;
      if (single_attempt) (void)Solve(op, rhs, x, 10, tolerance);
      else (void)SolveRestarted(op, rhs, x, 10, tolerance, 1);
      std::cerr << "BiCGSTAB accepted an invalid tolerance\n";
      return 1;
   }
   bool ok = CheckBasicCases();
   ok = CheckRhoRecovery() && ok;
   ok = CheckFiniteBreakdowns() && ok;
   ok = CheckRejectedVerification() && ok;
   return ok ? 0 : 1;
}

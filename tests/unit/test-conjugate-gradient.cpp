// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>
#include <array>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>

namespace
{
using namespace gendil;

bool Check(bool condition, const char* message)
{
   if (!condition) std::cerr << message << '\n';
   return condition;
}

Vector Values(std::initializer_list<Real> values)
{
   Vector result(values.size());
   auto* data = result.WriteHostData();
   size_t i = 0;
   for (Real value : values) data[i++] = value;
   return result;
}

bool Same(const Vector& a, const Vector& b)
{
   if (a.Size() != b.Size()) return false;
   const auto* left = a.ReadHostData();
   const auto* right = b.ReadHostData();
   for (size_t i = 0; i < a.Size(); ++i)
      if (!std::isfinite(left[i]) || left[i] != right[i]) return false;
   return true;
}

Real HostDot(const Vector& a, const Vector& b)
{
   Real result = 0;
   const auto* left = a.ReadHostData();
   const auto* right = b.ReadHostData();
   for (size_t i = 0; i < a.Size(); ++i) result += left[i]*right[i];
   return result;
}

struct WorkspaceDot
{
   Real operator()(const Vector& a, const Vector& b) const { return HostDot(a, b); }
   template<class A, class B> Real operator()(const A&, const B&) const = delete;
};

struct Identity
{
   void operator()(const Vector& input, Vector& output) const { output = input; }
};

struct Diagonal
{
   void operator()(const Vector& input, Vector& output) const
   {
      constexpr std::array<Real, 3> diagonal{1, 2, 4};
      const auto* x = input.ReadHostData();
      auto* y = output.WriteHostData();
      for (size_t i = 0; i < diagonal.size(); ++i) y[i] = diagonal[i]*x[i];
   }
};

template<bool Scratch, class Operator>
auto Solve(const Operator& op, const Vector& rhs, Integer budget, Real tolerance, Vector& x)
{
   if constexpr (Scratch)
   {
      Vector tmp(x.Size()), z(x.Size()), residual(x.Size()), direction(x.Size());
      tmp = 71; z = -23; residual = 37; direction = -13;
      return ConjugateGradient(op, rhs, WorkspaceDot{}, budget, tolerance,
         x, tmp, z, residual, direction);
   }
   else return ConjugateGradient(op, rhs, budget, tolerance, x);
}

// Reapply A and use a separate hypot-based norm, without solver workspaces or
// its injected dot callback. For a zero RHS the reported metric is absolute.
template<class Operator>
Real TrueResidual(const Operator& op, const Vector& rhs, const Vector& x)
{
   Vector applied(rhs.Size());
   op(x, applied);
   Real residual_norm = 0, rhs_norm = 0;
   const auto* b = rhs.ReadHostData();
   const auto* ax = applied.ReadHostData();
   for (size_t i = 0; i < rhs.Size(); ++i)
   {
      residual_norm = std::hypot(residual_norm, b[i]-ax[i]);
      rhs_norm = std::hypot(rhs_norm, b[i]);
   }
   return rhs_norm == 0 ? residual_norm : residual_norm/rhs_norm;
}

template<class Operator>
bool CheckMetric(const IterativeSolverResult& result, const Operator& op,
                 const Vector& rhs, const Vector& x, const char* label)
{
   const Real actual = TrueResidual(op, rhs, x);
   return Check(std::isfinite(result.relative_error) && std::isfinite(actual)
      && std::abs(result.relative_error-actual) <= 8e-15+2e-12*actual, label);
}

template<bool Scratch>
bool CheckNativeCases()
{
   bool ok = true;
   const auto run = [&](const auto& op, const Vector& rhs, Vector& x, Integer budget,
                        Real tolerance, bool success, Integer iterations,
                        bool preserve, const char* label)
   {
      const Vector saved_rhs(rhs), saved_x(x);
      const auto result = Solve<Scratch>(op, rhs, budget, tolerance, x);
      bool passed = Check(result.success == success && result.iterations == iterations, label);
      passed = CheckMetric(result, op, rhs, x, label) && passed;
      passed = Check(Same(rhs, saved_rhs), "CG changed its RHS") && passed;
      if (preserve) passed = Check(Same(x, saved_x), label) && passed;
      if (result.success)
         passed = Check(TrueResidual(op, rhs, x) <= tolerance*(1+2e-12), label) && passed;
      return passed;
   };

   const auto rhs = Values({1, -2, 4});
   Vector x(3); x = 0;
   ok = run(Identity{}, rhs, x, 10, 1e-12, true, 1, false, "identity from zero") && ok;
   x = Values({-.5, .25, .75});
   ok = run(Identity{}, rhs, x, 10, 1e-12, true, 1, false, "identity from nonzero guess") && ok;
   x = rhs;
   ok = run(Identity{}, rhs, x, 10, 1e-12, true, 0, true, "exact initial guess preserved") && ok;
   x = rhs; x *= 1+4e-7;
   ok = run(Identity{}, rhs, x, 0, 1e-6, true, 0, true, "initial relative tolerance uses RHS norm") && ok;

   const auto diagonal_rhs = Values({1, -2, 8});
   x = 0;
   ok = run(Diagonal{}, diagonal_rhs, x, 10, 1e-12, true, 3, false, "diagonal SPD from zero") && ok;
   x = Values({-.5, .5, .25});
   ok = run(Diagonal{}, diagonal_rhs, x, 10, 1e-12, true, 3, false, "diagonal SPD nonzero guess") && ok;
   x = 0;
   ok = run(Diagonal{}, diagonal_rhs, x, 1, 1e-12, false, 1, false, "limited solve reports true relative residual") && ok;
   x = Values({.25, -.5, 1});
   ok = run(Diagonal{}, diagonal_rhs, x, 0, 1e-12, false, 0, true, "zero iteration budget preserves guess") && ok;

   Vector zero(3); zero = 0;
   x = Values({1, -2, 4});
   ok = run(Identity{}, zero, x, 0, 1e-12, false, 0, true, "zero RHS with zero budget uses absolute residual") && ok;
   ok = run(Identity{}, zero, x, 10, 1e-12, true, 1, false, "zero RHS solves nonzero guess") && ok;
   ok = run(Identity{}, zero, x, 10, 0, true, 0, true, "zero RHS exact initial guess") && ok;
   x = Values({1e-9, -2e-9, 3e-9});
   ok = run(Identity{}, zero, x, 10, 1e-8, true, 0, true, "zero RHS absolute tolerance preserves small guess") && ok;
   const auto tiny = Values({1e-100, -2e-100, 4e-100});
   x = 0;
   ok = run(Identity{}, tiny, x, 10, 1e-12, true, 1, false, "tiny nonzero RHS must not be discarded") && ok;
   const auto reported_tiny = Values({1e-31, 2e-31});
   Vector tiny_guess(2); tiny_guess = 0;
   ok = run(Identity{}, reported_tiny, tiny_guess, 10, 1e-12, true, 1, false,
      "reported tiny RHS below the former cutoff") && ok;
   x = 0;
   ok = run(Identity{}, rhs, x, 1, 0, true, 1, false, "zero tolerance accepts exact one-step solution") && ok;
   if (!ok) std::cerr << (Scratch ? "explicit scratch" : "short") << " CG cases failed\n";
   return ok;
}

template<bool Scratch>
bool CheckGenericMass()
{
   CartesianMesh<2> mesh({2, 1}, {.4, .7}, {.15, -.2}, false);
   const auto space = MakeFiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{}));
   const auto context = MakeWeakFormContext(MakeTrialField<"u">(space), MakeTestField<"v">(space),
      MakeIntegrationDomain<"mesh">(mesh));
   const auto mass = MakeGenericOperator<SerialKernelConfiguration>(
      integrate(Cells<"mesh">{}, TrialSpace<"u">{}*TestSpace<"v">{}), context,
      MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{}));
   Vector exact(space.GetNumberOfFiniteElementDofs()), rhs(exact.Size()), x(exact.Size());
   auto* exact_values = exact.WriteHostData();
   for (size_t i = 0; i < exact.Size(); ++i) exact_values[i] = .25+.03*i-.008*i*i;
   mass(exact, rhs);
   const Vector saved_rhs(rhs);
   x = .2;
   const auto result = Solve<Scratch>(mass, rhs, 40, 1e-11, x);
   bool ok = Check(result.success && result.iterations > 0 && result.iterations <= 40,
      "generic FE mass CG convergence");
   ok = CheckMetric(result, mass, rhs, x, "generic FE mass true residual metadata") && ok;
   ok = Check(TrueResidual(mass, rhs, x) <= 1e-11, "generic FE mass residual tolerance") && ok;
   ok = Check(Same(rhs, saved_rhs), "generic FE mass RHS unchanged") && ok;
   for (size_t i = 0; i < exact.Size(); ++i)
      ok = Check(std::isfinite(x[i]) && std::abs(x[i]-exact[i]) < 2e-9,
         "generic FE mass solution") && ok;
   return ok;
}

struct AlternateRhs
{
   const Vector& values;
   operator const Vector&() const { return values; }
};
static_assert(std::is_assignable_v<Vector&, const AlternateRhs&>);
static_assert(!std::is_invocable_v<WorkspaceDot, const AlternateRhs&, const AlternateRhs&>);

bool CheckWorkspaceOnlyDot()
{
   const auto rhs = Values({1, -2, 4});
   const AlternateRhs alternate{rhs};
   Vector x(3), tmp(3), z(3), residual(3), direction(3);
   x = 0;
   const auto result = ConjugateGradient(Identity{}, alternate, WorkspaceDot{}, 5, 1e-12,
      x, tmp, z, residual, direction);
   return Check(result.success && result.iterations == 1 && Same(x, rhs),
      "dot callback receives workspace vectors even for alternate RHS");
}

enum class Fault { RhsNorm, InitialNorm, Curvature, Alpha, Beta, RecurrenceNorm, FalseConvergence };

struct ScriptedDot
{
   Fault fault;
   Real bad = 0;
   mutable size_t norms = 0, curvatures = 0;
   Real operator()(const Vector& a, const Vector& b) const
   {
      if (&a == &b)
      {
         const size_t index = norms++;
         if ((fault == Fault::RhsNorm && index == 0)
             || (fault == Fault::InitialNorm && index == 1)
             || (fault == Fault::RecurrenceNorm && index == 2)) return bad;
         if (fault == Fault::FalseConvergence && index == 2) return 0;
         if (fault == Fault::Beta && index == 1) return 1e-300;
         if (fault == Fault::Beta && index == 2) return 1e308;
      }
      else if (curvatures++ == 0)
      {
         if (fault == Fault::Curvature) return bad;
         if (fault == Fault::Alpha) return std::numeric_limits<Real>::denorm_min();
      }
      return HostDot(a, b);
   }
};

bool CheckBreakdownsAndRestart()
{
   const auto rhs = Values({1, -2, 8});
   const Vector saved_rhs(rhs);
   bool ok = true;
   const auto run = [&](Fault fault, Real bad, Integer expected_iterations)
   {
      Vector x(3), tmp(3), z(3), residual(3), direction(3); x = 0;
      const ScriptedDot dot{fault, bad};
      size_t operator_calls = 0;
      const auto op = [&](const Vector& input, Vector& output) {
         ++operator_calls; Diagonal{}(input, output);
      };
      const auto result = ConjugateGradient(op, rhs, dot, 10, 0,
         x, tmp, z, residual, direction);
      bool passed = Check(!result.success && result.iterations == expected_iterations,
         "invalid CG scalar must report failure with completed-step count");
      passed = Check(Same(rhs, saved_rhs), "CG breakdown changed RHS") && passed;
      if (fault == Fault::RhsNorm)
         passed = Check(std::isinf(result.relative_error) && operator_calls == 0,
            "invalid RHS norm rejected before operator evaluation") && passed;
      else if (fault == Fault::InitialNorm)
         passed = Check(std::isinf(result.relative_error), "invalid true residual norm reports infinity") && passed;
      else passed = CheckMetric(result, Diagonal{}, rhs, x, "breakdown returns true residual metadata") && passed;
      return passed;
   };
   for (Real bad : {-1., std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::infinity()})
   {
      ok = run(Fault::RhsNorm, bad, 0) && ok;
      ok = run(Fault::InitialNorm, bad, 0) && ok;
      ok = run(Fault::RecurrenceNorm, bad, 1) && ok;
      ok = run(Fault::Curvature, bad, 0) && ok;
   }
   ok = run(Fault::Curvature, 0, 0) && ok;
   ok = run(Fault::Alpha, 0, 0) && ok;
   ok = run(Fault::Beta, 0, 1) && ok;

   // One dishonest recurrence norm simulates loss of residual accuracy. The
   // next true-residual measurement is exact, forcing a restart from b-Ax.
   Vector x(3), tmp(3), z(3), residual(3), direction(3); x = 0;
   const ScriptedDot dot{Fault::FalseConvergence};
   size_t true_residual_evaluations = 0;
   const auto op = [&](const Vector& input, Vector& output) {
      if (&input == &x) ++true_residual_evaluations;
      Diagonal{}(input, output);
   };
   const auto result = ConjugateGradient(op, rhs, dot, 10, 1e-12,
      x, tmp, z, residual, direction);
   ok = Check(result.success && result.iterations > 1 && result.iterations <= 10
      && dot.curvatures > 1 && true_residual_evaluations >= 3,
      "false recurrence convergence must verify and restart") && ok;
   ok = CheckMetric(result, Diagonal{}, rhs, x, "restarted CG true residual metadata") && ok;
   ok = Check(TrueResidual(Diagonal{}, rhs, x) <= 1e-12 && Same(rhs, saved_rhs),
      "restarted CG meets actual tolerance and preserves RHS") && ok;

   // Verification consumes no iteration, but must not reset the step budget.
   x = 0;
   const ScriptedDot limited_dot{Fault::FalseConvergence};
   const auto limited = ConjugateGradient(Diagonal{}, rhs, limited_dot, 1, 1e-12,
      x, tmp, z, residual, direction);
   ok = Check(!limited.success && limited.iterations == 1 && limited_dot.curvatures == 1,
      "rejected convergence at the budget must fail without another step") && ok;
   ok = CheckMetric(limited, Diagonal{}, rhs, x, "limited verification returns true residual") && ok;
   return Check(TrueResidual(Diagonal{}, rhs, x) > 1e-12 && Same(rhs, saved_rhs),
      "limited verification preserves the actual unconverged result") && ok;
}

bool CheckEmptyShortSolve()
{
   Vector rhs, x;
   size_t calls = 0;
   const auto op = [&](const Vector&, Vector&) { ++calls; };
   const auto result = ConjugateGradient(op, rhs, 0, 0., x);
   return Check(result.success && result.iterations == 0 && result.relative_error == 0
      && calls == 0 && x.Size() == 0, "empty short CG must not invoke its operator");
}

int Rejection(std::string_view mode)
{
   Vector rhs(2), x(mode == "size-mismatch" ? 1 : 2); rhs = 1; x = 0;
   Real tolerance = 1e-12;
   if (mode == "negative-tolerance") tolerance = -1;
   else if (mode == "nan-tolerance") tolerance = std::numeric_limits<Real>::quiet_NaN();
   else if (mode == "infinite-tolerance") tolerance = std::numeric_limits<Real>::infinity();
   else if (mode != "size-mismatch") return 2;
   ConjugateGradient(Identity{}, rhs, 5, tolerance, x);
   return 0;
}
} // namespace

int main(int argc, char** argv)
{
   if (argc > 1) return Rejection(argv[1]);
   bool ok = CheckNativeCases<false>();
   ok = CheckNativeCases<true>() && ok;
   ok = CheckGenericMass<false>() && ok;
   ok = CheckGenericMass<true>() && ok;
   ok = CheckWorkspaceOnlyDot() && ok;
   ok = CheckBreakdownsAndRestart() && ok;
   return CheckEmptyShortSolve() && ok ? 0 : 1;
}

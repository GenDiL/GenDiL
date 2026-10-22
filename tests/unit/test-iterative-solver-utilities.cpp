// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/Solvers/iterativesolverutilities.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

namespace
{
using namespace gendil;
using namespace gendil::solver_detail;

bool Check(bool condition, const char* message)
{
   if (!condition) std::cerr << message << '\n';
   return condition;
}

bool CheckNormsAndConvergence()
{
   constexpr Real tiny = std::numeric_limits<Real>::denorm_min();
   constexpr Real large = std::numeric_limits<Real>::max();
   bool ok = true;
   for (Real valid : {Real(0), Real(-0.), tiny, Real(1), large})
      ok = Check(ValidNorm(valid), "finite nonnegative norm rejected") && ok;
   for (Real invalid : {-tiny, Real(-1), std::numeric_limits<Real>::quiet_NaN(),
                       std::numeric_limits<Real>::infinity(), -std::numeric_limits<Real>::infinity()})
   {
      ok = Check(!ValidNorm(invalid), "invalid norm admitted") && ok;
      ok = Check(!Converged(invalid, 1, 1, 1) && !Converged(0, invalid, 1, 1),
         "invalid norms must prevent convergence") && ok;
   }
   ok = Check(Converged(0, 0, 0, 0) && Converged(0, 1, 0, 0),
      "zero residual satisfies zero tolerances") && ok;
   ok = Check(Converged(.25, 4, .0625, 0) && !Converged(.5, 4, .0625, 0),
      "relative tolerance uses the RHS norm and includes equality") && ok;
   ok = Check(Converged(.25, 0, 0, .25) && !Converged(.5, 0, 100, .25),
      "zero RHS uses only the absolute tolerance") && ok;
   ok = Check(Converged(.25, 100, 0, .3) && !Converged(.25, 100, 0, .2),
      "absolute tolerance works independently of relative tolerance") && ok;
   ok = Check(!Converged(tiny, large, 0, 0),
      "underflowed relative quotient must not satisfy zero tolerances") && ok;
   ok = Check(Converged(tiny, large, std::numeric_limits<Real>::epsilon(), 0),
      "positive relative tolerance admits a sufficiently small residual") && ok;
   return Check(Converged(large, large, 2, 0),
      "relative convergence avoids a tolerance-product overflow") && ok;
}

bool CheckResults()
{
   bool ok = true;
   for (bool success : {false, true})
   {
      const auto relative = MakeIterativeSolverResult(success, 7, .25, 2);
      const auto absolute = MakeIterativeSolverResult(success, 7, .25, 0);
      const auto overflow = MakeIterativeSolverResult(success, 7,
         std::numeric_limits<Real>::max(), std::numeric_limits<Real>::denorm_min());
      ok = Check(relative.success == success && relative.iterations == 7
         && relative.relative_error == .125, "relative result metadata") && ok;
      ok = Check(absolute.success == success && absolute.iterations == 7
         && absolute.relative_error == .25, "zero RHS result reports absolute error") && ok;
      ok = Check(overflow.success == success && overflow.iterations == 7
         && std::isinf(overflow.relative_error) && overflow.relative_error > 0,
         "quotient overflow preserves the supplied success flag") && ok;
      for (Real invalid : {Real(-1), std::numeric_limits<Real>::quiet_NaN(),
                           std::numeric_limits<Real>::infinity()})
      {
         const auto invalid_residual = MakeIterativeSolverResult(success, 9, invalid, 1);
         const auto invalid_rhs = MakeIterativeSolverResult(success, 9, 0, invalid);
         for (const auto& result : {invalid_residual, invalid_rhs})
            ok = Check(!result.success && result.iterations == 9
               && std::isinf(result.relative_error) && result.relative_error > 0,
               "invalid norm forces failure with infinite error") && ok;
      }
   }
   const auto exact = MakeIterativeSolverResult(true, 0, 0, 0);
   return Check(exact.success && exact.iterations == 0 && exact.relative_error == 0,
      "zero residual result") && ok;
}

// Passive buffers deliberately have no vector arithmetic or reduction API.
struct Buffer { std::array<Real, 3> values; };
struct CallLog
{
   std::array<char, 3> order{};
   size_t count = 0;
   void Record(char operation) { if (count < order.size()) order[count] = operation; ++count; }
};

struct Apply
{
   CallLog& log;
   explicit Apply(CallLog& state) : log(state) {}
   Apply(const Apply&) = delete;
   Apply(Apply&&) = delete;
   void operator()(const Buffer& x, Buffer& work)
   {
      log.Record('A');
      for (size_t i = 0; i < 3; ++i) work.values[i] = (i+2)*x.values[i];
   }
};

struct Copy
{
   CallLog& log;
   explicit Copy(CallLog& state) : log(state) {}
   Copy(const Copy&) = delete;
   Copy(Copy&&) = delete;
   void operator()(const Buffer& rhs, Buffer& residual)
   {
      log.Record('C');
      residual.values = rhs.values;
   }
};

struct Axpy
{
   CallLog& log;
   Real coefficient = 0;
   explicit Axpy(CallLog& state) : log(state) {}
   Axpy(const Axpy&) = delete;
   Axpy(Axpy&&) = delete;
   void operator()(Real alpha, const Buffer& work, Buffer& residual)
   {
      log.Record('X');
      coefficient = alpha;
      for (size_t i = 0; i < 3; ++i) residual.values[i] += alpha*work.values[i];
   }
};

bool CheckResidualFormation()
{
   const Buffer rhs{{5, 7, -1}};
   Buffer x{{1, -2, .5}}, residual{{101, 102, 103}}, work{{-101, -102, -103}};
   const auto saved_x = x.values, saved_rhs = rhs.values;
   CallLog log;
   Apply apply(log); Copy copy(log); Axpy axpy(log);
   static_assert(std::is_same_v<decltype(ComputeTrueResidual(
      apply, rhs, x, residual, work, copy, axpy)), void>);
   ComputeTrueResidual(apply, rhs, x, residual, work, copy, axpy);
   bool ok = Check(log.count == 3 && log.order == std::array{'A', 'C', 'X'}
      && axpy.coefficient == -1, "residual callbacks are borrowed and called once in dependency order");
   ok = Check(residual.values == std::array<Real, 3>{3, 13, -3}
      && work.values == std::array<Real, 3>{2, -6, 2}, "residual formation computes rhs - A*x") && ok;
   return Check(x.values == saved_x && rhs.values == saved_rhs,
      "residual formation preserves solution and RHS") && ok;
}
} // namespace

int main()
{
   bool ok = CheckNormsAndConvergence();
   ok = CheckResults() && ok;
   return CheckResidualFormation() && ok ? 0 : 1;
}

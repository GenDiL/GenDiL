// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Solvers/GMRES/restartedgmres.hpp>
#include <gendil/Solvers/GMRES/preconditioning.hpp>
#include <gendil/Solvers/GMRES/leastsquares.hpp>
#include <gendil/Algebra/Vector/reductions.hpp>
#include <gendil/Utilities/View/Layouts/stridedlayout.hpp>
#include <gendil/Utilities/KernelContext/KernelConfigurations/host.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <utility>

using namespace gendil;

namespace {

// Deliberately avoid the owning global Vector and delete all vector copies.
// The same passive buffers exercise borrowing views on every path.
struct TestVector
{
   std::array<Real, 3> values{};
   TestVector() = default;
   TestVector(const TestVector&) = delete;
   TestVector& operator=(const TestVector&) = delete;
   Real& operator[](Integer i) { return values[i]; }
   Real operator[](Integer i) const { return values[i]; }
};

struct TestWorkspace
{
   TestVector basis_storage[4], work, direction_storage[3];
   Real h[12]{}, cosines[3]{}, sines[3]{}, reduced_rhs[4]{}, coefficients[3]{};
   StridedView<1, TestVector> basis = MakeFIFOView(basis_storage, 4);
   StridedView<1, TestVector> directions = MakeFIFOView(direction_storage, 3);
   StridedView<2> hessenberg = MakeFIFOView(h, 4, 3);
};

struct TestOptions : GMRESOptions
{
   TestOptions()
   {
      restart = 3;
      max_iterations = 100;
      relative_tolerance = 1.e-12;
      absolute_tolerance = 0;
   }
};

template<gmres_detail::PreconditioningMode mode>
using ModeTag = std::integral_constant<gmres_detail::PreconditioningMode, mode>;

static_assert(gmres_detail::PreconditioningMode::FixedRight !=
              gmres_detail::PreconditioningMode::FlexibleRight);

struct SequentialOperations
{
   template<class X, class Y>
   void Copy(const X& x, Y& y) const
   { for (int i = 0; i < 3; ++i) { y[i] = x[i]; } }

   template<class X>
   void Zero(X& x) const
   { for (int i = 0; i < 3; ++i) { x[i] = 0; } }

   template<class X>
   void Normalize(X& x, Real norm) const
   { for (int i = 0; i < 3; ++i) { x[i] /= norm; } }

   template<class X, class Y>
   void Axpy(Real a, const X& x, Y& y) const
   { for (int i = 0; i < 3; ++i) { y[i] += a * x[i]; } }

   template<class X, class Y>
   Real Dot(const X& x, const Y& y) const
   {
      Real result = 0;
      for (int i = 0; i < 3; ++i) { result += x[i] * y[i]; }
      return result;
   }

   template<class X>
   Real Norm(const X& x) const
   { return std::hypot(x[0], std::hypot(x[1], x[2])); }

   template<class Basis, class Coefficients, class Y>
   void Combine(const Basis& basis, Integer count,
                const Coefficients& coefficients, Real alpha, Y& y) const
   {
      for (Integer j = 0; j < count; ++j)
      { Axpy(alpha * coefficients[j], basis(j), y); }
   }
};

// Each callback is borrowed by the numerical algorithms, including mutable
// callables whose state cannot be copied or moved during a solve.
template<class Function>
struct BorrowedCallback
{
   Function function;
   Integer calls = 0;
   explicit BorrowedCallback(Function input) : function(std::move(input)) { }
   BorrowedCallback(const BorrowedCallback&) = delete;
   BorrowedCallback(BorrowedCallback&&) = delete;

   template<class... Args>
   decltype(auto) operator()(Args&&... args)
   {
      ++calls;
      return function(std::forward<Args>(args)...);
   }
};

template<class Map, class Rhs, class Operations, class Preconditioning,
         class Orthogonalization, class Inverse = std::nullptr_t>
auto SolveWithCallbacks(const Map& map, const Rhs& rhs, const GMRESOptions& options,
   TestWorkspace& workspace, TestVector& solution, Operations& operations,
   Preconditioning, const Orthogonalization&, Inverse&& inverse = nullptr)
{
   using namespace gmres_detail;
   constexpr auto mode = Preconditioning::value;
   auto apply = [&](const auto& input, auto& output) { map(input, output); };
   BorrowedCallback copy{[&](const auto& input, auto& output)
   { operations.Copy(input, output); }};
   BorrowedCallback divide{[&](auto& output, Real length)
   { operations.Normalize(output, length); }};
   BorrowedCallback norm{[&](const auto& input) { return operations.Norm(input); }};
   BorrowedCallback dot{[&](const auto& x, const auto& y)
   { return operations.Dot(x, y); }};
   auto axpy = [&](Real alpha, const auto& input, auto& output)
   { operations.Axpy(alpha, input, output); };
   auto combine = [&](auto& basis, Integer count, const auto& coefficients,
                      Real alpha, auto& output)
   { operations.Combine(basis, count, coefficients, alpha, output); };
   BorrowedCallback true_residual{[&]
   {
      solver_detail::ComputeTrueResidual(apply, rhs, solution, workspace.basis(0),
         workspace.work, copy, axpy);
      return norm(workspace.basis(0));
   }};
   auto expand = [&](Integer column)
   {
      if constexpr (mode == PreconditioningMode::None)
      {
         apply(workspace.basis(column), workspace.work);
      }
      else
      {
         auto& direction = workspace.directions(
            mode == PreconditioningMode::FixedRight ? 0 : column);
         inverse(workspace.basis(column), direction);
         apply(direction, workspace.work);
      }
   };
   auto store = [](auto& coefficients, Integer i, Real value)
   { StoreCoefficient(coefficients, i, value); };
   auto project = [&](auto& basis, Integer count, const auto& input, auto& coefficients)
   { ProjectFromDot(dot, basis, count, input, coefficients); };
   auto accumulate = [](auto& destination, const auto& source, Integer count)
   {
      for (Integer i = 0; i < count; ++i) { destination[i] += source[i]; }
   };
   auto orthogonalize = [&](Integer column)
   {
      auto projections = MakeFIFOView(&workspace.hessenberg(0, column), 4);
      if constexpr (std::is_same_v<Orthogonalization, ModifiedGramSchmidt>)
      {
         return OrthogonalizeModifiedGramSchmidt(
            dot, axpy, store, workspace.basis, column + 1, workspace.work, projections);
      }
      else
      {
         return OrthogonalizeClassicalGramSchmidtTwoPass(
            project, combine, accumulate, workspace.basis, column + 1,
            workspace.work, projections, workspace.coefficients);
      }
   };
   BorrowedCallback build_column{[&](Integer column)
   {
      return BuildArnoldiColumn(workspace.work, column, expand, orthogonalize, norm);
   }};
   BorrowedCallback correction{[&](Integer count)
   {
      if constexpr (mode == PreconditioningMode::FixedRight)
      {
         operations.Zero(workspace.work);
         combine(workspace.basis, count, workspace.coefficients, 1, workspace.work);
         inverse(workspace.work, workspace.directions(0));
         axpy(1, workspace.directions(0), solution);
      }
      else if constexpr (mode == PreconditioningMode::FlexibleRight)
      {
         combine(workspace.directions, count, workspace.coefficients, 1, solution);
      }
      else
      {
         combine(workspace.basis, count, workspace.coefficients, 1, solution);
      }
   }};
   auto initialize = [&](Real beta)
   {
      InitializeReducedLeastSquares(workspace.cosines, workspace.sines,
         workspace.reduced_rhs, workspace.coefficients, options.restart, beta);
   };
   auto reduce = [&](Integer column, Real remainder)
   {
      return UpdateReducedQR(workspace.hessenberg, workspace.cosines,
         workspace.sines, workspace.reduced_rhs, column, remainder);
   };
   auto backsolve = [&](Integer count)
   {
      return BackSubstitute(workspace.hessenberg, workspace.reduced_rhs,
         workspace.coefficients, count);
   };
   return RestartedGMRES(options, workspace, norm(rhs), true_residual,
      build_column, correction, copy, divide, initialize, reduce, backsolve);
}

struct TestMap
{
   template<class X, class Y>
   void operator()(const X& x, Y& y) const
   {
      y[0] = 4 * x[0] + x[1];
      y[1] = x[0] + 3 * x[1] + x[2];
      y[2] = x[1] + 2 * x[2];
   }
};

struct ScaledIdentity
{
   Real scale = 1;
   template<class X, class Y>
   void operator()(const X& x, Y& y) const
   { for (int i = 0; i < 3; ++i) { y[i] = scale * x[i]; } }
};

struct NonsymmetricMap
{
   template<class X, class Y>
   void operator()(const X& x, Y& y) const
   {
      y[0] = 4 * x[0] + 2 * x[1];
      y[1] = x[0] + 3 * x[1] + x[2];
      y[2] = -x[1] + 4 * x[2];
   }
};

struct Jacobi
{
   Integer calls = 0;
   Jacobi() = default;
   Jacobi(const Jacobi&) = delete;
   template<class X, class Y>
   void operator()(const X& x, Y& y)
   {
      ++calls;
      y[0] = x[0] / 4;
      y[1] = x[1] / 3;
      y[2] = x[2] / 2;
   }
};

struct VaryingPreconditioner
{
   Integer calls = 0;
   VaryingPreconditioner() = default;
   VaryingPreconditioner(const VaryingPreconditioner&) = delete;
   template<class X, class Y>
   void operator()(const X& x, Y& y)
   {
      const Integer iteration = calls++;
      const Real scale = 1 + 0.2 * (iteration % 3);
      for (int i = 0; i < 3; ++i) { y[i] = x[i] / scale; }
   }
};

int failures = 0;

void Check(bool condition, const char* name)
{
   if (!condition)
   {
      std::printf("GMRES core failure: %s\n", name);
      ++failures;
   }
}

template<class Orthogonalization>
void CheckSequential(const Orthogonalization& orthogonalization)
{
   SequentialOperations operations;
   const std::array<Real, 3> rhs{5, 5, 3};
   TestOptions options;
   TestWorkspace workspace;
   TestVector solution;
   const auto solve = [&](const auto& map, const auto& b, const auto& config,
                          const auto& preconditioning, auto&... inverse)
   {
      return SolveWithCallbacks(map, b, config, workspace, solution, operations,
         preconditioning, orthogonalization, inverse...);
   };
   const ModeTag<gmres_detail::PreconditioningMode::None> unpreconditioned;
   auto result = solve(TestMap{}, rhs, options, unpreconditioned);
   Check(result.success && result.relative_error < options.relative_tolerance &&
         std::abs(solution[0] - 1) < 1.e-11 &&
         std::abs(solution[1] - 1) < 1.e-11 &&
         std::abs(solution[2] - 1) < 1.e-11, "noncopyable vectors");
   Check(std::hypot(5 - 4 * solution[0] - solution[1],
      std::hypot(5 - solution[0] - 3 * solution[1] - solution[2],
         3 - solution[1] - 2 * solution[2])) < 1.e-10,
      "independently recomputed original-system residual");
   solution.values.fill(1);
   result = solve(TestMap{}, rhs, options, unpreconditioned);
   Check(result.success && result.iterations == 0,
      "exact initial guess needs no Arnoldi columns");

   operations.Zero(solution);
   const std::array<Real, 3> nonsymmetric_rhs{6, 5, 3};
   result = solve(NonsymmetricMap{}, nonsymmetric_rhs, options, unpreconditioned);
   Check(result.success &&
      std::hypot(6 - 4 * solution[0] - 2 * solution[1],
         std::hypot(5 - solution[0] - 3 * solution[1] - solution[2],
            3 + solution[1] - 4 * solution[2])) < 1.e-10,
      "nonsymmetric original-system residual");

   operations.Zero(solution);
   options.restart = 1;
   constexpr Real unused = 83;
   for (Real& value : workspace.h) { value = unused; }
   for (Real& value : workspace.cosines) { value = unused; }
   for (Real& value : workspace.sines) { value = unused; }
   for (Real& value : workspace.reduced_rhs) { value = unused; }
   for (Real& value : workspace.coefficients) { value = unused; }
   for (Integer j = 1; j < 4; ++j)
   { workspace.basis(j).values.fill(unused); }
   result = solve(TestMap{}, rhs, options, unpreconditioned);
   Check(result.success && result.iterations > 3, "multiple restarts");
   bool untouched = workspace.hessenberg(2, 0) == unused &&
                    workspace.hessenberg(3, 0) == unused;
   for (Integer j = 1; j < 3; ++j)
   {
      untouched &= workspace.cosines[j] == unused && workspace.sines[j] == unused &&
                   workspace.coefficients[j] == unused;
      for (Integer i = 0; i < 4; ++i)
      { untouched &= workspace.hessenberg(i, j) == unused; }
   }
   for (Integer j = 2; j < 4; ++j)
   { untouched &= workspace.reduced_rhs[j] == unused; }
   for (Integer j = 1; j < 4; ++j)
      for (Real value : workspace.basis(j).values) { untouched &= value == unused; }
   Check(untouched, "smaller restart leaves unused operands unchanged");

   operations.Zero(solution);
   options.restart = 3;
   Jacobi inverse;
   result = solve(TestMap{}, rhs, options,
      ModeTag<gmres_detail::PreconditioningMode::FixedRight>{}, inverse);
   Check(result.success && inverse.calls == result.iterations + 1,
         "fixed right preconditioner reapplied to final correction");

   operations.Zero(solution);
   VaryingPreconditioner varying;
   options.restart = 1;
   result = solve(TestMap{}, rhs, options,
                  ModeTag<gmres_detail::PreconditioningMode::FlexibleRight>{}, varying);
   Check(result.success && result.iterations > 3 &&
         varying.calls == result.iterations,
         "flexible binary preconditioner history spans restarts without correction calls");

   // These scales expose reciprocal normalization, squared Givens radii, and
   // residual-scaled breakdown tests that reject valid operator columns.
   for (Real scale : {1.e-300, 1.e-150, 1., 1.e150, 1.e300})
   {
      operations.Zero(solution);
      options.restart = 3;
      const std::array<Real, 3> scaled_rhs{scale, scale, scale};
      result = solve(ScaledIdentity{scale}, scaled_rhs, options, unpreconditioned);
      Check(result.success && result.iterations == 1 &&
            std::abs(solution[0] - 1) < 1.e-11, "scaled identity");
   }

   operations.Zero(solution);
   result = solve(ScaledIdentity{0}, rhs, options, unpreconditioned);
   Check(!result.success && result.iterations == 1 &&
         result.relative_error == 1 && solution[0] == 0, "zero operator");

   const std::array<Real, 3> huge_rhs{1.e100, 0, 0};
   result = solve(ScaledIdentity{1.e-300}, huge_rhs, options, unpreconditioned);
   Check(!result.success && solution[0] == 0 &&
         std::isfinite(result.relative_error), "bad backsolve preserves solution");

   const std::array<Real, 3> nonfinite_rhs{
      std::numeric_limits<Real>::quiet_NaN(), 0, 0};
   result = solve(ScaledIdentity{}, nonfinite_rhs, options, unpreconditioned);
   Check(!result.success && result.iterations == 0 &&
         std::isinf(result.relative_error), "nonfinite RHS");

   options.max_iterations = 0;
   result = solve(ScaledIdentity{}, rhs, options, unpreconditioned);
   Check(!result.success && result.iterations == 0 && result.relative_error == 1,
         "zero iteration budget reports true initial residual");

   const std::array<Real, 3> zero_rhs{0, 0, 0};
   solution[0] = 1.e-8;
   options.absolute_tolerance = 1.e-7;
   result = solve(ScaledIdentity{}, zero_rhs, options, unpreconditioned);
   Check(result.success && result.iterations == 0 && result.relative_error == 1.e-8,
         "absolute initial stopping with zero RHS");
}

void CheckArnoldiOperations()
{
   const std::array<std::array<Real, 3>, 2> vectors{{{1, 0, 0}, {0, 1, 0}}};
   auto basis = [&](Integer i) -> const auto& { return vectors[i]; };
   std::array<Real, 3> work{2, 3, 4};
   std::array<Real, 2> projections{}, corrections{};
   SequentialOperations operations;
   int dot_calls = 0, axpy_calls = 0;
   auto dot = [&](const auto& x, const auto& y)
   { ++dot_calls; return operations.Dot(x, y); };
   auto axpy = [&](Real alpha, const auto& x, auto& y)
   { ++axpy_calls; operations.Axpy(alpha, x, y); };
   auto store = [](auto& c, Integer i, Real value) { c[i] = value; };
   Check(gmres_detail::OrthogonalizeModifiedGramSchmidt(
      dot, axpy, store, basis, 2, work, projections), "MGS completes");
   Check(work == std::array<Real, 3>{0, 0, 4} &&
         projections == std::array<Real, 2>{2, 3} &&
         dot_calls == 2 && axpy_calls == 2, "MGS exact projections and call counts");

   std::array<char, 4> calls{};
   size_t next_call = 0;
   auto project = [&](auto& v, Integer count, const auto& x, auto& c)
   {
      calls.at(next_call++) = 'P';
      for (Integer j = 0; j < count; ++j) { c[j] = dot(v(j), x); }
   };
   auto combine = [&](auto& v, Integer count, const auto& c, Real alpha, auto& y)
   {
      calls.at(next_call++) = 'C';
      for (Integer j = 0; j < count; ++j) { axpy(alpha * c[j], v(j), y); }
   };
   auto accumulate = [](auto& c, const auto& second, Integer count)
   { for (Integer j = 0; j < count; ++j) { c[j] += second[j]; } };
   work = {2, 3, 4};
   Check(gmres_detail::OrthogonalizeClassicalGramSchmidtTwoPass(
      project, combine, accumulate, basis, 2, work, projections, corrections),
      "CGS2 completes");
   Check(work == std::array<Real, 3>{0, 0, 4} &&
         projections == std::array<Real, 2>{2, 3} &&
         calls == std::array<char, 4>{'P', 'C', 'P', 'C'},
         "CGS2 exact projections and classical update order");

   bool correct_column = false;
   auto expand = [&](Integer column)
   { correct_column = column == 1; work = {2, 3, 4}; };
   auto orthogonalize = [&](Integer column)
   { return gmres_detail::OrthogonalizeModifiedGramSchmidt(
         dot, axpy, store, basis, column + 1, work, projections); };
   auto norm = [&](const auto& x) { return operations.Norm(x); };
   const auto result = gmres_detail::BuildArnoldiColumn(
      work, 1, expand, orthogonalize, norm);
   Check(correct_column && result.valid &&
         std::abs(result.norm_before_orthogonalization - std::sqrt(Real(29))) < 1.e-14 &&
         result.orthogonal_remainder_norm == 4,
         "Arnoldi column reports the selected column and both norms");
}

void CheckTrueResidual()
{
   using Values = std::array<Real, 2>;
   Values residual{}, work{}, solution{1, 2};
   const Values rhs{5, 10};
   auto apply = [](const auto& x, auto& y)
   { y[0] = 2 * x[0]; y[1] = 3 * x[1]; };
   auto copy = [](const auto& x, auto& y) { y = x; };
   auto axpy = [](Real alpha, const auto& x, auto& y)
   { for (size_t i = 0; i < y.size(); ++i) { y[i] += alpha * x[i]; } };
   auto norm = [](const auto& x) { return std::hypot(x[0], x[1]); };
   solver_detail::ComputeTrueResidual(apply, rhs, solution, residual, work, copy, axpy);
   const Real length = norm(residual);
   Check(length == 5 && residual == Values{3, 4} && solution == Values{1, 2},
         "true residual composition preserves the solution");
}

void CheckNumericalHelpers()
{
   Real h[2]{3, -999}, cosines[1]{-999}, sines[1]{-999};
   Real coefficients[1]{-999}, reduced_rhs[2]{-999, -999};
   auto hessenberg = MakeFIFOView(h, 2, 1);
   HostKernelConfiguration context;
   gmres_detail::InitializeReducedLeastSquares(
      context, cosines, sines, reduced_rhs, coefficients, 1, 5);
   Check(cosines[0] == 1 && sines[0] == 0 && coefficients[0] == 0 &&
         reduced_rhs[0] == 5 && reduced_rhs[1] == 0,
         "reduced initialization replaces poisoned active entries");
   Check(gmres_detail::UpdateReducedQR(
      context, hessenberg, cosines, sines, reduced_rhs, 0, 4).status ==
         gmres_detail::ReducedStatus::Usable && h[0] == 5 && h[1] == 0 &&
         reduced_rhs[0] == 3 && reduced_rhs[1] == -4,
         "3-4-5 Givens rotation records the supplied subdiagonal");
   Check(gmres_detail::BackSubstitute(
      context, hessenberg, reduced_rhs, coefficients, 1) && coefficients[0] == Real(0.6),
      "exact one-column back substitution");

   TestOptions exact_tolerance;
   exact_tolerance.relative_tolerance = 0;
   Check(!solver_detail::Converged(1.e-300, 1.e300,
      exact_tolerance.relative_tolerance, exact_tolerance.absolute_tolerance),
         "underflowed relative quotient cannot satisfy exact tolerance");
   Check(std::isfinite(Hypot(1.e300, 1.e300)), "large stable hypot");
   Check(Hypot(1.e-300, 1.e-300) > 0, "small stable hypot");
   TestWorkspace workspace;
   workspace.hessenberg(0, 0) = 1.e300;
   workspace.hessenberg(1, 0) = 1.e300;
   workspace.reduced_rhs[0] = 1;
   Check(gmres_detail::UpdateReducedQR(workspace.hessenberg, workspace.cosines,
            workspace.sines, workspace.reduced_rhs, 0,
            workspace.hessenberg(1, 0)).status ==
         gmres_detail::ReducedStatus::Usable &&
         std::isfinite(workspace.hessenberg(0, 0)), "large Givens rotation");

   workspace.hessenberg(0, 0) = 0;
   workspace.hessenberg(1, 0) = 0;
   Check(gmres_detail::UpdateReducedQR(workspace.hessenberg, workspace.cosines,
            workspace.sines, workspace.reduced_rhs, 0,
            workspace.hessenberg(1, 0)).status ==
         gmres_detail::ReducedStatus::Singular, "zero Givens radius");

   workspace.hessenberg(0, 0) = std::numeric_limits<Real>::infinity();
   Check(gmres_detail::UpdateReducedQR(workspace.hessenberg, workspace.cosines,
            workspace.sines, workspace.reduced_rhs, 0,
            workspace.hessenberg(1, 0)).status ==
         gmres_detail::ReducedStatus::NonFinite, "nonfinite reduced column");

   workspace.hessenberg(0, 0) = 0;
   workspace.reduced_rhs[0] = 1;
   workspace.coefficients[0] = 7;
   Check(!gmres_detail::BackSubstitute(workspace.hessenberg, workspace.reduced_rhs,
            workspace.coefficients, 1) &&
         workspace.coefficients[0] == 7, "zero diagonal does not fabricate coefficient");

   workspace.hessenberg(0, 0) = 1.e-300;
   Check(gmres_detail::BackSubstitute(workspace.hessenberg, workspace.reduced_rhs,
            workspace.coefficients, 1) &&
         std::isfinite(workspace.coefficients[0]), "small nonzero diagonal retained");
   workspace.reduced_rhs[0] = 1.e300;
   Check(!gmres_detail::BackSubstitute(workspace.hessenberg, workspace.reduced_rhs,
            workspace.coefficients, 1),
         "nonfinite coefficient rejects backsolve");
}

void CheckRestartControl()
{
   using namespace gmres_detail;
   const auto exercise = [](Integer budget, bool singular_second, bool bad_column,
                            bool misleading_estimate)
   {
      TestOptions options;
      options.restart = 3;
      options.max_iterations = budget;
      options.relative_tolerance = 0;
      TestWorkspace buffers;
      SequentialOperations operations;
      Integer residual_calls = 0, build_calls = 0, correction_calls = 0;
      Integer accepted_count = 0, solve_calls = 0;
      auto residual = [&]
      {
         ++residual_calls;
         buffers.basis(0).values = {1, 0, 0};
         return correction_calls == 0 ? Real(1) : Real(0.5);
      };
      auto build = [&](Integer column)
      {
         ++build_calls;
         buffers.work.values = {0, 1, 0};
         return ArnoldiColumnResult{!(bad_column && column == 1), 2, 1};
      };
      auto correct = [&](Integer count)
      {
         ++correction_calls;
         accepted_count = count;
      };
      auto copy = [&](const TestVector& x, TestVector& y) { operations.Copy(x, y); };
      auto divide = [&](TestVector& x, Real length) { operations.Normalize(x, length); };
      auto initialize = [&](Real) { };
      auto reduce = [&](Integer column, Real)
      {
         const auto status = singular_second && column == 1
            ? ReducedStatus::Singular : ReducedStatus::Usable;
         return ReducedQRResult{status, misleading_estimate ? Real(0) : Real(1)};
      };
      auto solve = [&](Integer) { ++solve_calls; return true; };
      const auto result = RestartedGMRES(options, buffers, Real(1), residual, build,
         correct, copy, divide, initialize, reduce, solve);
      Check(!result.success && result.iterations == build_calls &&
         result.relative_error == (correction_calls == 0 ? Real(1) : Real(0.5)),
         "restart controller reports recomputed residual and attempted columns");
      if (bad_column)
      {
         Check(build_calls == 2 && solve_calls == 0 && correction_calls == 0 &&
            residual_calls == 2, "failed column rejects incomplete correction");
      }
      else if (singular_second)
      {
         Check(build_calls == 2 && accepted_count == 1 && correction_calls == 1 &&
            solve_calls == 1, "dependent column retains prior accepted correction");
      }
      else if (misleading_estimate)
      {
         Check(correction_calls == budget && accepted_count == 1 &&
            residual_calls == budget + 1,
            "reduced convergence estimates cannot replace true residual checks");
      }
      else
      {
         Check(build_calls == budget && accepted_count == budget &&
            correction_calls == 1 && solve_calls == 1 && residual_calls == 2,
            "mid-cycle budget applies exactly the accepted correction");
      }
   };
   exercise(2, false, false, false);
   exercise(3, true, false, false);
   exercise(3, false, true, false);
   exercise(2, false, false, true);
}

} // namespace

int main()
{
   CheckArnoldiOperations();
   CheckTrueResidual();
   CheckSequential(ModifiedGramSchmidt{});
   CheckSequential(ClassicalGramSchmidtTwoPass{});
   CheckNumericalHelpers();
   CheckRestartControl();
   return failures == 0 ? 0 : 1;
}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
// CPU source-ordering probe, not CUDA/HIP evidence. All checks survive NDEBUG.
#include <gendil/Solvers/GMRES/gmres.hpp>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <span>

namespace cpu_probe
{
using gendil::Integer;
using gendil::Real;

[[noreturn]] void Fail(const char *message)
{
   std::fprintf(stderr, "FAIL: %s\n", message);
   std::abort();
}
void Check(bool condition, const char *message)
{
   if (!condition)
      Fail(message);
}
bool Near(Real a, Real b)
{
   return std::abs(a - b) < 1e-12;
}

void DelayReader(const std::atomic<bool>& primary_ready)
{
   const double deadline = omp_get_wtime() + 5;
   while (!primary_ready.load())
   {
      Check(omp_get_wtime() < deadline,
            "timed out waiting for the primary OpenMP participant");
   }
   // Delay this reader without adding a collective completion boundary.
   const double observation_time = omp_get_wtime() + 0.015;
   while (omp_get_wtime() < observation_time) {}
}

// Native-shaped collective context: exercise the actual native-context overloads
// with real CPU participants, including the is_threaded_v selection.
struct Context
{
   using kernel_configuration_type = Context;
   static constexpr bool is_host_configuration = false;
   static constexpr size_t thread_block_dim = 1;
   static constexpr size_t batch_size = 1;
   unsigned lane;
   unsigned synchronizations = 0;
   static constexpr unsigned GetNumberOfThreads()
   {
      return 4;
   }
   unsigned GetLinearThreadIndex() const
   {
      return lane;
   }
   void Synchronize()
   {
      ++synchronizations;
#pragma omp barrier
   }
};
static_assert(gendil::is_threaded_v<Context>);

template <class F> void Collectively(F &&action)
{
   constexpr unsigned participants = 4;
   const int dynamic_teams = omp_get_dynamic();
   omp_set_dynamic(0);
#pragma omp parallel num_threads(participants) shared(action)
   {
      Check(omp_get_num_threads() == participants,
            "cooperative probe requires four OpenMP participants");
      Context context{static_cast<unsigned>(omp_get_thread_num())};
      action(context);
   }
   omp_set_dynamic(dynamic_teams);
}

// Atomic cells make the intentionally delayed old-reader probe race-free even
// if the QR entry barrier regresses. It observes the race in ordering, not UB.
struct TrackedH
{
   std::array<std::atomic<Real>, 6> cells{};
   std::atomic<bool> armed{false}, old_reader_finished{false}, early_write{false};
   struct Cell
   {
      TrackedH &parent;
      size_t index;
      operator Real() const
      {
         return parent.cells[index].load();
      }
      Cell &operator=(Real value)
      {
         if (parent.armed.load() && !parent.old_reader_finished.load())
            parent.early_write.store(true);
         parent.cells[index].store(value);
         return *this;
      }
   };
   Cell operator()(size_t i, size_t j)
   {
      return {*this, i + 3 * j};
   }
   Real operator()(size_t i, size_t j) const
   {
      return cells[i + 3 * j].load();
   }
};

void QRCase(Real diagonal, Real next, gendil::gmres_detail::ReducedStatus expected)
{
   TrackedH h;
   h(0, 0) = diagonal;
   h(1, 0) = 99;
   std::array<Real, 2> cosines{}, sines{}, coefficients{};
   std::array<Real, 3> rhs{};
   std::atomic<bool> primary_entering{false};
   h.armed = true;
   Collectively(
       [&](Context &context)
       {
          gendil::gmres_detail::InitializeReducedLeastSquares(
              context, cosines, sines, rhs, coefficients, 2, 5);
          if (context.lane == 0)
             primary_entering = true;
          if (context.lane == 3)
          {
             DelayReader(primary_entering);
             Check(Real(h(1, 0)) == 99, "QR overwrote an earlier coefficient reader");
             h.old_reader_finished = true;
          }
          // Legitimate uniform norm with no barrier: no reliance on a reduction to
          // finish unrelated coefficient reads before the first QR mutation.
          const std::array<Real, 2> work{3, 4};
          auto norm_without_barrier = [](const auto &values)
          {
             return std::hypot(values[0], values[1]);
          };
          Check(norm_without_barrier(work) == 5, "barrier-free mathematical norm");
          const unsigned before = context.synchronizations;
          const auto result =
              gendil::gmres_detail::UpdateReducedQR(context, h, cosines, sines, rhs, 0, next);
          Check(result.status == expected, "uniform QR status");
          Check(!h.early_write.load(), "QR first write precedes old reader completion");
          Check(context.synchronizations - before ==
                    (expected == gendil::gmres_detail::ReducedStatus::Usable ? 4 : 3),
                "QR owns only its entry, validation, read-completion and publication "
                "boundaries");
          if (expected == gendil::gmres_detail::ReducedStatus::Usable)
          {
             Check(Near(result.residual_norm_estimate, 4),
                   "QR returns replicated residual estimate");
             Check(gendil::gmres_detail::BackSubstitute(context, h, rhs, coefficients, 1),
                   "back substitution");
             Check(Near(coefficients[0], 0.6), "back substitution result");
          }
          // The solve boundary owns failure/reuse completion, independent of the norm.
          context.Synchronize();
          if (context.lane == 0)
             coefficients.fill(17);
          context.Synchronize();
          Check(coefficients[0] == 17,
                "shared scratch reusable after terminal completion");
       });
}

struct Basis
{
   std::array<std::array<Real, 2>, 3> values{{{{1, 0}}, {{0, 1}}, {{2, 3}}}};
   auto &operator()(Integer i)
   {
      return values[i];
   }
};

void ProjectionCase()
{
   Basis basis;
   std::array<Real, 2> input{4, 5};
   std::array<Real, 3> coefficients{-1, -1, -1};
   Collectively(
       [&](Context &context)
       {
          unsigned calls = 0;
          // Replicated read-only inputs: this user dot needs no coordination/scratch.
          auto dot = [&](const auto &x, const auto &y)
          {
             ++calls;
             return x[0] * y[0] + x[1] * y[1];
          };
          const unsigned before = context.synchronizations;
          gendil::LocalProject(context, dot, basis, 3, input, coefficients);
          Check(calls == 3,
                "projection retains the supplied dot instance and call order");
          Check(context.synchronizations - before == 1,
                "projection publishes once, not once per coefficient");
          Check(coefficients[0] == 4 && coefficients[1] == 5 && coefficients[2] == 23,
                "all participants see completed coefficients");
          context.Synchronize();
          gendil::gmres_detail::StoreCoefficient(context, coefficients, 1, 9);
          Check(coefficients[1] == 9, "MGS scalar publication");
       });
}

void BackSubstitutionFailure(Real diagonal)
{
   TrackedH h;
   h(0, 0) = diagonal;
   std::array<Real, 1> rhs{1}, solution{7};
   Collectively(
       [&](Context &context)
       {
          Check(!gendil::gmres_detail::BackSubstitute(context, h, rhs, solution, 1),
                "uniform reduced failure");
          Check(solution[0] == 7, "invalid coefficient was not committed");
          context.Synchronize();
       });
}

void SharedActivePrefixAxpy()
{
   std::array<Real, 8> destination{1, 2, 3, 4, 5, 99, 99, 99};
   const std::array<Real, 8> correction{2, 4, 6, 8, 10, 123, 123, 123};
   auto output = std::span<Real>{destination}.first(5);
   auto input = std::span<const Real>{correction}.first(5);
   Collectively(
       [&](Context &context)
       {
          gendil::Axpy(context, Real(1), input, output);
          for (size_t i = 0; i < 5; ++i)
             Check(destination[i] == 3 * Real(i + 1),
                   "one writer per active coefficient");
          for (size_t i = 5; i < 8; ++i)
             Check(destination[i] == 99, "coefficient Axpy preserves inactive capacity");
          Check(context.synchronizations == 1, "shared Axpy owns its completion");
       });
}

struct LargeScaleMap
{
   Context &context;
   mutable unsigned calls = 0;
   template <class X, class Y> void operator()(const X &input, Y &output) const
   {
      ++calls;
      gendil::VisitLocal(context,
                           output,
                           [&](size_t i)
                           {
                              output[i] = 1e300 * input[i];
                           });
      context.Synchronize();
   }
};

void LocalSolveEarlyFailureReuse()
{
   std::array<std::array<Real, 2>, 2> vectors{};
   std::array<Real, 2> work{}, rhs{1e-100, 0}, solution{};
   std::array<Real, 7> dense{}; // m=1: H(2), c(1), s(1), g(2), y(1).
   auto basis =
       gendil::MakeView(gendil::ReferenceContainer{vectors}, gendil::VectorLayout{2});
   auto storage = gendil::MakeVectorView(dense.data(), dense.size());
   auto reduced = gendil::MakeGMRESPackedDense(storage, 1, 2);
   reduced.hessenberg(1, 0) = 99;
   gendil::NoDirections directions;
   auto buffers = gendil::TieGMRESBuffers(basis,
                                          work,
                                          directions,
                                          reduced.hessenberg,
                                          reduced.cosines,
                                          reduced.sines,
                                          reduced.reduced_rhs,
                                          reduced.coefficients);
   std::atomic<bool> primary_finishing_norm{false}, final_reader_finished{false};
   gendil::GMRESOptions options;
   options.restart = 1;
   options.max_iterations = 2;
   options.relative_tolerance = 1e-12;
   options.absolute_tolerance = 0;
   Collectively(
       [&](Context &context)
       {
          LargeScaleMap system{context};
          unsigned norm_calls = 0, dot_calls = 0;
          // A legitimate weighted inner product with intentionally unstable multiply
          // order: w*b overflows during Arnoldi, although the explicit norm is finite.
          auto dot = [&](const auto &a, const auto &b)
          {
             ++dot_calls;
             return a[0] * (1e200 * b[0]) + a[1] * (1e200 * b[1]);
          };
          auto norm = [&](const auto &values)
          {
             ++norm_calls;
             if (norm_calls == 4)
             {
                if (context.lane == 0)
                   primary_finishing_norm = true;
                if (context.lane == 3)
                {
                   DelayReader(primary_finishing_norm);
                }
             }
             const Real value = std::hypot(1e100 * values[0], 1e100 * values[1]);
             if (norm_calls == 4 && context.lane == 3)
                final_reader_finished = true;
             // Intermediate norms precede mutation of their own vector operand;
             // complete those reads just like a native collective reduction.
             // Only the final norm intentionally relies on the driver's terminal
             // completion boundary, which is the regression under test here.
             if (norm_calls != 4) { context.Synchronize(); }
             return value;
          };
          auto project = [&](auto &sequence, Integer count, const auto &x, auto &c)
          {
             gendil::LocalProject(context, dot, sequence, count, x, c);
          };
          auto combine =
              [&](auto &sequence, Integer count, const auto &c, Real alpha, auto &out)
          {
             gendil::LocalCombine(context, sequence, count, c, alpha, out);
          };
          gendil::gmres_detail::NoPreconditioner preconditioner;
          const auto result = gendil::gmres_detail::LocalSolve<gendil::gmres_detail::PreconditioningMode::None>(
              context,
              system,
              preconditioner,
              rhs,
              solution,
              dot,
              options,
              buffers,
              norm,
              project,
              combine,
              gendil::ClassicalGramSchmidtTwoPass{});
          Check(!result.success && result.iterations == 1,
                "actual LocalSolve rejects nonfinite projection");
          Check(norm_calls == 4 && dot_calls == 1 && system.calls == 3,
                "early orthogonalization failure path");
          Check(reduced.hessenberg(1, 0) == 99, "early failure never entered reduced QR");
          Check(final_reader_finished.load(),
                "LocalSolve terminal completion precedes reuse");
          if (context.lane == 0)
             vectors[0].fill(17);
          context.Synchronize();
          Check(vectors[0][0] == 17, "actual LocalSolve buffers reusable on return");
       });
}

struct CoupledMap
{
   Context &context;
   mutable unsigned calls = 0;

   template <class X, class Y> void operator()(const X &input, Y &output) const
   {
      ++calls;
      gendil::VisitLocal(context,
                           output,
                           [&](size_t i)
                           {
                              if (i == 0)
                              {
                                 output[i] = 4 * input[0] + input[1];
                              }
                              if (i == 1)
                              {
                                 output[i] = input[0] + 3 * input[1] + input[2];
                              }
                              if (i == 2)
                              {
                                 output[i] = input[1] + 2 * input[2];
                              }
                           });
      context.Synchronize(); // Complete the map's own neighbor reads and writes.
   }
};

struct ReplicatedCollectiveDot
{
   Context &context;
   unsigned calls = 0;

   explicit ReplicatedCollectiveDot(Context &supplied) : context(supplied)
   {
   }
   ReplicatedCollectiveDot(const ReplicatedCollectiveDot &) = delete;

   template <class X, class Y> Real operator()(const X &first, const Y &second)
   {
      ++calls;
      Real result = 0;
      for (size_t i = 0; i < 3; ++i)
      {
         result += first[i] * second[i];
      }
      // Every CPU participant evaluates the complete same product. This barrier
      // completes operand reads before the next operation may overwrite them;
      // it is not a claim about unrelated Hessenberg reads or GPU reductions.
      context.Synchronize();
      return result;
   }
};

template<bool Variable>
struct CompletedBinaryPreconditioner
{
   Context& context;
   unsigned calls = 0;
   std::array<std::array<Real, 3>, 2> history{};

   explicit CompletedBinaryPreconditioner(Context& supplied) : context(supplied) { }
   CompletedBinaryPreconditioner(const CompletedBinaryPreconditioner&) = delete;

   template<class Input, class Output>
   void operator()(const Input& input, Output& output)
   {
      const unsigned iteration = calls++;
      auto& direction = history[iteration % history.size()];
      for (size_t i = 0; i < direction.size(); ++i)
      {
         const Real variation = Variable ? Real(1) + Real((iteration + i) % 3) / 8 : Real(1);
         direction[i] = input[i] / ((Real(2) + Real(i)) * variation);
      }
      gendil::VisitLocal(context, output, [&](const size_t i) { output[i] = direction[i]; });
      context.Synchronize(); // Complete all input reads and publish the binary output.
   }
};

template <int Mode, class Orthogonalization>
void SuccessfulCooperativeSolve(Orthogonalization orthogonalization, const char *name)
{
   std::array<std::array<Real, 3>, 3> vectors{};
   std::array<std::array<Real, 3>, 2> direction_values{};
   std::array<Real, 3> work{}, rhs{6, 10, 8}, solution{};
   std::array<Real, 15> dense{};
   auto basis =
       gendil::MakeView(gendil::ReferenceContainer{vectors}, gendil::VectorLayout{3});
   auto storage = gendil::MakeVectorView(dense.data(), dense.size());
   auto reduced = gendil::MakeGMRESPackedDense(storage, 2, 3);
   auto directions = gendil::MakeView(
      gendil::ReferenceContainer{direction_values}, gendil::VectorLayout{2});
   auto buffers = gendil::TieGMRESBuffers(basis,
                                          work,
                                          directions,
                                          reduced.hessenberg,
                                          reduced.cosines,
                                          reduced.sines,
                                          reduced.reduced_rhs,
                                          reduced.coefficients);
   gendil::GMRESOptions options;
   options.restart = 2;
   options.max_iterations = 300;
   options.relative_tolerance = 1e-12;
   options.absolute_tolerance = 0;
   std::array<gendil::IterativeSolverResult, 4> results{};
   std::array<unsigned, 4> applications{}, dot_calls{}, preconditioner_calls{};

   Collectively(
       [&](Context &context)
       {
          CoupledMap system{context};
          ReplicatedCollectiveDot dot{context};
          CompletedBinaryPreconditioner<Mode == 2> preconditioner{context};
          const auto result = [&]
          {
             if constexpr (Mode == 0)
             {
                return gendil::GMRES(
                   context, system, rhs, solution, dot, options, buffers, orthogonalization);
             }
             else if constexpr (Mode == 1)
             {
                return gendil::GMRES(context, system, preconditioner, rhs, solution,
                   dot, options, buffers, orthogonalization);
             }
             else
             {
                return gendil::FGMRES(context, system, preconditioner, rhs, solution,
                   dot, options, buffers, orthogonalization);
             }
          }();

          Check(result.success, "complete cooperative solve converges");
          Check(result.iterations > options.restart, "complete solve executes a restart");
          const Integer corrections =
              (result.iterations + options.restart - 1) / options.restart;
          Check(corrections >= 2, "multiple completed restart cycles");
          Check(system.calls == result.iterations + corrections + 1,
                "expansions, initial residual and correction residuals all executed");
          if constexpr (Mode == 1)
          {
             Check(preconditioner.calls == result.iterations + corrections,
                "fixed-right correction reapplies the exact binary preconditioner");
          }
          else if constexpr (Mode == 2)
          {
             Check(preconditioner.calls == result.iterations,
                "flexible correction uses stored directions without reapplying P");
             for (size_t j = 0; j < direction_values.size(); ++j)
             {
                for (size_t i = 0; i < direction_values[j].size(); ++i)
                {
                   Check(direction_values[j][i] == preconditioner.history[j][i],
                      "FGMRES preserves the exact most recent direction history");
                }
             }
          }
          const Real r0 = rhs[0] - (4 * solution[0] + solution[1]);
          const Real r1 = rhs[1] - (solution[0] + 3 * solution[1] + solution[2]);
          const Real r2 = rhs[2] - (solution[1] + 2 * solution[2]);
          Check(std::hypot(r0, r1, r2) <= 1e-12 * std::sqrt(200.0),
                "independent original-system residual");
          Check(std::abs(solution[0] - 1) < 1e-10 && std::abs(solution[1] - 2) < 1e-10 &&
                    std::abs(solution[2] - 3) < 1e-10,
                "independent known solution");
          results[context.lane] = result;
          applications[context.lane] = system.calls;
          dot_calls[context.lane] = dot.calls;
          preconditioner_calls[context.lane] = preconditioner.calls;
          // No caller-side completion before reuse: the public local driver owns it.
          if (context.lane == 0)
          {
             dense.fill(17);
             vectors[0].fill(19);
          }
          context.Synchronize(); // Publish the test's new writes and result records.
          for (size_t lane = 0; lane < 4; ++lane)
          {
             Check(results[lane].success &&
                       results[lane].iterations == result.iterations &&
                       results[lane].relative_error == result.relative_error,
                   "every participant receives the same complete result");
             Check(applications[lane] == system.calls && dot_calls[lane] == dot.calls &&
                      preconditioner_calls[lane] == preconditioner.calls,
                   "uniform collective application sequence");
          }
          Check(dense[0] == 17 && vectors[0][0] == 19,
                "completed solver scratch is reusable");
       });
   std::printf(
       "%s cooperative solve: %zu steps, %u map calls, %u dot calls, %u P calls, error %.3e\n",
       name,
       size_t(results[0].iterations),
       applications[0],
       dot_calls[0],
       preconditioner_calls[0],
       results[0].relative_error);
}

void ThreadLocalNoBarriers()
{
   struct Local
   {
      static constexpr bool IsHost()
      {
         return true;
      }
      unsigned calls = 0;
      void Synchronize()
      {
         ++calls;
      }
   } context;
   // No is_threaded_v metadata: independent thread-local solves must not acquire
   // native block-wide barriers, even if their native config exposes that method.
   static_assert(!gendil::is_threaded_v<Local>);
   TrackedH h;
   h(0, 0) = 3;
   std::array<Real, 1> c{}, s{}, y{};
   std::array<Real, 2> g{};
   gendil::gmres_detail::InitializeReducedLeastSquares(context, c, s, g, y, 1, 5);
   auto result = gendil::gmres_detail::UpdateReducedQR(context, h, c, s, g, 0, 4);
   Check(result.status == gendil::gmres_detail::ReducedStatus::Usable, "thread-local QR");
   Check(gendil::gmres_detail::BackSubstitute(context, h, g, y, 1), "thread-local back substitution");
   gendil::gmres_detail::StoreCoefficient(context, y, 0, 2);
   Check(context.calls == 0, "thread-local dense operations have no collective barriers");
}
} // namespace cpu_probe

int main()
{
   using namespace cpu_probe;
   QRCase(3, 4, gendil::gmres_detail::ReducedStatus::Usable);
   QRCase(0, 0, gendil::gmres_detail::ReducedStatus::Singular);
   QRCase(std::numeric_limits<Real>::infinity(), 4, gendil::gmres_detail::ReducedStatus::NonFinite);
   QRCase(std::numeric_limits<Real>::quiet_NaN(), 4, gendil::gmres_detail::ReducedStatus::NonFinite);
   ProjectionCase();
   BackSubstitutionFailure(0);
   BackSubstitutionFailure(std::numeric_limits<Real>::infinity());
   SharedActivePrefixAxpy();
   LocalSolveEarlyFailureReuse();
   SuccessfulCooperativeSolve<0>(gendil::ModifiedGramSchmidt{}, "MGS");
   SuccessfulCooperativeSolve<0>(gendil::ClassicalGramSchmidtTwoPass{}, "CGS2");
   SuccessfulCooperativeSolve<1>(gendil::ModifiedGramSchmidt{}, "Fixed-right MGS");
   SuccessfulCooperativeSolve<1>(gendil::ClassicalGramSchmidtTwoPass{}, "Fixed-right CGS2");
   SuccessfulCooperativeSolve<2>(gendil::ModifiedGramSchmidt{}, "Flexible-right MGS");
   SuccessfulCooperativeSolve<2>(gendil::ClassicalGramSchmidtTwoPass{}, "Flexible-right CGS2");
   ThreadLocalNoBarriers();
   std::puts(
       "CPU collective completion probe passed (4 participants; always-active checks)");
}

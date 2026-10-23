// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Solvers/GMRES/gmres.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

namespace gmres_allocation_probe {

std::atomic<bool> active{false};
std::atomic<size_t> calls{0};

void* Allocate(const size_t bytes)
{
   if (active.load(std::memory_order_relaxed))
   {
      calls.fetch_add(1, std::memory_order_relaxed);
   }
   if (void* result = std::malloc(bytes == 0 ? 1 : bytes)) { return result; }
   throw std::bad_alloc{};
}

struct Guard
{
   Guard()
   {
      calls.store(0, std::memory_order_relaxed);
      active.store(true, std::memory_order_relaxed);
   }
   ~Guard() { active.store(false, std::memory_order_relaxed); }
};

} // namespace gmres_allocation_probe

void* operator new(const size_t bytes) { return gmres_allocation_probe::Allocate(bytes); }
void* operator new[](const size_t bytes) { return gmres_allocation_probe::Allocate(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, size_t) noexcept { std::free(pointer); }

using namespace gendil;

namespace {

int failures = 0;

void Check(const bool condition, const char* expression, const int line)
{
   if (!condition)
   {
      std::fprintf(stderr, "GMRES variants:%d: %s\n", line, expression);
      ++failures;
   }
}

#define CHECK(...) Check((__VA_ARGS__), #__VA_ARGS__, __LINE__)

bool Near(const Real left, const Real right, const Real tolerance = 1.e-10)
{
   return std::isfinite(left) && std::isfinite(right) &&
      std::abs(left - right) <= tolerance * std::max(Real(1), std::abs(right));
}

void Set(Vector& vector, const std::array<Real, 3>& values)
{
   Real* data = vector.WriteHostData();
   for (size_t i = 0; i < 3; ++i) { data[i] = values[i]; }
}

bool Near(const Vector& vector, const std::array<Real, 3>& expected,
          const Real tolerance = 1.e-10)
{
   const Real* data = vector.ReadHostData();
   for (size_t i = 0; i < 3; ++i)
   {
      if (!Near(data[i], expected[i], tolerance)) { return false; }
   }
   return true;
}

struct Matrix
{
   std::array<Real, 9> values{4, 1, 0, 1, 3, 1, 0, 1, 2};
   size_t InputSize() const { return 3; }
   size_t OutputSize() const { return 3; }

   template<class Input, class Output>
   void operator()(const Input& input, Output& output) const
   {
      const Real* x = ReadHostVector(input);
      Real* y = WriteHostVector(output);
      for (size_t i = 0; i < 3; ++i)
      {
         y[i] = 0;
         for (size_t j = 0; j < 3; ++j) { y[i] += values[3 * i + j] * x[j]; }
      }
   }

   template<class Input, class Output>
   void ApplyAdd(const Input& input, Output& output) const
   {
      const Real* x = ReadHostVector(input);
      Real* y = ReadWriteHostVector(output);
      for (size_t i = 0; i < 3; ++i)
      {
         for (size_t j = 0; j < 3; ++j) { y[i] += values[3 * i + j] * x[j]; }
      }
   }
};

struct ScaledIdentity
{
   Real scale = 1;
   size_t size = 3;
   size_t InputSize() const { return size; }
   size_t OutputSize() const { return size; }

   template<class Input, class Output>
   void operator()(const Input& input, Output& output) const
   {
      const Real* x = ReadHostVector(input);
      Real* y = WriteHostVector(output);
      for (size_t i = 0; i < size; ++i) { y[i] = scale * x[i]; }
   }

   template<class Input, class Output>
   void ApplyAdd(const Input& input, Output& output) const
   {
      const Real* x = ReadHostVector(input);
      Real* y = ReadWriteHostVector(output);
      for (size_t i = 0; i < size; ++i) { y[i] += scale * x[i]; }
   }
};

struct CountingJacobi
{
   Integer calls = 0;
   CountingJacobi() = default;
   CountingJacobi(const CountingJacobi&) = delete;
   CountingJacobi& operator=(const CountingJacobi&) = delete;

   template<class Input, class Output>
   void operator()(const Input& input, Output& output)
   {
      ++calls;
      const Real* x = ReadHostVector(input);
      Real* y = WriteHostVector(output);
      y[0] = x[0] / 4;
      y[1] = x[1] / 3;
      y[2] = x[2] / 2;
   }
};

struct VaryingInverse
{
   Integer next_iteration = 0;
   std::array<Integer, 300> history{};

   template<class Input, class Output>
   void operator()(const Input& input, Output& output)
   {
      const Integer iteration = next_iteration++;
      if (iteration < history.size()) { history[iteration] = iteration; }
      const Real factor = 1 + Real(iteration % 3) / 4;
      const Real* x = ReadHostVector(input);
      Real* y = WriteHostVector(output);
      for (size_t i = 0; i < 3; ++i) { y[i] = x[i] / factor; }
   }
};

struct InnerGMRESInverse
{
   const Matrix& map;
   const GMRESOptions& options;
   GMRESWorkspace& workspace;
   EuclideanDot<HostKernelConfiguration>& dot;
   mutable Integer calls = 0;
   mutable Integer inner_iterations = 0;
   mutable bool valid_inner_results = true;

   template<class Input, class Output>
   void operator()(const Input& input, Output& output) const
   {
      ++calls;
      Zero<false>(output);
      const auto result = GMRES<HostKernelConfiguration>(
         map, input, output, dot, options, workspace);
      inner_iterations += result.iterations;
      valid_inner_results = valid_inner_results && result.iterations == 1 &&
         std::isfinite(result.relative_error);
   }
};

struct WeightedDot
{
   mutable Integer calls = 0;

   template<class First, class Second>
   Real operator()(const First& first, const Second& second) const
   {
      ++calls;
      const Real* x = ReadHostVector(first);
      const Real* y = ReadHostVector(second);
      return x[0] * y[0] + 2 * x[1] * y[1] + 4 * x[2] * y[2];
   }
};

struct InvalidDot
{
   Real result;

   template<class First, class Second>
   Real operator()(const First&, const Second&) const { return result; }
};

// These deliberately old-shaped members are traps: public solves must not
// discover metric customization. Only callbacks explicitly selected below run.
struct MetricTraps
{
   Vector& scratch;
   Integer dots = 0, discovered = 0;
   template<class X, class Y> Real operator()(const X& x, const Y& y)
   {
      ++dots;
      return Dot<HostKernelConfiguration>(scratch, x, y);
   }
   template<class X> Real Norm(const X&) { ++discovered; return -1; }
   template<class... Args> void Project(Args&&...) { ++discovered; }
   template<class... Args> void Combine(Args&&...) { ++discovered; }
};

// Simulate a failed operator application after the initial true-residual check.
struct FaultyMap : ScaledIdentity
{
   Real invalid;
   mutable Integer calls = 0;

   template<class Input, class Output>
   void operator()(const Input& input, Output& output) const
   {
      ScaledIdentity::operator()(input, output);
      if (++calls == 2) { WriteHostVector(output)[0] = invalid; }
   }
};

namespace local_protocol {

using Array = std::array<Real, 2>;
using Context = KernelContext<HostKernelConfiguration, 0>;

// Both replacement paths compute the same map. Separate counters reveal which
// dispatch the local solver used without changing the mathematical operator.
struct Map
{
   mutable size_t callable_calls = 0;
   mutable size_t apply_calls = 0;

   void operator()(const Array& input, Array& output) const
   {
      ++callable_calls;
      output[0] = 2 * input[0];
      output[1] = 4 * input[1];
   }
};

size_t GetInputSize(const Map&) { return 2; }
size_t GetOutputSize(const Map&) { return 2; }
void Apply(const Map& map, const Array& input, Array& output)
{
   ++map.apply_calls;
   output[0] = 2 * input[0];
   output[1] = 4 * input[1];
}
void ApplyAdd(const Map&, const Array& input, Array& output)
{
   output[0] += 2 * input[0];
   output[1] += 4 * input[1];
}

struct CallableOnly
{
   void operator()(const Array& input, Array& output) const { output = input; }
};

struct MissingAdd : CallableOnly
{
   size_t InputSize() const { return 2; }
   size_t OutputSize() const { return 2; }
};

struct MissingSizes : CallableOnly
{
   void ApplyAdd(const Array& input, Array& output) const
   {
      output[0] += input[0];
      output[1] += input[1];
   }
};

// The preconditioner accepts this basis type, but the system map deliberately
// accepts only Array. Right-preconditioned solves apply A to directions, not V.
struct DistinctBasis
{
   Array values{};
   size_t size() const { return values.size(); }
   Real& operator[](const size_t index) { return values[index]; }
   const Real& operator[](const size_t index) const { return values[index]; }
};

struct IdentityInverse
{
   template<class Input, class Output>
   void operator()(const Input& input, Output& output) const
   {
      output[0] = input[0];
      output[1] = input[1];
   }
};

template<class Entry>
using ScratchVectorCollection = decltype(MakeView(
   ReferenceContainer{std::declval<std::array<Entry, 3>&>()},
   VectorLayout{3}));

using ScratchDirectionCollection = decltype(MakeView(
   ReferenceContainer{std::declval<std::array<Array, 2>&>()},
   VectorLayout{2}));
using ScratchDenseStorage = decltype(MakeVectorView(
   std::declval<Real*>(), size_t{15}));
using ScratchDense = decltype(MakeGMRESPackedDense(
   std::declval<ScratchDenseStorage>(), size_t{2}, size_t{3}));

template<class Basis = Array, class Work = Array>
using Scratch = decltype(TieGMRESBuffers(
   std::declval<ScratchVectorCollection<Basis>&>(), std::declval<Work&>(),
   std::declval<ScratchDirectionCollection&>(),
   std::declval<decltype(std::declval<ScratchDense&>().hessenberg)&>(),
   std::declval<decltype(std::declval<ScratchDense&>().cosines)&>(),
   std::declval<decltype(std::declval<ScratchDense&>().sines)&>(),
   std::declval<decltype(std::declval<ScratchDense&>().reduced_rhs)&>(),
   std::declval<decltype(std::declval<ScratchDense&>().coefficients)&>()));

struct LocalProduct
{
   Context& context;
   template<class X, class Y> Real operator()(const X& x, const Y& y)
   { return LocalDot(context, x, y); }
};

template<class SystemMap, class Workspace>
concept AcceptsLocalGMRES = requires(Context& context, Workspace& workspace,
   LocalProduct& dot, const GMRESOptions& options, const SystemMap& map,
   const Array& rhs, Array& solution)
{
   GMRES(context, map, rhs, solution, dot, options, workspace);
};

template<class SystemMap, class Workspace>
concept AcceptsLocalRightGMRES = requires(Context& context, Workspace& workspace,
   LocalProduct& dot, const GMRESOptions& options, const SystemMap& map,
   const Array& rhs, const IdentityInverse& inverse, Array& solution)
{
   GMRES(context, map, inverse, rhs, solution, dot, options, workspace);
};

template<class SystemMap, class Workspace>
concept AcceptsLocalFGMRES = requires(Context& context, Workspace& workspace,
   LocalProduct& dot, const GMRESOptions& options, const SystemMap& map,
   const Array& rhs, const IdentityInverse& inverse, Array& solution)
{
   FGMRES(context, map, inverse, rhs, solution, dot, options, workspace);
};

static_assert(VectorMap<Map, Array, Array>);
static_assert(!VectorMap<Map, Array, const Array>);
static_assert(!VectorMap<CallableOnly, Array, Array>);
static_assert(!VectorMap<MissingAdd, Array, Array>);
static_assert(!VectorMap<MissingSizes, Array, Array>);
static_assert(AcceptsLocalGMRES<Map, Scratch<>>);
static_assert(AcceptsLocalRightGMRES<Map, Scratch<>>);
static_assert(AcceptsLocalFGMRES<Map, Scratch<>>);
static_assert(AcceptsLocalGMRES<CallableOnly, Scratch<>>);
static_assert(AcceptsLocalRightGMRES<CallableOnly, Scratch<>>);
static_assert(AcceptsLocalFGMRES<CallableOnly, Scratch<>>);
static_assert(AcceptsLocalGMRES<MissingAdd, Scratch<>>);
static_assert(AcceptsLocalRightGMRES<MissingAdd, Scratch<>>);
static_assert(AcceptsLocalFGMRES<MissingAdd, Scratch<>>);
static_assert(AcceptsLocalGMRES<MissingSizes, Scratch<>>);
static_assert(AcceptsLocalRightGMRES<MissingSizes, Scratch<>>);
static_assert(AcceptsLocalFGMRES<MissingSizes, Scratch<>>);
static_assert(!AcceptsLocalGMRES<Map, Scratch<Array, const Array>>);
static_assert(!AcceptsLocalRightGMRES<Map, Scratch<Array, const Array>>);
static_assert(!AcceptsLocalFGMRES<Map, Scratch<Array, const Array>>);
static_assert(!AcceptsLocalGMRES<Map, Scratch<DistinctBasis>>);
static_assert(AcceptsLocalRightGMRES<Map, Scratch<DistinctBasis>>);
static_assert(AcceptsLocalFGMRES<Map, Scratch<DistinctBasis>>);

template<int Mode, class Orthogonalization>
void CheckDispatch(const Orthogonalization& orthogonalization)
{
   Real arena[Context::shared_memory_block_size]{};
   Context context(arena);
   using Basis = std::conditional_t<Mode == 0, Array, DistinctBasis>;
   std::array<Basis, 3> basis{};
   std::array<Array, 2> directions{};
   Array work{}, solution{};
   const Array rhs{2, 8};
   std::array<Real, 15> dense{};
   auto basis_view = MakeView(
      ReferenceContainer{basis}, VectorLayout{basis.size()});
   auto direction_view = MakeView(
      ReferenceContainer{directions}, VectorLayout{directions.size()});
   auto dense_view = MakeVectorView(dense.data(), dense.size());
   auto packed_dense = MakeGMRESPackedDense(dense_view, 2, 3);
   auto scratch = TieGMRESBuffers(
      basis_view, work, direction_view, packed_dense.hessenberg,
      packed_dense.cosines, packed_dense.sines, packed_dense.reduced_rhs,
      packed_dense.coefficients);
   Map map;
   Array callable_result{}, apply_result{};
   map(rhs, callable_result);
   Apply(map, rhs, apply_result);
   CHECK(callable_result == apply_result);
   Array sum{1, -1};
   ApplyAdd(map, rhs, sum);
   CHECK(sum[0] == 5 && sum[1] == 31);
   map.callable_calls = map.apply_calls = 0;

   const GMRESOptions options{2, 12, 1.e-12, 0};
   LocalProduct dot{context};
   IterativeSolverResult result{};
   if constexpr (Mode == 0)
   {
      result = GMRES(context, map, rhs, solution, dot, options, scratch, orthogonalization);
   }
   else if constexpr (Mode == 1)
   {
      result = GMRES(context, map, IdentityInverse{}, rhs, solution, dot, options, scratch, orthogonalization);
   }
   else
   {
      result = FGMRES(context, map, IdentityInverse{}, rhs, solution, dot, options, scratch, orthogonalization);
   }
   CHECK(result.success && result.iterations == 2);
   CHECK(Near(solution[0], 1) && Near(solution[1], 2));
   CHECK(map.callable_calls == 0 && map.apply_calls == result.iterations + 2);
}

} // namespace local_protocol

template<class Orthogonalization>
void Solves(const Orthogonalization& orthogonalization)
{
   const Matrix map;
   GMRESOptions options;
   options.restart = 3;
   options.max_iterations = 150;
   options.relative_tolerance = 1.e-12;
   Vector rhs(3), solution(3);
   Set(rhs, {5, 5, 3});
   Set(solution, {0, 0, 0});
   const auto owning = GMRES(map, rhs, options, solution, orthogonalization);
   CHECK(owning.success && owning.iterations <= 3);
   CHECK(Near(solution, {1, 1, 1}));

   // Physical capacity exceeds both the vector dimension and active restart.
   auto workspace = MakeGMRESWorkspace(3, 7);
   const Real* basis_storage = nullptr;
   {
      auto workspace_view = MakeGMRESWorkspaceView(workspace);
      auto original_basis = workspace_view.basis(0);
      WriteHostVector(original_basis)[0] = 0;
      basis_storage = ReadHostVector(original_basis);
   }
   Real* dense_storage = workspace.dense_workspace.data();
   options.restart = 2;
   EuclideanDot<HostKernelConfiguration> product{workspace.reduction_workspace};
   Set(solution, {0.4, -0.3, 0.2});
   const auto reused = GMRES<HostKernelConfiguration>(
      map, rhs, solution, product, options, workspace, orthogonalization);
   CHECK(reused.success && reused.iterations > options.restart);
   CHECK(options.max_iterations == 150);
   CHECK(Near(solution, {1, 1, 1}));
   CHECK(workspace.restart_capacity == 7);
   {
      auto workspace_view = MakeGMRESWorkspaceView(workspace);
      CHECK(GetStride<1>(workspace_view.hessenberg.layout) == 8);
      CHECK(ReadHostVector(workspace_view.basis(0)) == basis_storage);
   }
   CHECK(workspace.dense_workspace.data() == dense_storage);

   // The legacy entry point lends actual owning vectors without copying them.
   std::vector<Vector> basis;
   basis.reserve(3);
   for (int i = 0; i < 3; ++i) { basis.emplace_back(3); }
   Vector work(3), legacy_solution(3);
   std::vector<Real> h(6), c(2), s(2), g(3), y(2);
   Set(legacy_solution, {0.4, -0.3, 0.2});
   auto dot = [](const Vector& first, const Vector& second)
   {
      const Real* x = first.ReadHostData();
      const Real* z = second.ReadHostData();
      return x[0] * z[0] + x[1] * z[1] + x[2] * z[2];
   };
   const auto legacy_map = [&map](const Vector& input, Vector& output)
   { map(input, output); };
   const auto legacy_identity = [](const Vector& input, Vector& output)
   { output = input; };
   static_assert(!VectorMap<decltype(legacy_map), Vector, Vector>);
   const auto legacy = GMRES_no_alloc(legacy_map, rhs, dot,
      options.max_iterations, options.restart,
      options.relative_tolerance, legacy_solution,
      basis, work, h, c, s, g, y, orthogonalization);
   CHECK(legacy.success && legacy.iterations == reused.iterations);
   CHECK(Near(legacy_solution, {1, 1, 1}));
   CHECK(Near(legacy.relative_error, reused.relative_error));

   Set(rhs, {0, 0, 0});
   Set(legacy_solution, {1.e-8, 0, 0});
   const auto zero_rhs_legacy = GMRES_no_alloc(legacy_identity, rhs, dot,
      0, 2, 1.e-7, legacy_solution, basis, work, h, c, s, g, y,
      orthogonalization);
   CHECK(zero_rhs_legacy.success && zero_rhs_legacy.iterations == 0);
   CHECK(zero_rhs_legacy.relative_error == 1.e-8);
   Set(legacy_solution, {1, 0, 0});
   const auto zero_restart = GMRES_no_alloc(legacy_identity, rhs, dot,
      10, 0, 1.e-7, legacy_solution, basis, work, h, c, s, g, y,
      orthogonalization);
   CHECK(!zero_restart.success && zero_restart.iterations == 0);
   CHECK(zero_restart.relative_error == 1 && Near(legacy_solution, {1, 0, 0}));
   Set(rhs, {5, 5, 3});

   options.max_iterations = 150;
   options.restart = 3;
   CountingJacobi fixed;
   Set(solution, {0, 0, 0});
   auto result = GMRES<HostKernelConfiguration>(
      map, fixed, rhs, solution, product, options, workspace, orthogonalization);
   CHECK(result.success && Near(solution, {1, 1, 1}));
   CHECK(fixed.calls == result.iterations + 1);
   CHECK(workspace.direction_capacity == 1);

   CountingJacobi flexible_fixed;
   Set(solution, {0, 0, 0});
   result = FGMRES<HostKernelConfiguration>(
      map, flexible_fixed, rhs, solution, product, options, workspace, orthogonalization);
   CHECK(result.success && Near(solution, {1, 1, 1}));
   CHECK(flexible_fixed.calls == result.iterations);
   CHECK(workspace.direction_capacity >= static_cast<size_t>(options.restart));

   VaryingInverse varying;
   options.restart = 1;
   Set(solution, {0, 0, 0});
   result = FGMRES<HostKernelConfiguration>(
      map, varying, rhs, solution, product, options, workspace, orthogonalization);
   CHECK(result.success && result.iterations > 3 && Near(solution, {1, 1, 1}));
   CHECK(varying.next_iteration == result.iterations && result.iterations <= varying.history.size());
   for (Integer i = 0; i < result.iterations; ++i) { CHECK(varying.history[i] == i); }

   // Exercise convenient owning overloads independently of reusable overloads.
   options.restart = 3;
   Set(solution, {0, 0, 0});
   result = GMRES<HostKernelConfiguration>(
      map, CountingJacobi{}, rhs, solution, product, options, workspace, orthogonalization);
   CHECK(result.success && Near(solution, {1, 1, 1}));
   Set(solution, {0, 0, 0});
   result = FGMRES<HostKernelConfiguration>(
      map, VaryingInverse{}, rhs, solution, product, options, workspace, orthogonalization);
   CHECK(result.success && Near(solution, {1, 1, 1}));

   // A metric providing only dot products must also control the norm fallback.
   WeightedDot weighted;

   Set(solution, {0, 0, 0});
   result = GMRES<HostKernelConfiguration>(map, rhs, solution, weighted, options, workspace, orthogonalization);
   CHECK(result.success && weighted.calls > 0 && Near(solution, {1, 1, 1}));
   options.max_iterations = 0;
   Set(rhs, {1, 2, 3});
   Set(solution, {0, 1, 0});
   result = GMRES<HostKernelConfiguration>(
      ScaledIdentity{}, rhs, solution, weighted, options, workspace, orthogonalization);
   CHECK(!result.success && result.iterations == 0);
   CHECK(Near(result.relative_error, std::sqrt(39.0 / 45.0), 1.e-14));
   CHECK(Near(solution, {0, 1, 0}));
}

template<class Orthogonalization>
void EdgeCases(const Orthogonalization& orthogonalization)
{
   GMRESOptions options;
   options.restart = 3;
   options.max_iterations = 30;
   options.relative_tolerance = 1.e-12;
   auto workspace = MakeGMRESWorkspace(3, 5);
   EuclideanDot<HostKernelConfiguration> product{workspace.reduction_workspace};
   auto norm = [&](const auto& x) { return ScaledEuclideanNorm(product, x); };
   auto project = [&](auto& basis, Integer count, const auto& x, auto& c)
   { ProjectEuclidean(product, basis, count, x, c); };
   auto combine = [](auto& basis, Integer count, const auto& c, Real alpha, auto& x)
   { HostDrivenFusedCombine<false>(basis, count, c, alpha, x); };
   Vector rhs(3), solution(3);
   Set(rhs, {5, 5, 3});
   Set(solution, {1, 1, 1});
   auto result = GMRES<HostKernelConfiguration>(
      Matrix{}, rhs, solution, product, options, workspace, norm, project, combine, orthogonalization);
   CHECK(result.success && result.iterations == 0 && result.relative_error == 0);

   Set(rhs, {0, 0, 0});
   Set(solution, {0, 0, 0});
   result = GMRES<HostKernelConfiguration>(
      ScaledIdentity{}, rhs, solution, product, options, workspace, norm, project, combine,
      orthogonalization);
   CHECK(result.success && result.iterations == 0 && result.relative_error == 0);
   Set(solution, {0, 0, 2});
   result = GMRES<HostKernelConfiguration>(
      ScaledIdentity{}, rhs, solution, product, options, workspace, norm, project, combine,
      orthogonalization);
   CHECK(result.success && Near(solution, {0, 0, 0}));

   options.max_iterations = 0;
   options.absolute_tolerance = 1.e-7;
   Set(solution, {1.e-8, 0, 0});
   result = GMRES<HostKernelConfiguration>(
      ScaledIdentity{}, rhs, solution, product, options, workspace, norm, project, combine,
      orthogonalization);
   CHECK(result.success && result.iterations == 0);
   CHECK(result.relative_error == 1.e-8 && solution.ReadHostData()[0] == 1.e-8);
   options.max_iterations = 30;
   options.absolute_tolerance = 0;

   // Underflow of a reported relative residual must not satisfy exact tolerance.
   auto exact_options = options;
   exact_options.max_iterations = 0;
   exact_options.relative_tolerance = 0;

   Set(rhs, {1.e300, 1.e-300, 0});
   Set(solution, {1.e300, 0, 0});
   result = GMRES<HostKernelConfiguration>(
      ScaledIdentity{}, rhs, solution, product, exact_options, workspace, norm, project, combine,
      orthogonalization);
   CHECK(!result.success && result.iterations == 0);
   CHECK(solution.ReadHostData()[0] == 1.e300 && solution.ReadHostData()[1] == 0);

   Set(rhs, {1, 2, 3});
   Set(solution, {0, 0, 0});
   result = GMRES<HostKernelConfiguration>(
      ScaledIdentity{0}, rhs, solution, product, options, workspace, norm, project, combine,
      orthogonalization);
   CHECK(!result.success && result.iterations == 1 && result.relative_error == 1);
   CHECK(Near(solution, {0, 0, 0}));

   const Matrix singular{{1, 0, 0, 0, 0, 0, 0, 0, 2}};
   Set(rhs, {1, 0, 2});
   result = GMRES<HostKernelConfiguration>(
      singular, rhs, solution, product, options, workspace, norm, project, combine, orthogonalization);
   CHECK(result.success && Near(solution, {1, 0, 1}));
   Set(rhs, {1, 1, 2});
   Set(solution, {0, 0, 0});
   result = GMRES<HostKernelConfiguration>(
      singular, rhs, solution, product, options, workspace, norm, project, combine, orthogonalization);
   CHECK(!result.success && std::isfinite(result.relative_error));

   // Reject invalid metrics both in the initial norm and during projection.
   const Real infinity = std::numeric_limits<Real>::infinity();
   const Real nan = std::numeric_limits<Real>::quiet_NaN();
   for (const Real invalid : {-1.0, infinity, nan})
   {
      Set(solution, {0, 0, 0});
      InvalidDot invalid_dot{invalid};
      result = GMRES<HostKernelConfiguration>(
         ScaledIdentity{}, rhs, solution, invalid_dot, options, workspace, orthogonalization);
      CHECK(!result.success && result.iterations == 0 && std::isinf(result.relative_error));
      CHECK(Near(solution, {0, 0, 0}));
      auto invalid_norm = [invalid](const auto&) { return invalid; };
      result = GMRES<HostKernelConfiguration>(
         ScaledIdentity{}, rhs, solution, product, options, workspace, invalid_norm, project, combine,
         orthogonalization);
      CHECK(!result.success && result.iterations == 0 && std::isinf(result.relative_error));
   }
   for (const Real invalid : {infinity, nan})
   {
      Set(solution, {0, 0, 0});
      InvalidDot projection{invalid};
      auto invalid_project = [&](auto& basis, Integer count, const auto& input, auto& c)
      { ProjectFromDot(projection, basis, count, input, c); };
      result = GMRES<HostKernelConfiguration>(
         ScaledIdentity{}, rhs, solution, projection, options, workspace, norm, invalid_project, combine,
         orthogonalization);
      CHECK(!result.success && result.iterations == 1 && result.relative_error == 1);
      CHECK(Near(solution, {0, 0, 0}));
      result = GMRES<HostKernelConfiguration>(
         FaultyMap{{}, invalid}, rhs, solution, product, options, workspace, norm, project, combine,
         orthogonalization);
      CHECK(!result.success && result.iterations == 1 && result.relative_error == 1);
      CHECK(Near(solution, {0, 0, 0}));
      Set(rhs, {invalid, 0, 0});
      result = GMRES<HostKernelConfiguration>(
         ScaledIdentity{}, rhs, solution, product, options, workspace, norm, project, combine,
         orthogonalization);
      CHECK(!result.success && result.iterations == 0 && std::isinf(result.relative_error));
      Set(rhs, {1, 1, 2});
   }

   // An overflowing reduced correction must not mutate the last valid iterate.
   Set(rhs, {1.e100, 0, 0});
   Set(solution, {0, 0, 0});
   result = GMRES<HostKernelConfiguration>(
      ScaledIdentity{1.e-300}, rhs, solution, product, options, workspace, norm, project, combine,
      orthogonalization);
   CHECK(!result.success && result.iterations == 1 && result.relative_error == 1);
   CHECK(Near(solution, {0, 0, 0}));

   // Independently scale A and b; dependence tests must not use the RHS scale.
   for (const Real scale : {1.e-300, 1.e-150, 1.0, 1.e150, 1.e300})
   {
      Set(rhs, {scale, 0, 0});
      Set(solution, {0, 0, 0});
      result = GMRES<HostKernelConfiguration>(
         ScaledIdentity{scale}, rhs, solution, product, options, workspace, norm, project, combine,
         orthogonalization);
      CHECK(result.success && result.iterations == 1 && Near(solution, {1, 0, 0}));
      Set(solution, {0, 0, 0});
      result = GMRES<HostKernelConfiguration>(
         ScaledIdentity{}, rhs, solution, product, options, workspace, norm, project, combine,
         orthogonalization);
      CHECK(result.success && result.iterations == 1 && result.relative_error == 0);
      CHECK(solution.ReadHostData()[0] == scale);
   }
   const Real subnormal = 16 * std::numeric_limits<Real>::denorm_min();
   Set(rhs, {subnormal, 0, 0});
   Set(solution, {0, 0, 0});
   result = GMRES<HostKernelConfiguration>(
      ScaledIdentity{}, rhs, solution, product, options, workspace, norm, project, combine,
      orthogonalization);
   CHECK(result.success && result.iterations == 1 && result.relative_error == 0);
   CHECK(solution.ReadHostData()[0] == subnormal);

   Vector empty_rhs, empty_solution;
   result = GMRES(ScaledIdentity{1, 0}, empty_rhs, options, empty_solution,
      orthogonalization);
   CHECK(result.success && result.iterations == 0 && result.relative_error == 0);
}

template<class Orthogonalization>
void NonsymmetricRestarted(const Orthogonalization& orthogonalization)
{
   const Matrix map{{4, 1, -0.5, -2, 5, 1, 0.25, -1, 3}};
   Vector rhs(3), solution(3), residual(3);
   Set(solution, {1, -2, 0.5});
   map(solution, rhs);
   solution = 0;
   const GMRESOptions options{2, 150, 1.e-12, 0};
   auto workspace = MakeGMRESWorkspace(3, 2);
   EuclideanDot<HostKernelConfiguration> dot{workspace.reduction_workspace};
   const auto result = GMRES<HostKernelConfiguration>(
      map, rhs, solution, dot, options, workspace, orthogonalization);
   CHECK(result.success && result.iterations > options.restart);
   CHECK(Near(solution, {1, -2, 0.5}));
   map(solution, residual);
   const auto* b = rhs.ReadHostData();
   auto* r = residual.ReadWriteHostData();
   Real square = 0, rhs_square = 0;
   for (size_t i = 0; i < 3; ++i)
   {
      r[i] = b[i] - r[i];
      square += r[i] * r[i];
      rhs_square += b[i] * b[i];
   }
   CHECK(std::sqrt(square / rhs_square) <= options.relative_tolerance);
}

void Norms()
{
   auto workspace = MakeGMRESWorkspace(3, 3);
   EuclideanDot<HostKernelConfiguration> product{workspace.reduction_workspace};
   Vector input(3);
   for (const Real scale : {1.e-300, 1.e-200, 1.0, 1.e200, 1.e300})
   {
      Set(input, {3 * scale, 4 * scale, 0});
      const Real norm = ScaledEuclideanNorm(product, input);
      CHECK(std::isfinite(norm) && norm > 0);
      CHECK(std::abs(norm / scale - 5) < 1.e-13);
   }
   const Real unit = 8 * std::numeric_limits<Real>::denorm_min();
   Set(input, {3 * unit, 4 * unit, 0});
   CHECK(ScaledEuclideanNorm(product, input) == 5 * unit);
   Set(input, {0, 0, 0});
   CHECK(ScaledEuclideanNorm(product, input) == 0);
   Set(input, {std::numeric_limits<Real>::quiet_NaN(), 0, 0});
   CHECK(!std::isfinite(ScaledEuclideanNorm(product, input)));
   Set(input, {std::numeric_limits<Real>::infinity(), 0, 0});
   CHECK(!std::isfinite(ScaledEuclideanNorm(product, input)));
   Vector minimal_scratch(1);
   Set(input, {3, 4, 0});
   EuclideanDot<HostKernelConfiguration> minimal_product{minimal_scratch};
   CHECK(ScaledEuclideanNorm(minimal_product, input) == 5);
   CHECK(RequiredGMRESReductionWorkspaceSize(513, 3) == 12);

   Integer calls = 0;
   auto move_only_dot = [state = std::make_unique<Integer>(0), &calls]
      (const auto& first, const auto& second) mutable
   {
      ++*state;
      ++calls;
      const Real* x = ReadHostVector(first);
      const Real* y = ReadHostVector(second);
      return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
   };
   static_assert(!std::is_copy_constructible_v<decltype(move_only_dot)>);
   CHECK(NormFromDot(move_only_dot, input) == 5);
   CHECK(NormFromDot(std::move(move_only_dot), input) == 5);
   CHECK(NormFromDot(move_only_dot, input) == 5);
   CHECK(calls == 3);
}

void ExplicitOperationSelection()
{
   auto workspace = MakeGMRESWorkspace(3, 3);
   MetricTraps dot{workspace.reduction_workspace};
   EuclideanDot<HostKernelConfiguration> euclidean{workspace.reduction_workspace};
   Vector rhs(3), solution(3);
   Set(rhs, {5, 5, 3});
   solution = 0;
   const GMRESOptions options{3, 100, 1.e-12, 0};

   // Default NormFromDot and reference projection use this exact mutable dot;
   // similarly named metric members do not participate in operation selection.
   const auto reference = GMRES<HostKernelConfiguration>(
      Matrix{}, rhs, solution, dot, options, workspace, ClassicalGramSchmidtTwoPass{});
   CHECK(reference.success && Near(solution, {1, 1, 1}));
   CHECK(dot.dots > 0 && dot.discovered == 0);

   Integer norms = 0, projections = 0, combinations = 0;
   auto norm = [&](const auto& input)
   { ++norms; return ScaledEuclideanNorm(euclidean, input); };
   auto project = [&](auto& basis, Integer count, const auto& input, auto& c)
   { ++projections; ProjectEuclidean(euclidean, basis, count, input, c); };
   auto combine = [&](auto& basis, Integer count, const auto& c, Real alpha, auto& output)
   { ++combinations; HostDrivenFusedCombine<false>(basis, count, c, alpha, output); };
   const Integer previous_dots = dot.dots;
   solution = 0;
   const auto optimized = GMRES<HostKernelConfiguration>(
      Matrix{}, rhs, solution, dot, options, workspace, norm, project, combine,
      ClassicalGramSchmidtTwoPass{});
   CHECK(optimized.success && Near(solution, {1, 1, 1}));
   CHECK(norms > 0 && projections == 2 * optimized.iterations);
   CHECK(combinations >= projections && dot.dots == previous_dots && dot.discovered == 0);

   Integer move_only_calls = 0;
   auto move_only_dot = [state = std::make_unique<Integer>(0),
                         &scratch = workspace.reduction_workspace,
                         &move_only_calls]
      (const auto& first, const auto& second) mutable
   {
      ++*state;
      ++move_only_calls;
      return Dot<HostKernelConfiguration>(scratch, first, second);
   };
   static_assert(!std::is_copy_constructible_v<decltype(move_only_dot)>);
   solution = 0;
   const auto move_only = GMRES<HostKernelConfiguration>(
      Matrix{}, rhs, solution, move_only_dot, options, workspace);
   CHECK(move_only.success && Near(solution, {1, 1, 1}) && move_only_calls > 0);

   solution = 0;
   const auto temporary = GMRES<HostKernelConfiguration>(
      Matrix{}, rhs, solution,
      [&scratch = workspace.reduction_workspace](const auto& x, const auto& y)
      { return Dot<HostKernelConfiguration>(scratch, x, y); },
      options, workspace);
   CHECK(temporary.success && Near(solution, {1, 1, 1}));

   struct ContiguousBasis
   {
      Vector& storage;
      Integer& fallback_calls;
      Vector& ContiguousStorage() const { return storage; }
      size_t ContiguousStride() const { return 3; }
      auto operator()(Integer j) const
      { ++fallback_calls; return MakeVectorView(storage, 3 * j, 3); }
   };
   Vector storage(6);
   auto* entries = storage.WriteHostData();
   for (Integer i = 0; i < 6; ++i) { entries[i] = i + 1; }
   Integer fallback_calls = 0;
   ContiguousBasis basis{storage, fallback_calls};
   const std::array<Real, 2> coefficients{2, -1};
   solution = 1;
   HostDrivenFusedCombine<false>(basis, 2, coefficients, 3, solution);
   CHECK(Near(solution, {-5, -2, 1}));
   CHECK(fallback_calls == 0 && dot.discovered == 0);
}

template<class Orthogonalization>
void InnerSolverPreconditioning(const Orthogonalization& orthogonalization)
{
   const Matrix map;
   GMRESOptions outer_options;
   outer_options.restart = 3;
   outer_options.max_iterations = 12;
   outer_options.relative_tolerance = 1.e-12;
   GMRESOptions inner_options;
   inner_options.restart = 1;
   inner_options.max_iterations = 1;
   inner_options.relative_tolerance = 0;
   auto outer_workspace = MakeGMRESWorkspace(3, 3, 3);
   auto inner_workspace = MakeGMRESWorkspace(3, 1);
   EuclideanDot<HostKernelConfiguration> outer_dot{outer_workspace.reduction_workspace};
   EuclideanDot<HostKernelConfiguration> inner_dot{inner_workspace.reduction_workspace};
   InnerGMRESInverse inverse{map, inner_options, inner_workspace, inner_dot};
   Vector rhs(3), solution(3);
   Set(rhs, {5, 5, 3});
   Set(solution, {0.2, -0.3, 0.1});
   const Real* inner_basis = nullptr;
   const Real* outer_basis = nullptr;
   {
      auto inner_view = MakeGMRESWorkspaceView(inner_workspace);
      auto outer_view = MakeGMRESWorkspaceView(outer_workspace);
      auto initial_inner_basis = inner_view.basis(0);
      auto initial_outer_basis = outer_view.basis(0);
      Zero<false>(initial_inner_basis);
      Zero<false>(initial_outer_basis);
      inner_basis = ReadHostVector(initial_inner_basis);
      outer_basis = ReadHostVector(initial_outer_basis);
   }
   IterativeSolverResult result{};
   size_t allocation_count = 0;
   {
      gmres_allocation_probe::Guard guard;
      result = FGMRES<HostKernelConfiguration>(
         map, inverse, rhs, solution, outer_dot, outer_options, outer_workspace, orthogonalization);
      allocation_count = gmres_allocation_probe::calls.load(std::memory_order_relaxed);
   }
   CHECK(result.success && Near(solution, {1, 1, 1}));
   CHECK(inverse.calls == result.iterations && inverse.calls > 1);
   CHECK(inverse.inner_iterations == inverse.calls && inverse.valid_inner_results);
   CHECK(allocation_count == 0);
   {
      auto inner_view = MakeGMRESWorkspaceView(inner_workspace);
      auto outer_view = MakeGMRESWorkspaceView(outer_workspace);
      CHECK(ReadHostVector(inner_view.basis(0)) == inner_basis);
      CHECK(ReadHostVector(outer_view.basis(0)) == outer_basis);
   }
}

struct OrthogonalizationTrace
{
   std::array<char, 32> events{};
   std::array<std::array<Real, 4>, 6> projected_inputs{};
   size_t event_count = 0;
   size_t dot_count = 0;

   void Record(const char event)
   {
      if (event_count < events.size()) { events[event_count] = event; }
      ++event_count;
   }

   template<class First, class Second>
   Real Dot(const First& first, const Second& second)
   {
      Record('d');
      Real result = 0;
      for (size_t i = 0; i < 4; ++i)
      {
         if (dot_count < projected_inputs.size())
         {
            projected_inputs[dot_count][i] = second[i];
         }
         result += first[i] * second[i];
      }
      ++dot_count;
      return result;
   }

   template<class Input, class Output>
   void Axpy(const Real coefficient, const Input& input, Output& output)
   {
      Record('a');
      for (size_t i = 0; i < 4; ++i) { output[i] += coefficient * input[i]; }
   }

   template<class Basis, class Input, class Coefficients>
   void Project(const Basis& basis, const Integer count, const Input& input,
                Coefficients& coefficients)
   {
      Record('P');
      for (Integer j = 0; j < count; ++j) { coefficients[j] = Dot(basis(j), input); }
   }

   template<class Basis, class Coefficients, class Output>
   void Combine(const Basis& basis, const Integer count,
                const Coefficients& coefficients, const Real scale, Output& output)
   {
      Record('C');
      for (Integer j = 0; j < count; ++j)
      {
         Axpy(scale * coefficients[j], basis(j), output);
      }
   }
};

template<class Orthogonalization>
void BasisOrthogonality(const Orthogonalization& orthogonalization)
{
   std::array<std::array<Real, 4>, 4> basis{{
      {{0.5, 0.5, 0.5, 0.5}}, {{0.5, -0.5, 0.5, -0.5}},
      {{0.5, 0.5, -0.5, -0.5}}, {{0, 0, 0, 0}}}};
   const std::array<Real, 4> last_direction{0.5, -0.5, -0.5, 0.5};
   const std::array<Real, 3> expected_projections{1.2, -2.3, 0.7};
   std::array<Real, 4> work{};
   for (size_t i = 0; i < 4; ++i)
   {
      work[i] = 0.1 * last_direction[i];
      for (size_t j = 0; j < 3; ++j)
      {
         work[i] += expected_projections[j] * basis[j][i];
      }
   }
   const auto original_work = work;
   std::array<Real, 25> dense{};
   auto basis_view = MakeView(
      ReferenceContainer{basis}, VectorLayout{basis.size()});
   NoDirections directions;
   auto dense_view = MakeVectorView(dense.data(), dense.size());
   auto packed_dense = MakeGMRESPackedDense(dense_view, 3, 4);
   auto workspace = TieGMRESBuffers(
      basis_view, work, directions, packed_dense.hessenberg,
      packed_dense.cosines, packed_dense.sines, packed_dense.reduced_rhs,
      packed_dense.coefficients);
   OrthogonalizationTrace operations;
   auto vectors = [&](Integer j) -> auto& { return basis[j]; };
   auto h = gmres_detail::MakeHessenbergColumnView(workspace.hessenberg, 2);
   auto& corrections = workspace.coefficients;
   if constexpr (std::is_same_v<Orthogonalization, ClassicalGramSchmidtTwoPass>)
   {
      auto project = [&](auto&& vectors, Integer count, const auto& input, auto& c)
      { operations.Project(vectors, count, input, c); };
      auto combine = [&](auto&& vectors, Integer count, const auto& c,
                         Real alpha, auto& output)
      { operations.Combine(vectors, count, c, alpha, output); };
      auto accumulate = [](auto& destination, const auto& source, Integer count)
      { for (Integer j = 0; j < count; ++j) { destination[j] += source[j]; } };
      CHECK(gmres_detail::OrthogonalizeClassicalGramSchmidtTwoPass(
         project, combine, accumulate, vectors, Integer(3), work, h, corrections));
      const std::array<char, 16> expected_events{
         'P', 'd', 'd', 'd', 'C', 'a', 'a', 'a',
         'P', 'd', 'd', 'd', 'C', 'a', 'a', 'a'};
      CHECK(operations.event_count == expected_events.size());
      for (size_t i = 0; i < expected_events.size(); ++i)
      {
         CHECK(operations.events[i] == expected_events[i]);
      }
      // Each first-pass dot must observe the original, unmodified work vector.
      for (size_t j = 0; j < 3; ++j)
      {
         CHECK(operations.projected_inputs[j] == original_work);
      }
   }
   else
   {
      auto dot = [&](const auto& first, const auto& second)
      { return operations.Dot(first, second); };
      auto axpy = [&](Real alpha, const auto& input, auto& output)
      { operations.Axpy(alpha, input, output); };
      auto store = [](auto& coefficients, Integer j, Real value)
      { coefficients[j] = value; };
      CHECK(gmres_detail::OrthogonalizeModifiedGramSchmidt(
         dot, axpy, store, vectors, Integer(3), work, h));
      const std::array<char, 6> expected_events{'d', 'a', 'd', 'a', 'd', 'a'};
      CHECK(operations.event_count == expected_events.size());
      for (size_t i = 0; i < expected_events.size(); ++i)
      {
         CHECK(operations.events[i] == expected_events[i]);
      }
   }
   for (size_t j = 0; j < 3; ++j)
   {
      CHECK(Near(workspace.hessenberg(j, 2), expected_projections[j], 1.e-13));
   }
   Real norm_square = 0;
   for (const Real value : work) { norm_square += value * value; }
   const Real norm = std::sqrt(norm_square);
   CHECK(Near(norm, 0.1, 1.e-13));
   for (size_t i = 0; i < 4; ++i) { basis[3][i] = work[i] / norm; }
   // The complete basis, including the newly normalized residual, is orthonormal.
   for (size_t i = 0; i < 4; ++i)
   {
      for (size_t j = 0; j < 4; ++j)
      {
         Real product = 0;
         for (size_t k = 0; k < 4; ++k) { product += basis[i][k] * basis[j][k]; }
         CHECK(Near(product, i == j ? 1.0 : 0.0, 1.e-13));
      }
   }
}

void WorkspaceGrowth()
{
   GMRESOptions options{2, 20, 1.e-12, 0};
   auto workspace = MakeGMRESWorkspace(3, 2);
   EuclideanDot<HostKernelConfiguration> product{workspace.reduction_workspace};
   const ModifiedGramSchmidt orthogonalization{};

   Vector rhs(3), solution(3);
   rhs = 1;
   solution = 0;
   auto result = GMRES<HostKernelConfiguration>(
      ScaledIdentity{1, 3}, rhs, solution, product, options, workspace, orthogonalization);
   CHECK(result.success && Near(solution, {1, 1, 1}));

   options.restart = 4;
   Vector grown_rhs(5), grown_solution(5);
   grown_rhs = 1;
   grown_solution = 0;
   result = GMRES<HostKernelConfiguration>(
      ScaledIdentity{1, 5}, grown_rhs, grown_solution, product, options, workspace, orthogonalization);
   CHECK(result.success && workspace.vector_size == 5 &&
      workspace.restart_capacity >= 4);
   for (size_t i = 0; i < 5; ++i)
   { CHECK(Near(grown_solution.ReadHostData()[i], 1)); }

   grown_solution = 0;
   result = GMRES<HostKernelConfiguration>(
      ScaledIdentity{1, 5}, ScaledIdentity{1, 5}, grown_rhs, grown_solution, product, options, workspace,
      orthogonalization);
   CHECK(result.success && workspace.direction_capacity >= 1);

   grown_solution = 0;
   result = FGMRES<HostKernelConfiguration>(
      ScaledIdentity{1, 5}, ScaledIdentity{1, 5}, grown_rhs, grown_solution, product, options, workspace,
      orthogonalization);
   CHECK(result.success && workspace.direction_capacity >= 4);
   for (size_t i = 0; i < 5; ++i)
   { CHECK(Near(grown_solution.ReadHostData()[i], 1)); }
}

struct NoncopyableVector
{
   std::array<Real, 3> values{};
   size_t size = 3;
   NoncopyableVector() = default;
   NoncopyableVector(const NoncopyableVector&) = delete;
   NoncopyableVector& operator=(const NoncopyableVector&) = delete;
};

void BorrowedWorkspace()
{
   {
      GMRESWorkspace initial;
      CHECK(initial.reduction_workspace.Size() == 0);
      PrepareGMRESAlgorithmWorkspace(initial, 4, 3, 2);
      auto prepared = MakeGMRESWorkspaceView(initial);
      CHECK(initial.vector_size == 4 && initial.restart_capacity == 3 &&
            initial.direction_capacity == 2);
      CHECK(GetExtentValue<1>(prepared.hessenberg) == 3 &&
            prepared.basis.Size() == 4 && prepared.cosines.Size() == 3);
      auto second_basis = prepared.basis(1);
      WriteHostVector(second_basis)[0] = 7;
      CHECK(ReadHostVector(initial.basis_storage)[4] == 7);
   }

   std::array<NoncopyableVector, 4> basis;
   std::array<NoncopyableVector, 3> directions;
   NoncopyableVector work;
   std::array<Real, 31> dense{};
   auto basis_view = MakeView(
      ReferenceContainer{basis}, VectorLayout{basis.size()});
   auto direction_view = MakeView(
      ReferenceContainer{directions}, VectorLayout{directions.size()});
   auto dense_view = MakeVectorView(dense.data(), dense.size());
   auto packed_dense = MakeGMRESPackedDense(dense_view, 3, 6);
   auto view = TieGMRESBuffers(
      basis_view, work, direction_view, packed_dense.hessenberg,
      packed_dense.cosines, packed_dense.sines, packed_dense.reduced_rhs,
      packed_dense.coefficients);
   static_assert(std::is_same_v<decltype(view.basis(0)), NoncopyableVector&>);
   static_assert(std::is_same_v<decltype(view.directions(0)), NoncopyableVector&>);
   static_assert(std::is_same_v<decltype(view.work), NoncopyableVector&>);
   static_assert(std::is_same_v<decltype(view.hessenberg(0, 0)), Real&>);
   auto copy = view;
   copy.basis(2).values[1] = 17;
   copy.directions(1).values[0] = 19;
   copy.work.values[2] = 23;
   copy.hessenberg(1, 2) = 29;
   copy.coefficients(2) = 31;
   CHECK(basis[2].values[1] == 17 && directions[1].values[0] == 19);
   CHECK(work.values[2] == 23 && basis[2].size == 3 && work.size == 3);
   CHECK(dense[13] == 29 && dense[30] == 31);
   CHECK(GetStride<1>(view.hessenberg.layout) == 6 &&
         GetVectorSize(view.directions) == 3);
   Real h[12]{}, c[3]{}, s[3]{}, g[4]{}, y[3]{};
   auto h_view = MakeView(
      PointerContainer<Real>{h},
      StridedLayout<2, DynamicExtents<2>>{
         std::array<size_t, 2>{1, 4}, DynamicExtents<2>{4, 3}});
   auto cosine_view = MakeVectorView(c + 0, 3);
   auto sine_view = MakeVectorView(s + 0, 3);
   auto rhs_view = MakeVectorView(g + 0, 4);
   auto coefficient_view = MakeVectorView(y + 0, 3);
   NoDirections no_directions;
   auto separate = TieGMRESBuffers(
      basis_view, work, no_directions, h_view, cosine_view, sine_view,
      rhs_view, coefficient_view);
   separate.hessenberg(2, 2) = 37;
   CHECK(h[10] == 37);
   Real* pointer = h;
   auto pointer_h_view = MakeView(
      PointerContainer<Real>{pointer},
      StridedLayout<2, DynamicExtents<2>>{
         std::array<size_t, 2>{1, 4}, DynamicExtents<2>{4, 3}});
   auto pointers = TieGMRESBuffers(
      basis_view, work, no_directions, pointer_h_view, cosine_view, sine_view,
      rhs_view, coefficient_view);
   pointer = nullptr;
   pointers.hessenberg(0, 0) = 41;
   CHECK(h[0] == 41);
   auto separate_directions = TieGMRESBuffers(
      basis_view, work, direction_view, h_view, cosine_view, sine_view,
      rhs_view, coefficient_view);
   separate_directions.directions(2).values[0] = 43;
   CHECK(directions[2].values[0] == 43);
   auto contiguous_dense_view = MakeVectorView(dense.data(), dense.size());
   auto contiguous_dense = MakeGMRESPackedDense(
      contiguous_dense_view, 3, 4);
   auto contiguous = TieGMRESBuffers(
      basis_view, work, no_directions, contiguous_dense.hessenberg,
      contiguous_dense.cosines, contiguous_dense.sines,
      contiguous_dense.reduced_rhs, contiguous_dense.coefficients);
   contiguous.cosines(0) = 47;
   CHECK(dense[12] == 47);

   static_assert(std::is_aggregate_v<GMRESWorkspace>);
   static_assert(std::is_aggregate_v<decltype(contiguous)>);
   auto owner = MakeGMRESWorkspace(3, 3, 1);
   const Real* base = nullptr;
   {
      auto owned = MakeGMRESWorkspaceView(owner);
      auto slice = owned.basis(1);
      WriteHostVector(slice)[0] = 53;
      CHECK(ReadHostVector(owned.basis(1))[0] == 53);
      CHECK(&owned.work == &owner.work_vector);
      base = ReadHostVector(owned.basis(0));
   }
   PrepareGMRESAlgorithmWorkspace(owner, 3, 2);
   CHECK(owner.restart_capacity == 3);
   {
      auto retained = MakeGMRESWorkspaceView(owner);
      CHECK(GetStride<1>(retained.hessenberg.layout) == 4);
      CHECK(ReadHostVector(retained.basis(0)) == base);
   }
   CHECK(owner.direction_capacity == 1);
   PrepareGMRESAlgorithmWorkspace(owner, 3, 3, 3);
   CHECK(owner.direction_capacity == 3);
   {
      auto new_view = MakeGMRESWorkspaceView(owner);
      CHECK(ReadHostVector(new_view.basis(0)) == base);
      auto direction = new_view.directions(2);
      WriteHostVector(direction)[0] = 59;
      CHECK(ReadHostVector(new_view.directions(2))[0] == 59);
   }

   PrepareGMRESAlgorithmWorkspace(owner, 4, 2, 1);
   CHECK(owner.vector_size == 4 && owner.restart_capacity == 2);
   CHECK(owner.direction_capacity == 1);
   CHECK(owner.basis_storage.Size() == 12 && owner.work_vector.Size() == 4);
   CHECK(owner.preconditioned_direction_storage.Size() == 4);
   PrepareGMRESAlgorithmWorkspace(owner, 4, 4, 2);
   owner.basis_storage = 0.0;
   owner.preconditioned_direction_storage = 0.0;
   const Real* grown_basis = ReadHostVector(owner.basis_storage);
   const Real* grown_directions =
      ReadHostVector(owner.preconditioned_direction_storage);
   const Real* grown_dense = owner.dense_workspace.data();
   PrepareGMRESAlgorithmWorkspace(owner, 4, 2, 1);
   CHECK(owner.restart_capacity == 4 && owner.direction_capacity == 2);
   CHECK(ReadHostVector(owner.basis_storage) == grown_basis);
   CHECK(ReadHostVector(owner.preconditioned_direction_storage) ==
      grown_directions);
   CHECK(owner.dense_workspace.data() == grown_dense);
}

} // namespace

int main()
{
   const GMRESOptions defaults;
   CHECK(defaults.restart == 30 && defaults.max_iterations == 300);
   CHECK(defaults.relative_tolerance == std::sqrt(std::numeric_limits<Real>::epsilon()));
   CHECK(defaults.absolute_tolerance == 0);
   Solves(ModifiedGramSchmidt{});
   Solves(ClassicalGramSchmidtTwoPass{});
   EdgeCases(ModifiedGramSchmidt{});
   EdgeCases(ClassicalGramSchmidtTwoPass{});
   InnerSolverPreconditioning(ModifiedGramSchmidt{});
   InnerSolverPreconditioning(ClassicalGramSchmidtTwoPass{});
   BasisOrthogonality(ModifiedGramSchmidt{});
   BasisOrthogonality(ClassicalGramSchmidtTwoPass{});
   local_protocol::CheckDispatch<0>(ModifiedGramSchmidt{});
   local_protocol::CheckDispatch<0>(ClassicalGramSchmidtTwoPass{});
   local_protocol::CheckDispatch<1>(ModifiedGramSchmidt{});
   local_protocol::CheckDispatch<1>(ClassicalGramSchmidtTwoPass{});
   local_protocol::CheckDispatch<2>(ModifiedGramSchmidt{});
   local_protocol::CheckDispatch<2>(ClassicalGramSchmidtTwoPass{});
   NonsymmetricRestarted(ModifiedGramSchmidt{});
   NonsymmetricRestarted(ClassicalGramSchmidtTwoPass{});
   Norms();
   ExplicitOperationSelection();
   WorkspaceGrowth();
   BorrowedWorkspace();
   return failures == 0 ? 0 : 1;
}

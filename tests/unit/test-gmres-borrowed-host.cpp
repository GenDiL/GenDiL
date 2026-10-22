// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#ifdef GENDIL_TEST_FORCE_DEBUG
#undef NDEBUG
#endif
#include <gendil/Solvers/GMRES/gmres.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

using namespace gendil;

namespace {
using Slice = decltype(MakeVectorView(std::declval<Vector&>(), 0, 3));
int failures = 0;
void Check(bool value) { if (!value) { ++failures; } }

using BindingBasis = std::array<std::array<Real, 2>, 3>;
using BindingBasisView = decltype(MakeView(
   ReferenceContainer{std::declval<BindingBasis&>()}, VectorLayout{3}));
using BindingScalarView = decltype(MakeVectorView(static_cast<Real*>(nullptr), 2));
using BindingHView = View<PointerContainer<Real>, StridedLayout<2, DynamicExtents<2>>>;

template<size_t Index, size_t Temporary, bool ConstTemporary = false, class T>
decltype(auto) BindingArgument(T& value)
{
   if constexpr (Index == Temporary && ConstTemporary)
   { return std::move(std::as_const(value)); }
   else if constexpr (Index == Temporary) { return std::move(value); }
   else { return (value); }
}

// Even shallow descriptors must be named lvalues when tied into buffers.
template<size_t Temporary, bool ConstTemporary = false>
concept CanTieBuffers = requires(BindingBasisView& basis, BindingScalarView& work,
   NoDirections& directions, BindingHView& h, BindingScalarView& c,
   BindingScalarView& s, BindingScalarView& g, BindingScalarView& y)
{
   TieGMRESBuffers(BindingArgument<0, Temporary, ConstTemporary>(basis),
      BindingArgument<1, Temporary, ConstTemporary>(work),
      BindingArgument<2, Temporary, ConstTemporary>(directions),
      BindingArgument<3, Temporary, ConstTemporary>(h),
      BindingArgument<4, Temporary, ConstTemporary>(c),
      BindingArgument<5, Temporary, ConstTemporary>(s),
      BindingArgument<6, Temporary, ConstTemporary>(g),
      BindingArgument<7, Temporary, ConstTemporary>(y));
};
static_assert(CanTieBuffers<8>);
static_assert(!CanTieBuffers<0> && !CanTieBuffers<1> &&
              !CanTieBuffers<2> && !CanTieBuffers<3>);
static_assert(!CanTieBuffers<4> && !CanTieBuffers<5> &&
              !CanTieBuffers<6> && !CanTieBuffers<7>);
static_assert(!CanTieBuffers<0, true> && !CanTieBuffers<1, true> &&
              !CanTieBuffers<2, true> && !CanTieBuffers<3, true>);
static_assert(!CanTieBuffers<4, true> && !CanTieBuffers<5, true> &&
              !CanTieBuffers<6, true> && !CanTieBuffers<7, true>);

void CheckPackedBufferBinding()
{
   BindingBasis basis_storage{};
   auto basis = MakeView(ReferenceContainer{basis_storage}, VectorLayout{3});
   std::array<Real, 2> work{};
   std::array<Real, 15> storage{};
   auto parent = MakeVectorView(storage.data(), storage.size());
   auto dense = MakeGMRESPackedDense(parent, 2, 3);
   NoDirections directions;
   auto buffers = TieGMRESBuffers(basis, work, directions, dense.hessenberg,
      dense.cosines, dense.sines, dense.reduced_rhs, dense.coefficients);
   buffers.basis(1)[0] = 3;
   buffers.hessenberg(2, 1) = 5;
   buffers.cosines[1] = 7;
   buffers.sines[1] = 11;
   buffers.reduced_rhs[2] = 13;
   buffers.coefficients[1] = 17;
   Check(basis_storage[1][0] == 3 && storage[5] == 5 && storage[7] == 7 &&
         storage[9] == 11 && storage[12] == 13 && storage[14] == 17);
}

struct NoncopyableWork
{
   Real values[2]{};
   NoncopyableWork() = default;
   NoncopyableWork(const NoncopyableWork&) = delete;
   NoncopyableWork& operator=(const NoncopyableWork&) = delete;
};

void CheckBufferBindingDoesNotCopy()
{
   std::array<NoncopyableWork, 3> basis_storage;
   NoncopyableWork work;
   std::array<NoncopyableWork, 2> direction_storage;
   auto basis = MakeView(ReferenceContainer{basis_storage}, VectorLayout{3});
   auto directions = MakeView(ReferenceContainer{direction_storage}, VectorLayout{2});
   std::array<Real, 15> dense_storage{};
   auto parent = MakeVectorView(dense_storage.data(), dense_storage.size());
   auto dense = MakeGMRESPackedDense(parent, 2, 3);
   auto buffers = TieGMRESBuffers(basis, work, directions, dense.hessenberg,
      dense.cosines, dense.sines, dense.reduced_rhs, dense.coefficients);
   static_assert(std::is_aggregate_v<decltype(buffers)>);
   static_assert(std::is_trivially_copyable_v<decltype(buffers)>);
   static_assert(std::is_same_v<decltype(buffers.basis), decltype(basis)&>);
   static_assert(std::is_same_v<decltype(buffers.work), NoncopyableWork&>);
   static_assert(std::is_same_v<decltype(buffers.directions), decltype(directions)&>);
   static_assert(std::is_reference_v<decltype(buffers.hessenberg)>);
   static_assert(std::is_reference_v<decltype(buffers.cosines)>);
   static_assert(std::is_reference_v<decltype(buffers.sines)>);
   static_assert(std::is_reference_v<decltype(buffers.reduced_rhs)>);
   static_assert(std::is_reference_v<decltype(buffers.coefficients)>);
   static_assert(std::is_same_v<decltype(buffers.basis(0)), NoncopyableWork&>);
   auto copied = buffers;
   copied.basis(1).values[0] = 53;
   copied.directions(1).values[1] = 59;
   copied.work.values[0] = 61;
   Check(basis_storage[1].values[0] == 53 &&
         direction_storage[1].values[1] == 59 && work.values[0] == 61);

   using InlineView = View<StaticContainer<Real, 2>, VectorLayout>;
   InlineView inline_work{StaticContainer<Real, 2>{}, VectorLayout{2}};
   NoDirections none;
   auto inline_buffers = TieGMRESBuffers(basis, inline_work, none,
      dense.hessenberg, dense.cosines, dense.sines, dense.reduced_rhs,
      dense.coefficients);
   static_assert(std::is_same_v<decltype(inline_buffers.work), InlineView&>);
   inline_buffers.work[1] = 67;
   Check(inline_work[1] == 67);
}

auto ReturnedPackedDense(Real* storage)
{
   // Retain the parent's origin and scalar stride after its descriptor dies.
   auto parent = MakeView(PointerContainer<Real>{storage}, MakeOffsetLayout(
      StridedLayout<1, DynamicExtents<1>>{
         std::array<size_t, 1>{3}, DynamicExtents<1>{19}}, 5));
   return MakeGMRESPackedDense(parent, 2, 5);
}

void CheckPackedDenseLifetime()
{
   std::array<Real, 70> storage;
   storage.fill(-91);
   auto dense = ReturnedPackedDense(storage.data());
   static_assert(!std::is_reference_v<decltype(dense.hessenberg)>);
   auto copied = dense;
   copied.hessenberg(2, 1) = 71;
   copied.cosines[1] = 73;
   copied.sines[1] = 79;
   copied.reduced_rhs[2] = 83;
   copied.coefficients[1] = 89;
   Check(GetExtentValue<0>(copied.hessenberg) == 3 &&
         GetExtentValue<1>(copied.hessenberg) == 2);
   for (size_t i = 0; i < storage.size(); ++i)
   {
      const Real expected = i == 26 ? 71 : i == 38 ? 73 : i == 44 ? 79 :
                            i == 53 ? 83 : i == 59 ? 89 : -91;
      Check(storage[i] == expected);
   }
}

// Invalid metadata must be rejected before even requesting writable solution
// storage, as well as before invoking a numerical map or preconditioner.
struct GuardedSolution { Slice slice; };
size_t GetVectorSize(const GuardedSolution&) { return 3; }
const Real* ReadHostVector(const GuardedSolution& value)
{ return gendil::ReadHostVector(value.slice); }
Real* WriteHostVector(GuardedSolution&)
{
   std::fputs("Unexpected solution mutation\n", stderr);
   std::exit(7);
}
Real* ReadWriteHostVector(GuardedSolution& value) { return WriteHostVector(value); }

// Deliberately supports native slices, not owning Vector inputs or temporaries.
struct SliceMap
{
   bool forbid_callbacks = false;
   size_t InputSize() const { return 3; }
   size_t OutputSize() const { return 3; }
   void operator()(const Slice& input, Slice& output) const
   {
      if (forbid_callbacks) { std::fputs("Unexpected numerical callback\n", stderr); std::exit(5); }
      const auto* x = ReadHostVector(input);
      auto* y = WriteHostVector(output);
      for (size_t i = 0; i < 3; ++i) { y[i] = 2 * x[i]; }
   }
   void operator()(Slice&&, Slice&) const = delete;
   void operator()(const GuardedSolution& input, Slice& output) const
   { (*this)(input.slice, output); }
   void ApplyAdd(const Slice& input, Slice& output) const
   {
      if (forbid_callbacks) { std::fputs("Unexpected numerical callback\n", stderr); std::exit(5); }
      const auto* x = ReadHostVector(input);
      auto* y = ReadWriteHostVector(output);
      for (size_t i = 0; i < 3; ++i) { y[i] += 2 * x[i]; }
   }
   void ApplyAdd(const GuardedSolution& input, Slice& output) const
   { ApplyAdd(input.slice, output); }
};

// A direct temporary call is unavailable; a bound slice is a valid operand.
void Apply(const SliceMap& map, const Slice& input, Slice& output)
{ map(input, output); }
void Apply(const SliceMap&, Slice&&, Slice&) = delete;

struct Inverse
{
   bool forbid_callbacks = false;
   template<class Input>
   void operator()(const Input& input, Slice& output) const
   {
      if (forbid_callbacks) { std::fputs("Unexpected numerical callback\n", stderr); std::exit(5); }
      const auto* x = ReadHostVector(input);
      auto* y = WriteHostVector(output);
      for (size_t i = 0; i < 3; ++i) { y[i] = x[i] / 2; }
   }
};

using HostDot = EuclideanDot<HostKernelConfiguration>;
template<class Buffers>
concept CanUnprecondition = requires(Buffers& buffers, HostDot& dot,
   const GMRESOptions& options, Vector& rhs, Slice& x)
{ GMRES<HostKernelConfiguration>(SliceMap{}, rhs, x, dot, options, buffers); };
template<class Buffers>
concept CanPrecondition = requires(Buffers& buffers, HostDot& dot,
   const GMRESOptions& options, Vector& rhs, Slice& x)
{
   GMRES<HostKernelConfiguration>(SliceMap{}, Inverse{}, rhs, x, dot, options, buffers);
   FGMRES<HostKernelConfiguration>(SliceMap{}, Inverse{}, rhs, x, dot, options, buffers);
};

auto ReturnedBuffers(Vector& owner, std::array<Real, 31>& dense)
{
   // This concrete adapter explicitly owns its known native descriptors by
   // value. It does not return references to helper-local descriptors.
   auto parent = MakeVectorView(owner, 2, 21);
   auto basis = MakeView(StaticContainer<Slice, 4>{{
      MakeVectorView(parent, 1, 3), MakeVectorView(parent, 4, 3),
      MakeVectorView(parent, 7, 3), MakeVectorView(parent, 10, 3)}}, VectorLayout{4});
   auto work = MakeVectorView(parent, 13, 3);
   auto directions = MakeView(StaticContainer<Slice, 1>{{
      MakeVectorView(parent, 16, 3)}}, VectorLayout{1});
   auto scalar_storage = MakeVectorView(dense.data(), dense.size());
   auto packed = MakeGMRESPackedDense(scalar_storage, 3, 6);
   return GMRESWorkspaceView<decltype(basis), decltype(work), decltype(directions),
      decltype(packed.hessenberg), decltype(packed.cosines), decltype(packed.sines),
      decltype(packed.reduced_rhs), decltype(packed.coefficients)>{
      basis, work, directions, packed.hessenberg, packed.cosines, packed.sines,
      packed.reduced_rhs, packed.coefficients};
}

void LifetimesAndBulkOrigin()
{
   Vector owner(26);
   owner = -17;
   std::array<Real, 31> dense{};
   auto buffers = ReturnedBuffers(owner, dense);
   static_assert(!std::is_reference_v<decltype(buffers.work)>);
   for (size_t j = 0; j < 4; ++j)
   {
      auto vector = buffers.basis(j);
      auto* values = WriteHostVector(vector);
      for (size_t i = 0; i < 3; ++i) { values[i] = Real(3 * j + i + 1); }
   }
   Vector input(3), output(3), reduction;
   input = 1; output = 0;
   // Unlike the returned explicit value adapter, VectorSlices borrows its
   // named parent descriptor. Keep that descriptor alive for both bulk calls.
   auto basis_region = MakeVectorView(owner, 3, 12);
   auto bulk_basis = MakeVectorSlices(basis_region, 3, 4);
   EuclideanDot<HostKernelConfiguration> dot{reduction};
   std::array<Real, 2> projections{};
   ProjectEuclidean(dot, bulk_basis, 2, input, projections);
   Check(projections[0] == 6 && projections[1] == 15);
   const std::array<Real, 2> coefficients{1, 2};
   HostDrivenFusedCombine<false>(bulk_basis, 2, coefficients, 1, output);
   const auto* result = output.ReadHostData();
   Check(result[0] == 9 && result[1] == 12 && result[2] == 15);
   WriteHostVector(buffers.work)[0] = 41;
   auto direction = buffers.directions(0);
   WriteHostVector(direction)[2] = 43;
   const auto* values = owner.ReadHostData();
   Check(values[15] == 41 && values[20] == 43);
   for (size_t i : {size_t{0}, size_t{1}, size_t{2}, size_t{21}, size_t{25}})
   { Check(values[i] == -17); }
   for (size_t j = 0; j < 4; ++j)
      for (size_t i = 0; i < 3; ++i)
      { Check(values[3 + 3 * j + i] == Real(3 * j + i + 1)); }
}

template<class Orthogonalization>
void SolveBorrowed(Orthogonalization orthogonalization)
{
   Vector owner(26), rhs(3), solution_storage(7), reduction;
   owner = -17; rhs = 2; solution_storage = -19;
   auto solution = MakeVectorView(solution_storage, 2, 3);
   Zero<false>(solution);
   std::array<Real, 31> dense{};
   auto buffers = ReturnedBuffers(owner, dense);
   GMRESOptions options{2, 20, 1e-12, 0};
   EuclideanDot<HostKernelConfiguration> dot{reduction};
   static_assert(CanUnprecondition<decltype(buffers)>);
   auto result = GMRES<HostKernelConfiguration>(
      SliceMap{}, rhs, solution, dot, options, buffers, orthogonalization);
   Check(result.success && result.iterations == 1);
   Check(reduction.Size() == 0);
   for (size_t i = 0; i < 3; ++i) { Check(std::abs(ReadHostVector(solution)[i] - 1) < 1e-12); }
   Check(solution_storage.ReadHostData()[0] == -19 && solution_storage.ReadHostData()[6] == -19);

   // Basis entries are now owning vectors, which SliceMap intentionally rejects.
   std::array<Vector, 4> basis{Vector(3), Vector(3), Vector(3), Vector(3)};
   Vector direction_owner(6), work_owner(3);
   auto directions = MakeVectorSlices(direction_owner, 3, 2);
   auto work = MakeVectorView(work_owner, 0, 3);
   auto basis_view = MakeView(ReferenceContainer{basis}, VectorLayout{4});
   auto scalar_storage = MakeVectorView(dense.data(), dense.size());
   auto packed = MakeGMRESPackedDense(scalar_storage, 3, 6);
   auto heterogeneous = TieGMRESBuffers(basis_view, work, directions,
      packed.hessenberg, packed.cosines, packed.sines, packed.reduced_rhs,
      packed.coefficients);
   static_assert(!CanUnprecondition<decltype(heterogeneous)>);
   static_assert(CanPrecondition<decltype(heterogeneous)>);
   Zero<false>(solution);
   result = GMRES<HostKernelConfiguration>(
      SliceMap{}, Inverse{}, rhs, solution, dot, options, heterogeneous, orthogonalization);
   Check(result.success && result.iterations == 1);
   Zero<false>(solution);
   result = FGMRES<HostKernelConfiguration>(
      SliceMap{}, Inverse{}, rhs, solution, dot, options, heterogeneous, orthogonalization);
   Check(result.success && result.iterations == 1);
}

struct DiagonalMap
{
   size_t size;
   size_t InputSize() const { return size; }
   size_t OutputSize() const { return size; }
   template<class Input, class Output>
   void operator()(const Input& input, Output& output) const
   {
      const auto* x = ReadHostVector(input);
      auto* y = WriteHostVector(output);
      for (size_t i = 0; i < size; ++i) { y[i] = Real(i % 3 + 1) * x[i]; }
   }
   template<class Input, class Output>
   void ApplyAdd(const Input& input, Output& output) const
   {
      const auto* x = ReadHostVector(input);
      auto* y = ReadWriteHostVector(output);
      for (size_t i = 0; i < size; ++i) { y[i] += Real(i % 3 + 1) * x[i]; }
   }
};

template<class Orthogonalization>
void UnequalCapacitiesAndNativeStrides(Orthogonalization orthogonalization)
{
   constexpr size_t n = 10, active = 8, h_columns = 32, h_rows = 33;
   constexpr size_t row_stride = 2, column_stride = 70;
   const Real poison = std::numeric_limits<Real>::quiet_NaN();
   Vector basis_owner((active + 1) * n), work(n), rhs(n), solution(n), reduction;
   basis_owner = poison; work = poison; solution = 0;
   auto basis = MakeVectorSlices(basis_owner, n, active + 1);
   NoDirections directions;
   std::array<Real, column_stride * h_columns> h_storage;
   h_storage.fill(poison);
   auto h = MakeView(PointerContainer<Real>{h_storage.data()},
      StridedLayout<2, DynamicExtents<2>>{
         std::array<size_t, 2>{row_stride, column_stride},
         DynamicExtents<2>{h_rows, h_columns}});
   // These really are smaller allocations, not capacity-32 arrays with short
   // descriptors. No shared reservation can be inferred from H's column count.
   std::array<Real, active> c, s, y;
   std::array<Real, active + 1> g;
   c.fill(poison); s.fill(poison); y.fill(poison); g.fill(poison);
   auto cosines = MakeVectorView(c.data(), c.size());
   auto sines = MakeVectorView(s.data(), s.size());
   auto reduced_rhs = MakeVectorView(g.data(), g.size());
   auto coefficients = MakeVectorView(y.data(), y.size());
   auto buffers = TieGMRESBuffers(basis, work, directions, h, cosines, sines,
      reduced_rhs, coefficients);
   auto* b = rhs.WriteHostData();
   for (size_t i = 0; i < n; ++i) { b[i] = Real(i % 3 + 1); }
   const GMRESOptions options{active, 20, 1e-12, 0};
   EuclideanDot<HostKernelConfiguration> dot{reduction};
   const auto result = GMRES<HostKernelConfiguration>(
      DiagonalMap{n}, rhs, solution, dot, options, buffers, orthogonalization);
   Check(result.success && result.iterations == 3);
   Check(basis.Size() == active + 1 && GetExtentValue<1>(h) == h_columns);
   for (size_t i = 0; i < n; ++i)
   { Check(std::abs(solution.ReadHostData()[i] - 1) < 1e-11); }
   for (size_t j = 0; j < h_columns; ++j)
      for (size_t offset = 0; offset < column_stride; ++offset)
      {
         const bool used = j < result.iterations && offset % row_stride == 0 &&
                           offset / row_stride <= j + 1;
         Check(used ? std::isfinite(h_storage[j * column_stride + offset]) :
                      std::isnan(h_storage[j * column_stride + offset]));
      }
   for (size_t j = result.iterations; j <= active; ++j)
      for (size_t i = 0; i < n; ++i)
      { Check(std::isnan(basis_owner.ReadHostData()[j * n + i])); }
}

void Reject(std::string_view mode)
{
   Vector owner(26), rhs(3), solution_owner(3), reduction;
   owner = 9; rhs = 2; solution_owner = 31;
   GuardedSolution solution{MakeVectorView(solution_owner, 0, 3)};
   std::array<Real, 31> dense{};
   auto buffers = ReturnedBuffers(owner, dense);
   GMRESOptions options{2, 20, 1e-12, 0};
   if (mode == "short-rotations") { buffers.cosines.layout.parent.extent = 1; }
   else if (mode == "short-sines") { buffers.sines.layout.parent.extent = 1; }
   else if (mode == "short-rhs") { buffers.reduced_rhs.layout.parent.extent = 2; }
   else if (mode == "short-coefficients") { buffers.coefficients.layout.parent.extent = 1; }
   else if (mode == "short-basis") { buffers.basis.layout.extent = 1; }
   else if (mode == "short-work") { buffers.work.layout.parent.extent = 2; }
   else if (mode == "short-Hrows")
   { buffers.hessenberg.layout.extents = DynamicExtents<2>{2, 3}; }
   else if (mode == "restart")
   { buffers.hessenberg.layout.extents = DynamicExtents<2>{4, 1}; }
   EuclideanDot<HostKernelConfiguration> dot{reduction};
   if (mode == "short-directions") { (void)FGMRES<HostKernelConfiguration>(
      SliceMap{true}, Inverse{true}, rhs, solution, dot, options, buffers); }
   else { (void)GMRES<HostKernelConfiguration>(SliceMap{true}, rhs, solution, dot, options, buffers); }
   std::exit(6);
}
} // namespace

int main(int argc, char** argv)
{
   if (argc > 1) { Reject(argv[1]); }
   CheckPackedBufferBinding();
   CheckBufferBindingDoesNotCopy();
   CheckPackedDenseLifetime();
   LifetimesAndBulkOrigin();
   SolveBorrowed(ModifiedGramSchmidt{});
   SolveBorrowed(ClassicalGramSchmidtTwoPass{});
   UnequalCapacitiesAndNativeStrides(ModifiedGramSchmidt{});
   UnequalCapacitiesAndNativeStrides(ClassicalGramSchmidtTwoPass{});
   Check(RequiredGMRESReductionWorkspaceSize(255, 3) == 6);
   Check(RequiredGMRESReductionWorkspaceSize(256, 3) == 6);
   Check(RequiredGMRESReductionWorkspaceSize(257, 3) == 9);
   Check(RequiredGMRESReductionWorkspaceSize(4, 2) == 4);
   Check(RequiredGMRESReductionWorkspaceSize(513, 3) == 12);
   Check(RequiredGMRESReductionWorkspaceSize(0, 1) == 2);
   return failures ? 1 : 0;
}

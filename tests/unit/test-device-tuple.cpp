// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Utilities/utilities.hpp>
#include <gendil/Algebra/Vector/vector.hpp>
#include <gendil/FiniteElementMethod/finiteelementspace.hpp>
#include <gendil/FiniteElementMethod/WeakForm/weakform.hpp>
#include <gendil/FiniteElementMethod/WeakForm/pullback.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace device_tuple_test
{
using namespace gendil;

#ifndef GENDIL_USE_DEVICE
// These checks deliberately use host-only callbacks and unique_ptr storage.
// GPU configurations exercise device-callable values below; CPU configurations
// additionally verify the helpers' forwarding and ownership semantics.
struct QualifiedCallable
{
   int operator()(int value) & { return value+1; }
   int operator()(int value) && { return value+10; }
};

bool CheckForwarding()
{
   int value = 3;
   const int constant = 7;
   std::tuple<int&, const int> left{value, constant};
   std::tuple<const int&, std::unique_ptr<int>> right{constant, std::make_unique<int>(11)};
   auto joined = Concatenate(left, std::move(right));
   static_assert(std::is_same_v<decltype(joined),
      std::tuple<int&, const int, const int&, std::unique_ptr<int>>>);
   std::get<0>(joined) = 5;
   auto& result = Apply([](int& a, const int& b, const int& c, const auto& p) -> int&
      { a += b+c+*p; return a; }, joined);
   bool ok = value == 30 && &result == &value && !std::get<1>(right);
   const auto& constant_tuple = joined;
   static_assert(std::is_same_v<decltype(Apply(
      [](auto& first, const auto&...) -> decltype(auto) { return (first); }, constant_tuple)), int&>);

   auto moved = std::tuple{std::make_unique<int>(17)};
   ok = Apply([](std::unique_ptr<int> p) { return *p; }, std::move(moved)) == 17 && ok;
   ok = !std::get<0>(moved) && ok;
   QualifiedCallable callable;
   ok = Apply(callable, std::tuple{2}) == 3 && ok;
   ok = Apply(QualifiedCallable{}, std::tuple{2}) == 12 && ok;

   // The tuple overload must not capture existing tensor-index applications.
   auto indexed = [](GlobalIndex i, GlobalIndex j) { return 10*i+j; };
   const TensorIndex<2> index{GlobalIndex{2}, GlobalIndex{3}};
   ok = Apply(indexed, index) == 23 && ok;

   constexpr auto empty = Concatenate(std::tuple<>{}, std::tuple<>{});
   static_assert(std::is_same_v<std::remove_cv_t<decltype(empty)>, std::tuple<>>);
   static_assert(Apply([] { return 42; }, empty) == 42);
   static_assert(Apply([](int a, int b) { return a+b; }, std::tuple{2, 3}) == 5);
   auto from_empty = Concatenate(empty, std::tuple{std::make_unique<int>(19)});
   ok = *std::get<0>(from_empty) == 19 && ok;
   if (!ok) std::cerr << "Tuple forwarding/reference/empty checks failed\n";
   return ok;
}
#endif

// A non-constexpr constructor makes this a non-literal type. NVCC's relaxed
// constexpr option alone cannot make std::apply's corresponding return path
// device-callable. Keep both tuple and map construction in the actual kernel.
struct RuntimeValue
{
   Real value;
   GENDIL_HOST_DEVICE explicit RuntimeValue(Real x) : value(x) {}
};
struct First {};
struct Second {};
struct SumEntries
{
   template<class Entry>
   GENDIL_HOST_DEVICE
   Real operator()(Real sum, const Entry& entry) const { return sum+entry.value.value; }
};

GENDIL_HOST_DEVICE
Real EvaluateTupleMap(Real x)
{
   auto tuple = Concatenate(std::tuple<RuntimeValue>{RuntimeValue{x}},
                            std::tuple<RuntimeValue>{RuntimeValue{x+1}});
   const auto combined = Apply([] GENDIL_HOST_DEVICE (const RuntimeValue& a, const RuntimeValue& b)
      { return RuntimeValue{a.value+2*b.value}; }, tuple);
   auto entry = MakeEntry<First>(combined);
   auto map = ToStaticMap(std::tuple<decltype(entry)>{entry});
   auto appended = MapAppend(std::move(map), MakeEntry<Second>(RuntimeValue{x+3}));
   auto merged = MapInsertOrMerge<First>(std::move(appended), RuntimeValue{x+4},
      [] GENDIL_HOST_DEVICE (const RuntimeValue& a, const RuntimeValue& b)
      { return RuntimeValue{a.value+b.value}; });
   return MapFold(Real{0}, merged, SumEntries{});
}

GENDIL_HOST_DEVICE
Real EvaluatePullbackMap(Real x)
{
   // Build channels directly: no named-space specialization is needed for
   // this regression of non-literal ScaleExpr sums and StaticMap replacement.
   const auto first = PullbackResult{MakeStaticMap(MakeEntry<ValueChannel>(ScaleExpr{x}))};
   const auto second = PullbackResult{MakeStaticMap(
      MakeEntry<ValueChannel>(ScaleExpr{x+1}), MakeEntry<GradientChannel>(ScaleExpr{2*x}))};
   const auto merged = merge_two_pullback_results(first, second);
   return merged.template Get<ValueChannel>()(0, 0, 0, 0, 0, 0)
      + merged.template Get<GradientChannel>()(0, 0, 0, 0, 0, 0);
}

bool CheckKernelConstruction()
{
   constexpr std::array<Real, 4> values{-.5, 0, 1.25, 3};
   Vector input(values.size()), output(2*values.size());
   auto* host = input.WriteHostData();
   for (size_t i = 0; i < values.size(); ++i) host[i] = values[i];
   const auto* source = input.ReadDeviceData();
   auto* destination = output.WriteDeviceData();
   DeviceLoop(values.size(), [source, destination] GENDIL_HOST_DEVICE (size_t i)
   {
      destination[2*i] = EvaluateTupleMap(source[i]);
      destination[2*i+1] = EvaluatePullbackMap(source[i]);
   });
   GENDIL_DEVICE_SYNC;
   const auto* actual = output.ReadHostData();
   bool ok = true;
   for (size_t i = 0; i < values.size(); ++i)
   {
      const Real expected[2]{5*values[i]+9, 4*values[i]+1};
      for (size_t j = 0; j < 2; ++j)
      {
         if (!std::isfinite(actual[2*i+j]) ||
             std::abs(actual[2*i+j]-expected[j]) > 1.e-13*(1+std::abs(expected[j])))
         {
            std::cerr << "Device tuple/map result " << 2*i+j << ": got " << actual[2*i+j]
                      << ", expected " << expected[j] << '\n';
            ok = false;
         }
      }
   }
   return ok;
}
}

int main()
{
   bool ok = true;
#ifndef GENDIL_USE_DEVICE
   ok = device_tuple_test::CheckForwarding();
#endif
   return device_tuple_test::CheckKernelConstruction() && ok ? 0 : 1;
}

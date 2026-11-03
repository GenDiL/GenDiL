// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/staticalgebraconcepts.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"
#include <array>
#include <type_traits>
#include <utility>

namespace gendil
{
namespace array_arithmetic_detail
{
template<class T>
inline constexpr bool is_array = false;

template<class T, std::size_t N>
inline constexpr bool is_array<std::array<T, N>> = true;

template<class T>
inline constexpr bool is_recursive_array = false;

template<class T, class... Dims>
inline constexpr bool is_recursive_array<RecursiveArray<T, Dims...>> = true;

struct Add
{
   template<class L, class R>
   GENDIL_HOST_DEVICE constexpr auto operator()(const L& lhs, const R& rhs) const
      -> decltype(lhs + rhs) { return lhs + rhs; }
};

struct Subtract
{
   template<class L, class R>
   GENDIL_HOST_DEVICE constexpr auto operator()(const L& lhs, const R& rhs) const
      -> decltype(lhs - rhs) { return lhs - rhs; }
};

template<class Operation, class L, class R>
using scalar_result_t = std::remove_cvref_t<decltype(
   Operation{}(std::declval<const L&>(), std::declval<const R&>()))>;

template<class T>
concept StaticArray = StaticVector<T> || StaticMatrix<T>;

template<class L, class R>
concept SameShape =
   (StaticVector<L> && StaticVector<R> && static_extent_v<L> == static_extent_v<R>) ||
   (StaticMatrix<L> && StaticMatrix<R> &&
    static_num_rows_v<L> == static_num_rows_v<R> &&
    static_num_cols_v<L> == static_num_cols_v<R>);

// C++ operators require a class or enumeration operand. Raw arrays remain
// readable operands when paired with a class, but cannot define raw/raw syntax.
template<class T>
inline constexpr bool class_operand = std::is_class_v<std::remove_cvref_t<T>> ||
   std::is_union_v<std::remove_cvref_t<T>> || std::is_enum_v<std::remove_cvref_t<T>>;

template<class L, class R>
concept GenericBinaryOperands = SameShape<L, R> &&
   (class_operand<L> || class_operand<R>) &&
   !(is_recursive_array<std::remove_cvref_t<L>> &&
     is_recursive_array<std::remove_cvref_t<R>>);

template<class T>
using static_value_t = std::conditional_t<StaticMatrix<T>, matrix_value_t<T>, vector_value_t<T>>;

// Storage selection is independent of operand readability. Preserve standard
// arrays where requested; mixed/custom inputs always produce owning storage.
template<class Entry, class Shape, bool Standard, bool Matrix = StaticMatrix<Shape>>
struct result_storage
{
   using type = std::conditional_t<Standard,
      std::array<Entry, static_extent_v<Shape>>,
      SerialRecursiveArray<Entry, static_extent_v<Shape>>>;
};

template<class Entry, class Shape, bool Standard>
struct result_storage<Entry, Shape, Standard, true>
{
   using type = std::conditional_t<Standard,
      std::array<std::array<Entry, static_num_cols_v<Shape>>, static_num_rows_v<Shape>>,
      SerialRecursiveArray<Entry, static_num_rows_v<Shape>, static_num_cols_v<Shape>>>;
};

template<class Entry, class Shape, bool Standard = is_array<std::remove_cvref_t<Shape>>>
using result_t = typename result_storage<Entry, Shape, Standard>::type;

template<StaticArray Shape, class Function>
GENDIL_HOST_DEVICE
constexpr void ForEachEntry(const Function& function)
{
   // Expand the fixed index space explicitly, while retaining the integral
   // index interface promised by StaticVector and StaticMatrix accessors.
   const auto apply = [&](auto... indices)
   {
      function(static_cast<GlobalIndex>(indices)...);
   };
   if constexpr (StaticVector<Shape>)
   {
      ConstexprLoop<static_extent_v<Shape>>(apply);
   }
   else
   {
      ConstexprLoop<static_num_rows_v<Shape>, static_num_cols_v<Shape>>(apply);
   }
}

template<class Operation, class L, class R>
   requires SameShape<L, R>
GENDIL_HOST_DEVICE
constexpr auto ApplyStatic(const L& lhs, const R& rhs)
{
   using Left = static_value_t<L>;
   using Right = static_value_t<R>;
   using Entry = scalar_result_t<Operation, Left, Right>;
   result_t<Entry, L, is_array<std::remove_cvref_t<L>> &&
                      is_array<std::remove_cvref_t<R>>> result{};
   ForEachEntry<L>([&](auto... indices)
   {
      GetValue(result, indices...) = Operation{}(
         static_cast<Left>(GetValue(lhs, indices...)),
         static_cast<Right>(GetValue(rhs, indices...)));
   });
   return result;
}

template<class Result, StaticArray Input, class Operation>
GENDIL_HOST_DEVICE
constexpr Result ApplyEntrywise(const Input& input, const Operation& operation)
{
   using Scalar = static_value_t<Input>;
   Result result{};
   ForEachEntry<Input>([&](auto... indices)
   {
      GetValue(result, indices...) = operation(static_cast<Scalar>(GetValue(input, indices...)));
   });
   return result;
}

// Compatibility below the vector/matrix concept boundary: arbitrary nested
// ranks and custom scalar leaves retain their existing componentwise behavior.
// Only a nested std::array with exactly these extents exposes a scalar leaf.
template<class Array, std::size_t... Dims>
struct array_leaf {};

template<class T>
   requires (!is_array<T> && !is_recursive_array<T>)
struct array_leaf<T> { using type = T; };

template<class T, std::size_t N, std::size_t... Dims>
struct array_leaf<std::array<T, N>, N, Dims...>
   : array_leaf<std::remove_cv_t<T>, Dims...> {};

template<class Array, std::size_t... Dims>
using array_leaf_t = typename array_leaf<std::remove_cv_t<Array>, Dims...>::type;

template<class T>
GENDIL_HOST_DEVICE
constexpr const T& array_at(const T& value) { return value; }

template<class T, std::size_t N, class... Indices>
GENDIL_HOST_DEVICE
constexpr decltype(auto) array_at(const std::array<T, N>& value,
                                std::size_t first, Indices... rest)
{
   return array_at(value[first], rest...);
}

template<class T, class... Dims, class... Indices>
GENDIL_HOST_DEVICE
decltype(auto) array_at(const RecursiveArray<T, Dims...>& value, Indices... indices)
{
   return value(indices...);
}

// Deduce nested results without invoking an array operator during its own
// constraint checking. Mismatched inner extents have no result type.
template<class Operation, class L, class R,
         bool ArrayPair = is_array<L> && is_array<R>, class = void>
struct binary_result {};

template<class Operation, class L, class R>
struct binary_result<Operation, L, R, false,
                     std::void_t<scalar_result_t<Operation, L, R>>>
{ using type = scalar_result_t<Operation, L, R>; };

template<class Operation, class L, class R, std::size_t N>
struct binary_result<Operation, std::array<L, N>, std::array<R, N>, true,
   std::void_t<typename binary_result<
      Operation, std::remove_cv_t<L>, std::remove_cv_t<R>>::type>>
{
   using type = std::array<typename binary_result<
      Operation, std::remove_cv_t<L>, std::remove_cv_t<R>>::type, N>;
};

template<class Operation, class L, class R>
GENDIL_HOST_DEVICE
constexpr auto Apply(const L& lhs, const R& rhs)
   -> typename binary_result<Operation, L, R>::type
{
   if constexpr (is_array<L> && is_array<R>)
   {
      using Result = typename binary_result<Operation, L, R>::type;
      return [&]<std::size_t... I>(std::index_sequence<I...>)
      {
         return Result{Apply<Operation>(lhs[I], rhs[I])...};
      }(std::make_index_sequence<std::tuple_size_v<L>>{});
   }
   else
   {
      return Operation{}(lhs, rhs);
   }
}

template<class Operation, class Value, std::size_t... Dims, class L, class R>
GENDIL_HOST_DEVICE
auto ApplyMixed(const L& lhs, const R& rhs)
{
   SerialRecursiveArray<Value, Dims...> result;
   UnitLoop<Dims...>([&](auto... indices)
   {
      result(indices...) = Operation{}(
         array_at(lhs, indices...), array_at(rhs, indices...));
   });
   return result;
}
} // namespace array_arithmetic_detail

/** Add compatible readable static vectors or matrices componentwise.
 * Inputs need no writable storage or default constructor. Results own their
 * entries: two standard arrays retain standard storage; other combinations
 * return recursive arrays. Entry types follow scalar arithmetic promotion.
 * Native recursive-array operations retain their existing contracts.
 * Outside gendil, bring the desired operator into scope. Include this header
 * before defining templates that perform arithmetic on standard arrays.
 */
template<class L, class R>
   requires array_arithmetic_detail::GenericBinaryOperands<L, R>
GENDIL_HOST_DEVICE
constexpr auto operator+(const L& lhs, const R& rhs)
{
   return array_arithmetic_detail::ApplyStatic<array_arithmetic_detail::Add>(lhs, rhs);
}

/// Subtract compatible readable vectors or matrices in operand order.
template<class L, class R>
   requires array_arithmetic_detail::GenericBinaryOperands<L, R>
GENDIL_HOST_DEVICE
constexpr auto operator-(const L& lhs, const R& rhs)
{
   return array_arithmetic_detail::ApplyStatic<array_arithmetic_detail::Subtract>(lhs, rhs);
}

// These overloads are disjoint from arithmetic-valued vectors and matrices.
template<class L, class R, std::size_t N>
   requires (!(array_arithmetic_detail::StaticArray<std::array<L, N>> &&
               array_arithmetic_detail::StaticArray<std::array<R, N>>))
GENDIL_HOST_DEVICE
constexpr auto operator+(const std::array<L, N>& lhs, const std::array<R, N>& rhs)
   -> typename array_arithmetic_detail::binary_result<
      array_arithmetic_detail::Add, std::array<L, N>, std::array<R, N>>::type
{
   return array_arithmetic_detail::Apply<array_arithmetic_detail::Add>(lhs, rhs);
}

template<class L, class R, std::size_t N>
   requires (!(array_arithmetic_detail::StaticArray<std::array<L, N>> &&
               array_arithmetic_detail::StaticArray<std::array<R, N>>))
GENDIL_HOST_DEVICE
constexpr auto operator-(const std::array<L, N>& lhs, const std::array<R, N>& rhs)
   -> typename array_arithmetic_detail::binary_result<
      array_arithmetic_detail::Subtract, std::array<L, N>, std::array<R, N>>::type
{
   return array_arithmetic_detail::Apply<array_arithmetic_detail::Subtract>(lhs, rhs);
}

// Mixed storage uses matching scalar entries and returns an owning recursive
// array in either operand order, as matrix-vector multiplication does.
template<class L, class R, std::size_t N, std::size_t... Dims>
   requires (!(array_arithmetic_detail::StaticArray<std::array<L, N>> &&
               array_arithmetic_detail::StaticArray<SerialRecursiveArray<R, N, Dims...>>))
GENDIL_HOST_DEVICE
auto operator+(const std::array<L, N>& lhs,
               const SerialRecursiveArray<R, N, Dims...>& rhs)
   -> SerialRecursiveArray<array_arithmetic_detail::scalar_result_t<
      array_arithmetic_detail::Add,
      array_arithmetic_detail::array_leaf_t<L, Dims...>, R>, N, Dims...>
{
   using Value = array_arithmetic_detail::scalar_result_t<
      array_arithmetic_detail::Add, array_arithmetic_detail::array_leaf_t<L, Dims...>, R>;
   return array_arithmetic_detail::ApplyMixed<
      array_arithmetic_detail::Add, Value, N, Dims...>(lhs, rhs);
}

template<class L, class R, std::size_t N, std::size_t... Dims>
   requires (!(array_arithmetic_detail::StaticArray<SerialRecursiveArray<L, N, Dims...>> &&
               array_arithmetic_detail::StaticArray<std::array<R, N>>))
GENDIL_HOST_DEVICE
auto operator+(const SerialRecursiveArray<L, N, Dims...>& lhs,
               const std::array<R, N>& rhs)
   -> SerialRecursiveArray<array_arithmetic_detail::scalar_result_t<
      array_arithmetic_detail::Add,
      L, array_arithmetic_detail::array_leaf_t<R, Dims...>>, N, Dims...>
{
   using Value = array_arithmetic_detail::scalar_result_t<
      array_arithmetic_detail::Add, L, array_arithmetic_detail::array_leaf_t<R, Dims...>>;
   return array_arithmetic_detail::ApplyMixed<
      array_arithmetic_detail::Add, Value, N, Dims...>(lhs, rhs);
}

template<class L, class R, std::size_t N, std::size_t... Dims>
   requires (!(array_arithmetic_detail::StaticArray<std::array<L, N>> &&
               array_arithmetic_detail::StaticArray<SerialRecursiveArray<R, N, Dims...>>))
GENDIL_HOST_DEVICE
auto operator-(const std::array<L, N>& lhs,
               const SerialRecursiveArray<R, N, Dims...>& rhs)
   -> SerialRecursiveArray<array_arithmetic_detail::scalar_result_t<
      array_arithmetic_detail::Subtract,
      array_arithmetic_detail::array_leaf_t<L, Dims...>, R>, N, Dims...>
{
   using Value = array_arithmetic_detail::scalar_result_t<
      array_arithmetic_detail::Subtract, array_arithmetic_detail::array_leaf_t<L, Dims...>, R>;
   return array_arithmetic_detail::ApplyMixed<
      array_arithmetic_detail::Subtract, Value, N, Dims...>(lhs, rhs);
}

template<class L, class R, std::size_t N, std::size_t... Dims>
   requires (!(array_arithmetic_detail::StaticArray<SerialRecursiveArray<L, N, Dims...>> &&
               array_arithmetic_detail::StaticArray<std::array<R, N>>))
GENDIL_HOST_DEVICE
auto operator-(const SerialRecursiveArray<L, N, Dims...>& lhs,
               const std::array<R, N>& rhs)
   -> SerialRecursiveArray<array_arithmetic_detail::scalar_result_t<
      array_arithmetic_detail::Subtract,
      L, array_arithmetic_detail::array_leaf_t<R, Dims...>>, N, Dims...>
{
   using Value = array_arithmetic_detail::scalar_result_t<
      array_arithmetic_detail::Subtract, L, array_arithmetic_detail::array_leaf_t<R, Dims...>>;
   return array_arithmetic_detail::ApplyMixed<
      array_arithmetic_detail::Subtract, Value, N, Dims...>(lhs, rhs);
}
} // namespace gendil

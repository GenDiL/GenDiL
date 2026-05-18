// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"

namespace gendil {

template<StaticString Name>
struct StringKey {};

template<StaticString Name>
struct NameTag { static constexpr auto name = Name; };

// keyed entry
template<class Key, class T>
struct Entry {
   using key_type   = Key;
   using value_type = T;
   T value;
};

template<class Key, class... Entries>
struct find_entry;

template<class Key, class T, class... Rest>
struct find_entry<Key, Entry<Key, T>, Rest...> { using type = Entry<Key, T>; };

template<class Key, class OtherKey, class T, class... Rest>
struct find_entry<Key, Entry<OtherKey, T>, Rest...> : find_entry<Key, Rest...> {};

template<class Key>
struct find_entry<Key> {
   static_assert(sizeof(Key) == 0, "StaticMap: requested key is not present.");
};

template<class Key, class... Entries>
using find_entry_t = typename find_entry<Key, Entries...>::type;

template<class... Ts>
struct are_unique : std::true_type {};

template<class T, class... Rest>
struct are_unique<T, Rest...>
   : std::bool_constant<(!(std::is_same_v<T, Rest>) && ...) && are_unique<Rest...>::value>
{};

template<class Key, class T>
constexpr auto make_entry(T&& value)
{
   return Entry<Key, std::decay_t<T>>{ std::forward<T>(value) };
}

template<class Key, class... Entries>
constexpr decltype(auto) Get(const std::tuple<Entries...>& t) {
   using E = find_entry_t<Key, Entries...>;
   return std::get<E>(t).value;
}

template<StaticString Name, class... Entries>
constexpr decltype(auto) Get(const std::tuple<Entries...>& t) {
   using E = find_entry_t<StringKey<Name>, Entries...>;
   return std::get<E>(t).value;
}

template<class... Entries>
struct StaticMap {
   static_assert(
      are_unique<typename Entries::key_type...>::value,
      "StaticMap: duplicate keys are not allowed.");

   std::tuple<Entries...> entries;

   template<class Key>
   static consteval bool contains() {
      return (std::is_same_v<Key, typename Entries::key_type> || ...);
   }

   template<class Key>
   constexpr decltype(auto) get() & {
      using E = find_entry_t<Key, Entries...>;
      return (std::get<E>(entries).value);
   }

   template<class Key>
   constexpr decltype(auto) get() const & {
      using E = find_entry_t<Key, Entries...>;
      return (std::get<E>(entries).value);
   }
};

template<class... Es>
constexpr auto make_map(Es&&... es) {
   using Map = StaticMap<std::decay_t<Es>...>;
   return Map{ std::tuple<std::decay_t<Es>...>{ std::forward<Es>(es)... } };
}

// template<class Key, class... Entries>
// consteval bool contains( const StaticMap<Entries...> & map )
// {
//    return StaticMap<Entries...>::template contains<Key>();
// }

template<class Tuple>
constexpr auto tuple_to_map(Tuple&& t) {
   return std::apply(
      [](auto&&... es) {
      return make_map(std::forward<decltype(es)>(es)...);
      },
      std::forward<Tuple>(t)
   );
}

template<class... Entries, class NewEntry>
constexpr auto append(StaticMap<Entries...> map, NewEntry&& new_entry)
{
   using E = std::decay_t<NewEntry>;
   static_assert(
      !StaticMap<Entries...>::template contains<typename E::key_type>(),
      "StaticMap: duplicate key in MapAppend.");

   return std::apply(
      [&](auto&&... es)
      {
         return make_map(
            std::forward<decltype(es)>(es)...,
            std::forward<NewEntry>(new_entry));
      },
      std::move(map.entries)
   );
}

template<class Key, class EntryT, class MergeValueFn>
constexpr auto replace_entry(EntryT&& entry, MergeValueFn&& merge_value_fn)
{
   using E = std::remove_cvref_t<EntryT>;

   if constexpr (std::is_same_v<Key, typename E::key_type>)
   {
      auto new_value = merge_value_fn(std::forward<EntryT>(entry).value);
      return Entry<Key, std::remove_cvref_t<decltype(new_value)>>{ std::move(new_value) };
   }
   else
   {
      return std::forward<EntryT>(entry);
   }
}

template<class Key, class... Entries, class MergeValueFn>
constexpr auto replace(StaticMap<Entries...> map, MergeValueFn&& merge_value_fn)
{
   static_assert(
      StaticMap<Entries...>::template contains<Key>(),
      "StaticMap: key not present in MapReplace.");

   return std::apply(
      [&](auto&&... es)
      {
         return make_map(
            replace_entry<Key>(
               std::forward<decltype(es)>(es),
               merge_value_fn)...);
      },
      std::move(map.entries)
   );
}

template<class Key, class... Entries, class Value, class MergeFn>
constexpr auto map_insert_or_merge(
   StaticMap<Entries...> map,
   Value&& value,
   MergeFn&& merge_fn)
{
   using Map = StaticMap<Entries...>;
   auto value_copy = std::forward<Value>(value);

   if constexpr (Map::template contains<Key>())
   {
      return replace<Key>(
         std::move(map),
         [&](auto&& old_value)
         {
            return merge_fn(
               std::forward<decltype(old_value)>(old_value),
               value_copy);
         });
   }
   else
   {
      return append(
         std::move(map),
         make_entry<Key>(std::move(value_copy)));
   }
}

template<class Acc, class Fn, class... Entries>
constexpr auto fold(Acc acc, const StaticMap<Entries...>& map, Fn&& fn)
{
   return std::apply(
      [&](const auto&... es)
      {
         auto result = std::move(acc);
         ((result = fn(std::move(result), es)), ...);
         return result;
      },
      map.entries
   );
}

} // namespace gendil

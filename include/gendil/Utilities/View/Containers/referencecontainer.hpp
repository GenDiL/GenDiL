// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief Non-owning container adaptor bound to an lvalue container. */

#include "gendil/Utilities/types.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

namespace detail {

template<class Parent, class Index, class = void>
struct ReferenceContainerElement
{
   using type = std::remove_reference_t<decltype(
      std::declval<Parent&>()[std::declval<Index>()])>;
};

// A proxy access result is not the stored numerical element type.
template<class Parent, class Index>
struct ReferenceContainerElement<Parent, Index,
   std::void_t<typename Parent::element_type>>
{
   using access_type = decltype(std::declval<Parent&>()[std::declval<Index>()]);
   using type = std::conditional_t<std::is_reference_v<access_type>,
      std::remove_reference_t<access_type>, typename Parent::element_type>;
};

} // namespace detail

/**
 * @brief Delegate indexed access to a parent container held by reference.
 *
 * Const access delegates to the const parent; the parent determines whether
 * the resulting element access is read-only. A mutable adaptor delegates to
 * `Parent&`, which is still const when `Parent` is const. Neither access path
 * overrides a parent's reference, value, or proxy semantics.
 *
 * The parent must outlive the adaptor and must not move while it is referenced.
 * GPU access additionally requires a device-accessible parent: copying a
 * reference to a host object into a kernel does not relocate that object.
 */
template<class Parent, class IndexType = GlobalIndex>
class ReferenceContainer
{
public:
   using parent_type = Parent;
   using index_type = IndexType;
   using element_type = typename detail::ReferenceContainerElement<
      Parent, IndexType>::type;

   GENDIL_HOST_DEVICE
   explicit constexpr ReferenceContainer(Parent& parent) : parent_(parent) { }

   using unqualified_parent_type = std::remove_const_t<Parent>;
   ReferenceContainer(unqualified_parent_type&&) = delete;
   ReferenceContainer(const unqualified_parent_type&&) = delete;

   GENDIL_HOST_DEVICE ReferenceContainer(const ReferenceContainer&) = default;
   GENDIL_HOST_DEVICE ReferenceContainer(ReferenceContainer&&) = default;
   ReferenceContainer& operator=(const ReferenceContainer&) = delete;
   ReferenceContainer& operator=(ReferenceContainer&&) = delete;

   GENDIL_HOST_DEVICE
   constexpr Parent& Get() { return parent_; }

   GENDIL_HOST_DEVICE
   constexpr const Parent& Get() const { return parent_; }

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) operator[](const IndexType index)
   {
      return parent_[index];
   }

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) operator[](const IndexType index) const
   {
      return std::as_const(parent_)[index];
   }

private:
   Parent& parent_;
};

template<class Parent>
ReferenceContainer(Parent&) -> ReferenceContainer<Parent>;

} // namespace gendil

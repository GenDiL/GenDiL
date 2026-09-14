// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Element views with weighted gather/scatter access to local DoFs.
 *
 * A restriction associates each element-local DoF with zero or more
 * algebraic entries @f$(I_k,w_k)@f$. For algebraic storage @f$x@f$, reading
 * local DoF @f$i@f$ gathers
 *
 * @f[
 *    (E x)_i = \sum_k w_k x_{I_k}.
 * @f]
 *
 * `MakeWeightedRestrictionView` exposes these local DoFs through the ordinary
 * `View<Container, Layout>` indexing syntax. The layout converts logical
 * local-DoF and element coordinates into a `WeightedRestrictionKey`; the
 * container converts that key into a borrowing `WeightedDofReference` proxy.
 * Converting or materializing the proxy gathers the local value. Assignment
 * applies adjoint scatter-assignment, `operator+=` performs non-atomic adjoint
 * scatter-add, and `AtomicAddInPlace` performs atomic adjoint scatter-add.
 *
 * These proxies intentionally do not behave like ordinary scalar references.
 * In particular, assignment represents an algebraic restriction operation,
 * not rebinding, inverse interpolation, or solution of a constraint. A proxy
 * borrows the restriction and storage handles held by its parent view and
 * must not outlive or survive movement of that view.
 *
 * Typical element code explicitly materializes gathered values and scatters
 * completed local residuals:
 *
 * @code{.cpp}
 * auto dofs = MakeWeightedRestrictionView( restriction, algebraic_values );
 * const auto value = Materialize( dofs( local_i, element ) );
 * dofs( local_i, element ) += local_residual;
 * @endcode
 */

#include "gendil/FiniteElementMethod/Restrictions/restrictionoperations.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/get.hpp"
#include "gendil/Utilities/View/view.hpp"

#include <array>
#include <type_traits>
#include <utility>

namespace gendil {

/**
 * @brief Identify an element and one native local DoF for weighted access.
 *
 * `local_dof` contains the native tensor coordinate expected by the
 * restriction, and `element` selects its mesh element. Reference-to-native
 * orientation must therefore be applied before constructing or accessing
 * this key.
 *
 * @tparam Dim Rank of the element-local tensor coordinate.
 */
template < Integer Dim >
struct WeightedRestrictionKey
{
   GlobalIndex element{};
   std::array< GlobalIndex, Dim > local_dof{};
};

/**
 * @brief Map logical element-view indices to a `WeightedRestrictionKey`.
 *
 * The first `local_rank` logical axes are the native local-DoF coordinates;
 * the final axis is the element index. Local-axis extents come statically from
 * the restriction's DoF shape. The runtime element extent is deliberately
 * omitted, so this layout does not by itself provide a complete logical
 * `Size()`.
 *
 * Unlike an affine layout, `Offset` returns a semantic DoF key rather than a
 * scalar storage offset. `WeightedRestrictionContainer` interprets that key by
 * constructing the corresponding weighted DoF proxy.
 *
 * @tparam Restriction Completed tensor restriction represented by the view.
 */
template < typename Restriction >
struct WeightedRestrictionLayout
{
   using restriction_type = std::remove_cvref_t< Restriction >;
   using dof_shape_type = typename restriction_type::dof_shape_type;

   static constexpr size_t local_rank = restriction_type::tensor_dim;
   static constexpr size_t rank = local_rank + 1;

   /** @brief Return the static extent of one local tensor axis. */
   template < size_t Axis >
      requires ( Axis < local_rank )
   GENDIL_HOST_DEVICE
   static constexpr auto Get( AxisTag< Axis > )
   {
      return std::integral_constant<
         GlobalIndex,
         static_cast< GlobalIndex >(
            seq_get_v< Axis, dof_shape_type > ) >{};
   }

   /** @brief Build the semantic DoF key addressed by logical coordinates. */
   template < typename... Indices >
      requires ( sizeof...( Indices ) == rank )
   GENDIL_HOST_DEVICE
   constexpr auto Offset( Indices... indices ) const
   {
      const auto coordinates = std::make_tuple( indices... );
      return MakeKey(
         coordinates,
         std::make_index_sequence< local_rank >{} );
   }

private:
   template < class Coordinates, size_t... Axis >
   GENDIL_HOST_DEVICE
   static constexpr auto MakeKey(
      const Coordinates & coordinates,
      std::index_sequence< Axis... > )
   {
      return WeightedRestrictionKey< static_cast< Integer >( local_rank ) >{
         static_cast< GlobalIndex >( std::get< local_rank >( coordinates ) ),
         { static_cast< GlobalIndex >( std::get< Axis >( coordinates ) )... } };
   }
};

/**
 * @brief Borrowing value/scatter proxy for one weighted element-local DoF.
 *
 * For restriction entries @f$(I_k,w_k)@f$, conversion to `value_type` evaluates
 * the weighted gather. Writing a local value @f$z@f$ applies the corresponding
 * adjoint weight to every visited algebraic coordinate:
 *
 * @f[
 *    x_{I_k} \mathrel{\leftarrow} w_k^* z,
 *    \qquad\text{or}\qquad
 *    x_{I_k} \mathrel{+}= w_k^* z.
 * @f]
 *
 * The exact weight action is provided by `ApplyRestrictionWeight` and
 * `ApplyAdjointRestrictionWeight`; current real scalar weights satisfy
 * @f$w_k^*=w_k@f$.
 *
 * Copy or move construction copies the borrowing handle. Copy or move
 * assignment has deliberately different semantics: it first gathers and
 * fully materializes the source, then scatters that value through the
 * destination DoF's restriction. Assignment never rebinds the destination
 * proxy.
 *
 * The proxy is writable only when both the storage object and its element
 * type are mutable. Scatter-assignment is valid only when the caller has a
 * complete final value and the visited coordinates are unique for that write.
 * Non-atomic addition additionally requires execution-level freedom from
 * concurrent writes; use `AtomicAddInPlace` when contributions may conflict.
 *
 * The proxy is permanently bound by reference to the supplied restriction
 * and storage handles. They must remain alive and unmoved for its entire
 * lifetime.
 *
 * @tparam Restriction Completed restriction defining algebraic entry visitation.
 * @tparam Storage Indexable algebraic storage handle.
 * @tparam Dim Rank of the native local-DoF coordinate.
 */
template < typename Restriction, typename Storage, Integer Dim >
class WeightedDofReference
{
public:
   using storage_type = Storage;
   using unqualified_storage_type = std::remove_cv_t< storage_type >;
   using element_type = typename unqualified_storage_type::element_type;
   using value_type = std::remove_const_t< element_type >;

   /** Whether this proxy exposes scatter operations. */
   static constexpr bool is_writable =
      !std::is_const_v< storage_type > &&
      !std::is_const_v< element_type >;

   /** @brief Borrow restriction and storage handles for an element-local DoF. */
   GENDIL_HOST_DEVICE
   constexpr WeightedDofReference(
      const Restriction & restriction,
      Storage & storage,
      const WeightedRestrictionKey< Dim > & key )
      : restriction_( restriction ), storage_( storage ), key_( key )
   { }

   /** @brief Copy the borrowing handle without gathering or scattering. */
   GENDIL_HOST_DEVICE
   constexpr WeightedDofReference(
      const WeightedDofReference & ) = default;

   /** @brief Move the borrowing handle without gathering or scattering. */
   GENDIL_HOST_DEVICE
   constexpr WeightedDofReference(
      WeightedDofReference && ) = default;

   /** @brief Gather and return the weighted value of this local DoF. */
   GENDIL_HOST_DEVICE
   constexpr operator value_type() const
   {
      return GatherRestrictionRow(
         restriction_, key_.element, key_.local_dof, storage_ );
   }

   /**
    * @brief Adjoint-scatter a complete local value by assignment.
    *
    * This visits every restriction entry and assigns its adjoint-weighted
    * value. It neither sums duplicate coordinates nor preserves prior values.
    */
   GENDIL_HOST_DEVICE
   constexpr WeightedDofReference & operator=(
      const value_type & value )
      requires is_writable
   {
      ScatterAssignRestrictionRowAdjoint(
         restriction_, key_.element, key_.local_dof, value, storage_ );
      return *this;
   }

   /**
    * @brief Materialize a source proxy, then adjoint-scatter it by assignment.
    *
    * Materialization is completed before any destination write, including
    * when the source and destination stencils overlap.
    */
   GENDIL_HOST_DEVICE
   constexpr WeightedDofReference & operator=(
      const WeightedDofReference & source )
      requires is_writable
   {
      const value_type value = static_cast< value_type >( source );
      return *this = value;
   }

   /** @brief Materialize an rvalue source proxy, then adjoint-scatter it. */
   GENDIL_HOST_DEVICE
   constexpr WeightedDofReference & operator=(
      WeightedDofReference && source )
      requires is_writable
   {
      const value_type value = static_cast< value_type >( source );
      return *this = value;
   }

   /**
    * @brief Add an adjoint-weighted local value without synchronization.
    *
    * The caller must ensure that concurrently executing work does not update
    * the same algebraic coordinates.
    */
   GENDIL_HOST_DEVICE
   constexpr WeightedDofReference & operator+=(
      const value_type & value )
      requires is_writable
   {
      ScatterAddRestrictionRowAdjoint(
         restriction_, key_.element, key_.local_dof, value, storage_ );
      return *this;
   }

   /** @brief Atomically add an adjoint-weighted local value. */
   GENDIL_HOST_DEVICE
   constexpr void AtomicAdd( const value_type & value ) const
      requires is_writable
   {
      AtomicScatterAddRestrictionRowAdjoint(
         restriction_, key_.element, key_.local_dof, value, storage_ );
   }

private:
   const Restriction & restriction_;
   Storage & storage_;
   WeightedRestrictionKey< Dim > key_;
};

/** @brief Identify an exact `WeightedDofReference` specialization. */
template < class T >
struct is_weighted_dof_reference : std::false_type { };

template < class Restriction, class Storage, Integer Dim >
struct is_weighted_dof_reference<
   WeightedDofReference< Restriction, Storage, Dim > > : std::true_type { };

/**
 * @brief Whether `T`, after removing cv/ref qualifiers, is a weighted DoF
 * proxy.
 */
template < class T >
inline constexpr bool is_weighted_dof_reference_v =
   is_weighted_dof_reference< std::remove_cvref_t< T > >::value;

/**
 * @brief Produce an independent value from a scalar or weighted DoF proxy.
 *
 * A weighted DoF proxy is explicitly gathered into its `value_type`.
 * Other inputs are copied or moved into their unqualified value type. This
 * operation is useful before `auto` deduction or unconstrained numeric
 * template dispatch, where an implicit proxy conversion may not participate.
 * The result never borrows the input proxy or preserves an input reference.
 *
 * @param value Value or weighted DoF proxy to materialize.
 * @return An independent unqualified value.
 */
template < class Value >
GENDIL_HOST_DEVICE
constexpr auto Materialize( Value && value )
{
   if constexpr ( is_weighted_dof_reference_v< Value > )
   {
      using Result = typename std::remove_cvref_t< Value >::value_type;
      return static_cast< Result >( value );
   }
   else
   {
      using Result = std::remove_cvref_t< Value >;
      return Result( std::forward< Value >( value ) );
   }
}

/**
 * @brief Atomically adjoint-scatter through a writable weighted DoF proxy.
 *
 * The proxy is intentionally accepted by value so a temporary proxy returned
 * by a named view's indexing operation can be passed directly. Copying the
 * proxy copies only its borrowing handle; it does not gather the local value.
 *
 * @param destination Writable weighted DoF proxy.
 * @param value Local value whose adjoint-weighted contributions are added.
 */
template < class Restriction, class Storage, Integer Dim, class Value >
GENDIL_HOST_DEVICE
void AtomicAddInPlace(
   WeightedDofReference< Restriction, Storage, Dim > destination,
   const Value & value )
   requires WeightedDofReference<
      Restriction, Storage, Dim >::is_writable
{
   destination.AtomicAdd(
      static_cast< typename WeightedDofReference<
         Restriction, Storage, Dim >::value_type >( value ) );
}

/**
 * @brief View container that turns semantic DoF keys into borrowing proxies.
 *
 * The restriction and storage objects are stored by value and are expected to
 * be lightweight execution handles. Mutable lvalue access returns a writable
 * proxy when the storage element type is mutable. Const lvalue access returns
 * a read-only proxy regardless of the underlying storage's mutability.
 *
 * Returned proxies borrow the two members of this container. Consequently,
 * indexing an rvalue container is disabled and a proxy must not outlive or
 * survive movement of its parent view. A writable view captured by value in a
 * kernel is const unless the closure is declared `mutable`.
 *
 * @tparam Restriction Completed restriction copied into the view container.
 * @tparam Storage Algebraic storage handle copied into the view container.
 */
template < typename Restriction, typename Storage >
struct WeightedRestrictionContainer
{
   using element_type = typename Storage::element_type;

   /** Restriction handle used to visit each local DoF's algebraic entries. */
   Restriction restriction;

   /** Algebraic storage handle read or updated by the weighted DoF proxy. */
   Storage storage;

   /** @brief Return a potentially writable proxy from a mutable container. */
   template < Integer Dim >
   GENDIL_HOST_DEVICE
   constexpr auto operator[]( const WeightedRestrictionKey< Dim > & key ) &
   {
      return WeightedDofReference< Restriction, Storage, Dim >{
         restriction, storage, key };
   }

   /** @brief Return a read-only proxy from a const container. */
   template < Integer Dim >
   GENDIL_HOST_DEVICE
   constexpr auto operator[]( const WeightedRestrictionKey< Dim > & key ) const &
   {
      return WeightedDofReference< Restriction, const Storage, Dim >{
         restriction, storage, key };
   }

   /** A DoF proxy borrows this container's restriction and storage handles. */
   template < Integer Dim >
   GENDIL_HOST_DEVICE
   auto operator[]( const WeightedRestrictionKey< Dim > & ) && = delete;

   template < Integer Dim >
   GENDIL_HOST_DEVICE
   auto operator[]( const WeightedRestrictionKey< Dim > & ) const && = delete;
};

/**
 * @brief Bind a completed restriction to an algebraic storage handle.
 *
 * The resulting view has the restriction's native local tensor axes followed
 * by an element axis. Indexing produces a borrowing weighted DoF proxy; use
 * `Materialize` when an independent gathered value is required.
 *
 * Both arguments are copied into the view's container after removing their
 * top-level cv/ref qualifiers. Top-level constness of the source handle does
 * not make its referenced elements read-only. Supply a storage type whose
 * `element_type` is const, such as `PointerContainer<const T>`, to create a
 * permanently read-only view. The supplied objects should be lightweight
 * handles; their referenced memory remains governed by their own ownership
 * and lifetime rules.
 *
 * @param restriction Completed restriction defining each local DoF's entries.
 * @param storage Non-pointer algebraic storage handle.
 * @return A proxy-backed logical element view.
 */
template < typename Restriction, typename Storage >
GENDIL_HOST_DEVICE
constexpr auto MakeWeightedRestrictionView(
   const Restriction & restriction,
   const Storage & storage )
   requires ( !std::is_pointer_v< std::remove_cvref_t< Storage > > )
{
   using RestrictionType = std::remove_cvref_t< Restriction >;
   using StorageType = std::remove_cvref_t< Storage >;
   return MakeView(
      WeightedRestrictionContainer< RestrictionType, StorageType >{
         restriction, storage },
      WeightedRestrictionLayout< RestrictionType >{} );
}

/**
 * @brief Bind a completed restriction directly to pointer-backed storage.
 *
 * This convenience overload wraps `storage` in `PointerContainer<T>` and
 * delegates to the storage-handle overload. Passing `const T*` therefore
 * creates a read-only weighted view.
 *
 * @param restriction Completed restriction defining each local DoF's entries.
 * @param storage Pointer to the algebraic storage selected for execution.
 * @return A proxy-backed logical element view.
 */
template < typename Restriction, typename T >
GENDIL_HOST_DEVICE
constexpr auto MakeWeightedRestrictionView(
   const Restriction & restriction,
   T * storage )
{
   return MakeWeightedRestrictionView(
      restriction,
      PointerContainer< T >{ storage } );
}

} // namespace gendil

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief Group existing vectors and dense operands for a GMRES solve. */

#include "gendil/Algebra/Vector/access.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace gendil {

/// An unpreconditioned solve has no preconditioned-direction sequence.
struct NoDirections {};

/**
 * @brief Run GMRES with vectors and reduced-system arrays supplied by the caller.
 *
 * The basis holds Krylov search vectors and work holds a system application.
 * H stores the reduced matrix; cosines and sines describe its Givens rotations;
 * g is the reduced right-hand side and y holds solution-correction coefficients.
 * Once these operands are formed, bind them without copying their entries:
 * @code
 * NoDirections directions;
 * auto buffers = TieGMRESBuffers(
 *    basis, work, directions, H, cosines, sines, g, y);
 * @endcode
 * basis(j) lends a vector; H(i,j) addresses the Hessenberg matrix; the four
 * remaining dense vectors use indexed access. Right-preconditioned solves
 * supply directions(j) for preconditioned basis vectors instead of NoDirections.
 * This is useful for separately allocated buffers or for storage inside a
 * kernel; the owning alternative is GMRESWorkspace.
 *
 * This aggregate describes neither ownership nor a common reserved capacity.
 * For active restart m, the basis needs at least m+1 vectors and H needs m+1
 * rows. H needs m columns, cosines/sines/y need m entries, and g needs m+1.
 * Every selected vector has the system's logical length. Fixed-right GMRES needs
 * one direction; FGMRES needs m. Unpreconditioned GMRES never uses directions.
 * Physical offsets and strides remain properties of the supplied views.
 * Operand types are independent: a concrete adapter may explicitly retain
 * known borrowing descriptors by value and owners by reference.
 *
 * Referenced objects, including descriptors, must remain alive and stable
 * during the solve. Storage must be accessible to the selected execution and
 * nonoverlapping where distinct operands are required. Cooperative local
 * solves require shared dense scratch and uniform participation; thread-local
 * solves use private storage. Modern host solves check logical lengths in debug
 * and release before numerical callbacks. Local CPU-debug solves assert them;
 * local release/device solves require valid lengths as a precondition. No
 * hidden backing allocation or memory residency is inspected.
 */
template<class Basis, class Work, class Directions,
         class H, class C, class S, class G, class Y>
struct GMRESWorkspaceView
{
   /// basis(j) supplies the j-th Krylov vector without copying its entries.
   Basis basis;
   /// Scratch vector for system applications and fixed-right corrections.
   Work work;
   /// Preconditioned vectors, or NoDirections for an unpreconditioned solve.
   Directions directions;
   /// Arnoldi coefficients, subsequently rotated into the reduced triangular system.
   H hessenberg;
   /// Cosines of the rotations applied to Hessenberg columns.
   C cosines;
   /// Sines of the same rotations.
   S sines;
   /// Reduced right-hand side, initialized to beta*e_0 and transformed by rotations.
   G reduced_rhs;
   /// Correction coefficients; also reused for the second CGS2 projection pass.
   Y coefficients;
};

/**
 * @brief Bind eight existing operands without converting or copying them.
 *
 * All arguments must be named lvalues, including cheap view descriptors and
 * NoDirections. Temporary arguments, even const temporaries, are rejected.
 * The returned aggregate references those exact objects; copies keep referring
 * to them. Keep every object and its backing storage alive and stable.
 * This binder performs no normalization, capacity checks, or synchronization.
 */
template<class... Operands>
   requires (sizeof...(Operands) == 8 &&
             (std::is_lvalue_reference_v<Operands&&> && ...))
GENDIL_HOST_DEVICE
auto TieGMRESBuffers(Operands&&... operands)
{
   return GMRESWorkspaceView<Operands&...>{operands...};
}

namespace gmres_detail {

// Packed reservation: ld*c Hessenberg entries, then c,c,c+1,c scalar entries.
GENDIL_HOST_DEVICE
inline size_t GMRESDenseSize(const size_t capacity, const size_t leading_dimension)
{
   [[maybe_unused]] const size_t limit = std::numeric_limits<size_t>::max();
   GENDIL_VERIFY(capacity <= (limit - 1) / 4,
      "GMRES dense workspace size overflow.");
   GENDIL_VERIFY(leading_dimension >= capacity + 1,
      "GMRES Hessenberg leading dimension must be at least capacity + 1.");
   const size_t auxiliary_size = 4 * capacity + 1;
   GENDIL_VERIFY(capacity == 0 ||
      leading_dimension <= (limit - auxiliary_size) / capacity,
      "GMRES dense workspace size overflow.");
   return leading_dimension * capacity + auxiliary_size;
}

/// The five descriptor values returned by the concrete packed-storage adapter.
template<class H, class C, class S, class G, class Y>
struct DenseArrays
{
   H hessenberg;
   C cosines;
   S sines;
   G reduced_rhs;
   Y coefficients;
};

} // namespace gmres_detail

/**
 * @brief Split a scalar view into the five packed dense operands used by GMRES.
 *
 * Use this helper when the reduced matrix and its four vectors share one
 * allocation. Supply a sized pointer-backed view, a reservation c, and a
 * leading dimension ld >= c+1. Separate arrays need no helper: form their
 * native views and bind them directly.
 *
 * Storage needs at least ld*c+4*c+1 logical entries. H has shape (c+1,c);
 * padding between its columns is retained. Cosines, sines, g, and coefficients
 * have lengths c, c, c+1, and c, and begin at offsets ld*c, ld*c+c, ld*c+2*c,
 * and ld*c+3*c+1 respectively.
 *
 * Offsets and ld count logical entries of storage, not raw pointer entries:
 * H(i,j) = storage[i+ld*j] and g(i) = storage[ld*c+2*c+i]. Thus a parent origin
 * and scalar stride remain effective, including structure-of-arrays storage.
 * Smaller active restarts never alter this packed reservation or its offsets.
 * The returned descriptors copy the supplied view, not its values or a reference
 * to that descriptor. The argument descriptor may therefore disappear while
 * the underlying storage remains live and unmoved.
 * Host construction checks packed arithmetic and the supplied logical length;
 * device construction requires those properties already to hold.
 */
template<class T, class Index, class Layout>
   requires (Layout::rank == 1 && has_complete_extents_v<Layout>)
GENDIL_HOST_DEVICE
auto MakeGMRESPackedDense(View<PointerContainer<T, Index>, Layout> storage,
                         const size_t capacity, const size_t leading_dimension)
{
   const size_t size = gmres_detail::GMRESDenseSize(capacity, leading_dimension);
   GENDIL_VERIFY(GetVectorSize(storage) >= size,
      "GMRES packed dense storage is too small.");
   const size_t h_size = leading_dimension * capacity;
   return gmres_detail::DenseArrays{
      MakeView(storage, StridedLayout<2, DynamicExtents<2>>{
         std::array<GlobalIndex, 2>{1, leading_dimension},
         DynamicExtents<2>{capacity + 1, capacity}}),
      MakeView(storage, MakeOffsetLayout(VectorLayout{capacity}, h_size)),
      MakeView(storage, MakeOffsetLayout(VectorLayout{capacity}, h_size + capacity)),
      MakeView(storage, MakeOffsetLayout(VectorLayout{capacity + 1}, h_size + 2 * capacity)),
      MakeView(storage, MakeOffsetLayout(VectorLayout{capacity}, h_size + 3 * capacity + 1))};
}

} // namespace gendil

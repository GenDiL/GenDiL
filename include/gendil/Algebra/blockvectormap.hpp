// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Allocation-free heterogeneous tuple block vector maps.
 *
 * A block vector map represents
 *
 * \f[
 *   y_i = \sum_j A_{ij} x_j,
 * \f]
 *
 * where every block `A_ij` satisfies the contextual `VectorMap` protocol for
 * the vector views passed to it. The outer tuple enumerates block rows and
 * each inner tuple enumerates block columns:
 *
 * @code
 * auto map = MakeBlockVectorMap(std::tuple{
 *    std::tuple{a00, a01},
 *    std::tuple{a10, ZeroVectorMap{row1_size, column1_size}}});
 * @endcode
 *
 * Blocks are heterogeneous and stored by value, so move-only sparse matrices
 * and nested block maps are supported. Application partitions the input and
 * output through non-owning vector views. It performs no block extraction,
 * type erasure, virtual dispatch, or per-application allocation.
 */

#include "gendil/Algebra/vectormapconcept.hpp"
#include "gendil/Algebra/Vector/operations.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

/**
 * @brief Sized structural zero used in heterogeneous block vector maps.
 *
 * Replacement application writes a zero vector of `OutputSize()`, whereas
 * additive application leaves the destination unchanged. The input is not
 * read, but its extent is validated so the zero block participates in the
 * same rectangular dimension checks as every other block.
 */
class ZeroVectorMap
{
public:
   /** Construct a zero map from an input space to an output space. */
   ZeroVectorMap(const std::size_t output_size, const std::size_t input_size)
      : output_size_(output_size), input_size_(input_size)
   {}

   /** Return the required input-vector extent. */
   std::size_t InputSize() const { return input_size_; }

   /** Return the required output-vector extent. */
   std::size_t OutputSize() const { return output_size_; }

   /**
    * Replace `output` with the zero vector.
    *
    * @param input Input used only for extent validation.
    * @param output Destination of length `OutputSize()`.
    */
   template<class InputVector, class OutputVector>
      requires (
         requires(const InputVector& input, const OutputVector& output)
         {
            { GetVectorSize(input) } -> std::same_as<std::size_t>;
            { GetVectorSize(output) } -> std::same_as<std::size_t>;
         } &&
         (HostWritableVector<OutputVector> ||
          DeviceWritableVector<OutputVector>))
   void operator()(const InputVector& input, OutputVector& output) const
   {
      GENDIL_VERIFY(
         GetVectorSize(input) == input_size_ &&
         GetVectorSize(output) == output_size_,
         "ZeroVectorMap vector dimensions do not match." );
      if constexpr (DeviceWritableVector<OutputVector>)
      {
         Zero<true>(output);
      }
      else
      {
         Zero<false>(output);
      }
   }

   /**
    * Add the zero vector to `output`, leaving its values unchanged.
    *
    * Both vector extents are still validated.
    */
   template<class InputVector, class OutputVector>
      requires requires(const InputVector& input, const OutputVector& output)
      {
         { GetVectorSize(input) } -> std::same_as<std::size_t>;
         { GetVectorSize(output) } -> std::same_as<std::size_t>;
      }
   void ApplyAdd(const InputVector& input, OutputVector& output) const
   {
      GENDIL_VERIFY(
         GetVectorSize(input) == input_size_ &&
         GetVectorSize(output) == output_size_,
         "ZeroVectorMap vector dimensions do not match." );
   }

private:
   std::size_t output_size_;
   std::size_t input_size_;
};

namespace block_vector_map_detail {

/** Return whether every block row has the first row's column count. */
template<class RowsTuple, size_t... Row>
consteval bool RowsHaveSameColumnCount(std::index_sequence<Row...>)
{
   constexpr size_t Columns = std::tuple_size_v<std::remove_cvref_t<
      std::tuple_element_t<0, std::remove_cvref_t<RowsTuple>>>>;
   return ((std::tuple_size_v<std::remove_cvref_t<
      std::tuple_element_t<Row, std::remove_cvref_t<RowsTuple>>>> ==
      Columns) && ...);
}

/** Add two block extents with checked conversion and overflow handling. */
inline size_t CheckedSizeAdd(const size_t lhs, const size_t rhs)
{
   return static_cast<size_t>(CheckedAdd(
      static_cast<GlobalIndex>(lhs),
      static_cast<GlobalIndex>(rhs),
      "Block vector-map extent overflow." ));
}

/** Apply one block additively to its corresponding vector subranges. */
template<class Map, class InputVector, class OutputVector>
   requires VectorMap<Map, InputVector, OutputVector>
void ApplyBlockAdd(
   const Map& map,
   const InputVector& input,
   OutputVector& output)
{
   ApplyAdd(map, input, output);
}

/** Test one block row against the vector-view types used for application. */
template<class RowTuple, class InputView, class OutputView, size_t... Column>
consteval bool RowContainsVectorMaps(std::index_sequence<Column...>)
{
   return (VectorMap<
      std::tuple_element_t<Column, std::remove_cvref_t<RowTuple>>,
      InputView,
      OutputView> && ...);
}

/** Test every block against the vector-view types used for application. */
template<class RowsTuple,
         class InputView,
         class OutputView,
         size_t... Row>
consteval bool AllBlocksAreVectorMaps(std::index_sequence<Row...>)
{
   return (RowContainsVectorMaps<
      std::tuple_element_t<Row, std::remove_cvref_t<RowsTuple>>,
      InputView,
      OutputView>(std::make_index_sequence<std::tuple_size_v<
         std::remove_cvref_t<std::tuple_element_t<
            Row,
            std::remove_cvref_t<RowsTuple>>>>>{}) && ...);
}

/**
 * Require sliceable vectors and blocks compatible with their generated views.
 *
 * This is contextual in the complete input and output vector types because
 * each block receives owner-bound subviews rather than those complete vectors.
 */
template<class RowsTuple, class InputVector, class OutputVector>
concept CompatibleBlockVectorMaps =
   requires(const InputVector& input, OutputVector& output)
   {
      MakeVectorView(input, std::size_t{0}, std::size_t{0});
      MakeVectorView(output, std::size_t{0}, std::size_t{0});
   } &&
   AllBlocksAreVectorMaps<
      RowsTuple,
      decltype(MakeVectorView(
         std::declval<const InputVector&>(),
         std::size_t{0},
         std::size_t{0})),
      decltype(MakeVectorView(
         std::declval<OutputVector&>(),
         std::size_t{0},
         std::size_t{0}))>(
            std::make_index_sequence<std::tuple_size_v<
               std::remove_cvref_t<RowsTuple>>>{});

} // namespace block_vector_map_detail

/**
 * @brief Heterogeneous rectangular block map stored as a tuple of row tuples.
 *
 * If block `A_ij` has dimensions `m_i` by `n_j`, every block in row `i`
 * must have output size `m_i` and every block in column `j` must have input
 * size `n_j`. Construction validates those relationships and computes the
 * aggregate dimensions
 *
 * \f[
 *   m = \sum_i m_i, \qquad n = \sum_j n_j.
 * \f]
 *
 * Replacement application zeros the complete output once and then invokes
 * `ApplyAdd` on every block. Additive application skips that initialization.
 * Both paths create only lightweight owner-bound views of the corresponding
 * input and output blocks.
 *
 * @tparam RowsTuple Tuple of equally sized row tuples containing concrete
 *                   vector-map objects.
 *
 * @note Blocks are owned by value. Passing an lvalue tuple to the factory
 *       copies it; passing an rvalue tuple moves it and permits move-only
 *       blocks.
 * @note In-place or overlapping input/output application is not guaranteed by
 *       the `VectorMap` protocol.
 */
template<class RowsTuple>
class BlockVectorMap
{
   static constexpr size_t NumRows =
      std::tuple_size_v<std::remove_cvref_t<RowsTuple>>;
   static_assert(NumRows > 0, "BlockVectorMap requires at least one row.");
   static constexpr size_t NumColumns = std::tuple_size_v<
      std::remove_cvref_t<std::tuple_element_t<0, RowsTuple>>>;
   static_assert(
      NumColumns > 0,
      "BlockVectorMap requires at least one column." );
   static_assert(
      block_vector_map_detail::RowsHaveSameColumnCount<RowsTuple>(
         std::make_index_sequence<NumRows>{}),
      "Every BlockVectorMap row must contain the same number of columns." );

public:
   /**
    * Construct the block map and validate its row and column dimensions.
    *
    * Extent sums are checked for overflow. A dimension mismatch is reported
    * before the map can be applied.
    */
   explicit BlockVectorMap(RowsTuple blocks)
      : blocks_(std::move(blocks))
   {
      InitializeDimensions(std::make_index_sequence<NumRows>{});
   }

   /** Return the sum of the block-column input extents. */
   size_t InputSize() const { return input_size_; }

   /** Return the sum of the block-row output extents. */
   size_t OutputSize() const { return output_size_; }

   /**
    * Apply the block map with replacement semantics.
    *
    * The complete destination is zeroed once, after which every block adds
    * its contribution to its row subview.
    */
   template<class InputVector, class OutputVector>
      requires
         (HostWritableVector<OutputVector> ||
          DeviceWritableVector<OutputVector>) &&
         block_vector_map_detail::CompatibleBlockVectorMaps<
            RowsTuple,
            InputVector,
            OutputVector>
   void operator()(const InputVector& input, OutputVector& output) const
   {
      ValidateVectorSizes(input, output);
      // Replacement semantics initialize the complete destination once;
      // individual block rows and terms only contribute additively.
      if constexpr (DeviceWritableVector<OutputVector>)
      {
         Zero<true>(output);
      }
      else
      {
         Zero<false>(output);
      }
      ApplyRows(input, output, std::make_index_sequence<NumRows>{});
   }

   /**
    * Add the block-map result to the existing destination.
    *
    * No output row is initialized by the block map or by an individual block.
    */
   template<class InputVector, class OutputVector>
      requires block_vector_map_detail::CompatibleBlockVectorMaps<
         RowsTuple,
         InputVector,
         OutputVector>
   void ApplyAdd(const InputVector& input, OutputVector& output) const
   {
      ValidateVectorSizes(input, output);
      ApplyRows(input, output, std::make_index_sequence<NumRows>{});
   }

private:
   template<size_t Row>
   void InitializeRow()
   {
      const auto& row = std::get<Row>(blocks_);
      row_sizes_[Row] = GetOutputSize(std::get<0>(row));
      std::apply(
         [&](const auto&... block)
         {
            GENDIL_VERIFY(
               ((GetOutputSize(block) == row_sizes_[Row]) && ...),
               "All blocks in one row must have equal output dimensions." );
         },
         row);
   }

   template<size_t Column, size_t... Row>
   void InitializeColumn(std::index_sequence<Row...>)
   {
      column_sizes_[Column] = GetInputSize(
         std::get<Column>(std::get<0>(blocks_)));
      GENDIL_VERIFY(
         ((GetInputSize(
              std::get<Column>(std::get<Row>(blocks_))) ==
            column_sizes_[Column]) && ...),
         "All blocks in one column must have equal input dimensions." );
   }

   template<size_t... Row>
   void InitializeDimensions(std::index_sequence<Row...> rows)
   {
      (InitializeRow<Row>(), ...);
      [&]<size_t... Column>(std::index_sequence<Column...>)
      {
         (InitializeColumn<Column>(rows), ...);
      }(std::make_index_sequence<NumColumns>{});
      for (size_t row = 0; row < NumRows; ++row)
      {
         row_offsets_[row] = output_size_;
         output_size_ = block_vector_map_detail::CheckedSizeAdd(
            output_size_, row_sizes_[row]);
      }
      for (size_t column = 0; column < NumColumns; ++column)
      {
         column_offsets_[column] = input_size_;
         input_size_ = block_vector_map_detail::CheckedSizeAdd(
            input_size_, column_sizes_[column]);
      }
   }

   template<class InputVector, class OutputVector>
   void ValidateVectorSizes(
      const InputVector& input,
      const OutputVector& output) const
   {
      GENDIL_VERIFY(
         GetVectorSize(input) == input_size_,
         "BlockVectorMap input vector has the wrong extent." );
      GENDIL_VERIFY(
         GetVectorSize(output) == output_size_,
         "BlockVectorMap output vector has the wrong extent." );
   }

   template<size_t Row, class InputVector, class OutputVector>
   void ApplyRow(const InputVector& input, OutputVector& output) const
   {
      auto row_output = MakeVectorView(
         output, row_offsets_[Row], row_sizes_[Row]);
      const auto& blocks = std::get<Row>(blocks_);
      [&]<size_t... Column>(std::index_sequence<Column...>)
      {
         (block_vector_map_detail::ApplyBlockAdd(
            std::get<Column>(blocks),
            MakeVectorView(
               input,
               column_offsets_[Column],
               column_sizes_[Column]),
            row_output), ...);
      }(std::make_index_sequence<NumColumns>{});
   }

   template<class InputVector,
            class OutputVector,
            size_t... Row>
   void ApplyRows(
      const InputVector& input,
      OutputVector& output,
      std::index_sequence<Row...>) const
   {
      (ApplyRow<Row>(input, output), ...);
   }

   RowsTuple blocks_;
   std::array<size_t, NumRows> row_offsets_{};
   std::array<size_t, NumRows> row_sizes_{};
   std::array<size_t, NumColumns> column_offsets_{};
   std::array<size_t, NumColumns> column_sizes_{};
   size_t input_size_ = 0;
   size_t output_size_ = 0;
};

/**
 * @brief Construct a heterogeneous block vector map with value ownership.
 *
 * @param blocks Outer tuple of block rows, each represented by an inner tuple.
 * @return A `BlockVectorMap` whose stored tuple has cv/ref qualifiers removed.
 *
 * Row count, column count, and row tuple lengths are compile-time properties.
 * Individual block dimensions are validated by the returned object's
 * constructor.
 */
template<class RowsTuple>
auto MakeBlockVectorMap(RowsTuple&& blocks)
{
   using StoredTuple = std::remove_cvref_t<RowsTuple>;
   return BlockVectorMap<StoredTuple>{std::forward<RowsTuple>(blocks)};
}

} // namespace gendil

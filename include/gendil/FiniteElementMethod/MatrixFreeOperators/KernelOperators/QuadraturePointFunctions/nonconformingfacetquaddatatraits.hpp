// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Leaf recognition and tree-offset traits for nonconforming facet
 * qdata.
 *
 * The traits in this header classify the two supported scalar leaf types and
 * describe the flattened leaf order of nested TensorProductData. Transformation
 * and mapping remain in nonconformingfacetquaddata.hpp.
 */

#include "gendil/Utilities/tensorproductdata.hpp"
#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/doftoquad.hpp"

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace gendil
{

/**
 * @brief Recognizes a one-dimensional finite-element qdata leaf supported by
 * nonconforming facet remapping.
 *
 * Only CachedDofToQuad leaves are currently supported.
 */
template<class T>
struct IsSupportedNonconformingFacetDofToQuad1D : std::false_type {};

template<class ShapeFunctions, class IntegrationRule>
struct IsSupportedNonconformingFacetDofToQuad1D<
   CachedDofToQuad<ShapeFunctions, IntegrationRule>> : std::true_type {};

template<class T>
inline constexpr bool IsSupportedNonconformingFacetDofToQuad1D_v =
   IsSupportedNonconformingFacetDofToQuad1D<
      std::remove_cvref_t<T>>::value;

/**
 * @brief Recognizes a static one-dimensional point-set qdata leaf.
 *
 * A supported leaf exposes static GetCoord(q) and GetWeight(q) operations whose
 * results are convertible to Real.
 */
template<class T>
struct IsStaticPointSetFacetQData1D : std::false_type {};

template<class T>
   requires requires(Integer q)
   {
      { std::remove_cvref_t<T>::GetCoord(q) } -> std::convertible_to<Real>;
      { std::remove_cvref_t<T>::GetWeight(q) } -> std::convertible_to<Real>;
   }
struct IsStaticPointSetFacetQData1D<T> : std::true_type {};

template<class T>
inline constexpr bool IsStaticPointSetFacetQData1D_v =
   IsStaticPointSetFacetQData1D<std::remove_cvref_t<T>>::value;

namespace nonconforming_facet_qdata_detail
{

/**
 * @brief Number of leaves in a nested TensorProductData subtree.
 *
 * Non-TensorProductData entries count as one leaf. Recursive sums therefore
 * define the left-to-right flattened coordinate count while retaining the
 * original nested tree shape.
 */
template<class QData>
struct TensorProductQDataLeafCount
   : std::integral_constant<Integer, 1> {};

template<class... Entries>
struct TensorProductQDataLeafCount<TensorProductData<Entries...>>
   : std::integral_constant<
        Integer,
        (TensorProductQDataLeafCount<Entries>::value + ...)> {};

/**
 * @brief Flattened coordinate offset of entry I among sibling subtrees.
 *
 * The offset is the prefix sum of leaf counts for entries preceding I. It is
 * compile-time recursive state used by the transformation implementation to
 * assign each qdata leaf its cell-coordinate index.
 */
template<size_t I, class... Entries>
struct TensorProductQDataEntryOffset;

template<class First, class... Rest>
struct TensorProductQDataEntryOffset<0, First, Rest...>
   : std::integral_constant<Integer, 0> {};

template<size_t I, class First, class... Rest>
struct TensorProductQDataEntryOffset<I, First, Rest...>
   : std::integral_constant<
        Integer,
        TensorProductQDataLeafCount<First>::value +
           TensorProductQDataEntryOffset<I - 1, Rest...>::value> {};

} // namespace nonconforming_facet_qdata_detail

} // namespace gendil

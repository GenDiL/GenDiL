// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/NumericalIntegration/QuadraturePoints/nonconformingpoints.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/doftoquad.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/nonconformingfacetquaddatatraits.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace gendil
{

namespace nonconforming_facet_qdata_detail
{
/**
 * @brief Recursively maps scalar facet qdata into a nonconforming face.
 *
 * TensorProductData nodes retain their nesting. Their cached DoF-to-quadrature
 * and static point-set leaves are assigned consecutive cell-coordinate indices
 * beginning at @p Offset.
 */
template<Integer Offset, class Face, class QData>
GENDIL_HOST_DEVICE
auto MakeNonconformingScalarFacetQuadDataAtOffset(
   const Face& face,
   const QData& qdata);

template<
   Integer Offset,
   class Face,
   class... Entries,
   size_t... I>
GENDIL_HOST_DEVICE
auto MakeNonconformingScalarTensorProductFacetQuadDataAtOffset(
   const Face& face,
   const TensorProductData<Entries...>& qdata,
   std::index_sequence<I...>)
{
   return MakeTensorProductData(
      MakeNonconformingScalarFacetQuadDataAtOffset<
         Offset + TensorProductQDataEntryOffset<I, Entries...>::value>(
            face,
            GetTensorProductEntry<I>(qdata))...);
}

template<Integer Offset, class Face, class QData>
GENDIL_HOST_DEVICE
auto MakeNonconformingScalarFacetQuadDataAtOffset(
   const Face& face,
   const QData& qdata)
{
   using Data = std::remove_cvref_t<QData>;
   if constexpr (is_tensor_product_data_v<Data>)
   {
      return MakeNonconformingScalarTensorProductFacetQuadDataAtOffset<Offset>(
         face,
         qdata,
         std::make_index_sequence<tensor_product_entry_count_v<Data>>{});
   }
   else if constexpr (IsSupportedNonconformingFacetDofToQuad1D_v<Data>)
   {
      return MakeNonconformingDofToQuadData(
         face,
         qdata,
         std::integral_constant<Integer, Offset>{});
   }
   else if constexpr (IsStaticPointSetFacetQData1D_v<Data>)
   {
      return NonconformingMappedPointSet1D<Data, Face, Offset>{face};
   }
   else
   {
      static_assert(
         dependent_false_v<Face, QData>,
         "Nonconforming facet scalar qdata leaf must be "
         "CachedDofToQuad or static point-set data.");
   }
}

template<class Face, class... ScalarMaps, size_t... Component>
GENDIL_HOST_DEVICE
auto MakeNonconformingVectorFacetQuadData(
   const Face& face,
   const VectorDofToQuad<ScalarMaps...>& local_face_qd,
   std::index_sequence<Component...>)
{
   return MakeVectorDofToQuad(
      MakeNonconformingScalarFacetQuadDataAtOffset<0>(
         face,
         GetVectorComponent<Component>(local_face_qd))...);
}

} // namespace nonconforming_facet_qdata_detail

/**
 * @brief Maps scalar or componentwise-vector facet qdata into a nonconforming
 * face.
 *
 * Scalar trees are transformed recursively. Each VectorDofToQuad component is
 * a separate scalar tree and therefore restarts its spatial coordinate offset
 * at zero.
 */
template<class Face, class LocalFaceQData>
GENDIL_HOST_DEVICE
auto MakeNonconformingFacetQuadData(
   const Face& face,
   const LocalFaceQData& local_face_qd)
{
   if constexpr (is_vector_dof_to_quad_v<LocalFaceQData>)
   {
      using QData = std::remove_cvref_t<LocalFaceQData>;
      return nonconforming_facet_qdata_detail::
         MakeNonconformingVectorFacetQuadData(
            face,
            local_face_qd,
            std::make_index_sequence<vector_component_count_v<QData>>{});
   }
   else
   {
      return nonconforming_facet_qdata_detail::
         MakeNonconformingScalarFacetQuadDataAtOffset<0>(
            face,
            local_face_qd);
   }
}

} // namespace gendil

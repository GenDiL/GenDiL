// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <type_traits>

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

namespace gendil {

namespace details
{

template<class QData1D>
GENDIL_HOST_DEVICE
Real GetWeight1D(const QData1D& qdata, Integer q)
{
   using QData = std::remove_cvref_t<QData1D>;
   if constexpr (requires { qdata.weights(q); })
   {
      return qdata.weights(q);
   }
   else if constexpr (requires { QData::GetWeight(q); })
   {
      return QData::GetWeight(q);
   }
   else if constexpr (requires { qdata.GetWeight(q); })
   {
      return qdata.GetWeight(q);
   }
   else
   {
      static_assert(
         dependent_false_v<QData1D>,
         "GetWeight(q, qdata) requires 1D qdata with weights(q) or "
         "a static GetWeight(q) point-set interface.");
   }
}

} // namespace details

/**
 * @brief Get the quadrature weight at a given quadrature point.
 * 
 * @tparam Dim The dimension of the space.
 * @tparam DofToQuads 1D DofToQuad objects for each dimension.
 * @param index The index of the quadrature point.
 * @param quad_data The DofToQuad objects containing 1D data for each dimension.
 * @return The quadrature weight.
 * 
 * @note Assumes tensor finite element with tensor integration rule.
 */
template < Integer Dim, typename... DofToQuads >
GENDIL_HOST_DEVICE
Real GetWeight( const TensorIndex< Dim > & index,
                const std::tuple< DofToQuads... > & quad_data );

template <
   Integer Dim,
   typename ... ScalarDofToQuads,
   typename ... RestVectorDofToQuads >
GENDIL_HOST_DEVICE
Real GetWeight( const TensorIndex< Dim > & index,
                const std::tuple< std::tuple< ScalarDofToQuads... >, RestVectorDofToQuads... > & quad_data )
{
   return GetWeight( index, std::get< 0 >( quad_data ) );
}

template < Integer Dim,
           typename... DofToQuads,
           size_t... Is >
GENDIL_HOST_DEVICE
Real GetWeight( const TensorIndex< Dim > & index,
                const std::tuple< DofToQuads... > & quad_data,
                std::index_sequence< Is... > )
{
   return Product(
      details::GetWeight1D( std::get< Is >( quad_data ), index[ Is ] )... );
}

template < Integer Dim, typename... DofToQuads >
GENDIL_HOST_DEVICE
Real GetWeight( const TensorIndex< Dim > & index,
                const std::tuple< DofToQuads... > & quad_data )
{
   return GetWeight( index,
                     quad_data,
                     std::make_index_sequence< Dim >{} );
}

template < CellFaceView FaceInfo, Integer Dim, typename... DofToQuads >
GENDIL_HOST_DEVICE
Real GetWeight(
   const FaceInfo & face_info,
   const TensorIndex< Dim > & index,
   const std::tuple< DofToQuads... > & face_quad_data )
{
   static_assert(
      2*Dim == sizeof...( DofToQuads ),
      "Dimension mismatch between FaceInfo and face_quad_data"
   );
   constexpr Integer local_face_index = FaceInfo::local_face_index_type::value;
   const auto & local_face_quad_data = std::get< local_face_index >( face_quad_data );
   return face_info.Measure() * GetWeight( index, local_face_quad_data );
}

}

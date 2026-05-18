// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#include "gendil/Algebra/staticvector.hpp"

namespace gendil {

// template < Integer Dim, typename QuadraturePointTensor, static_vector<Real> VecType, typename GradType >
// GENDIL_HOST_DEVICE
// void WriteQuadratureLocalValues(
//    const TensorIndex< Dim > & quad_index,
//    const VecType & field_q,
//    InterpolatedField< Empty, GradType > & field )
// {
//    WriteQuadratureLocalValues( quad_index, field_q, field.gradients, std::make_index_sequence< Dim >{} );
// }

template < Integer Dim, typename IntegrationRule, size_t... Is >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues( TensorIndex< Dim > quad_index,
                                 const Real & field_q,
                                 QuadraturePointValues< IntegrationRule > & field,
                                 std::index_sequence< Is... > )
{
   field( quad_index[Is]... ) = field_q;
}

template < Integer Dim, typename IntegrationRule >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues( TensorIndex< Dim > quad_index,
                                 const Real & field_q,
                                 QuadraturePointValues< IntegrationRule > & field )
{
   // TODO: add check that we're acccessing inside the bounds
   WriteQuadratureLocalValues( quad_index, field_q, field, std::make_index_sequence< Dim >{} );
}

template < Integer Dim, typename Values, typename Gradients, size_t... I  >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const TensorIndex< Dim > & quad_index,
   const Real & field_q,
   InterpolatedField<Values,Gradients> & field,
   std::index_sequence< I... > )
{
   field.values( quad_index[ I ]... ) = field_q;
}

template < Integer Dim, typename Tensor, size_t... I >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const TensorIndex< Dim > & quad_index,
   const Real & field_q,
   Tensor & field,
   std::index_sequence< I... > )
{
   field( quad_index[ I ]... ) = field_q;
}

template < Integer Dim, typename KernelContext, typename Tensor >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const Real & field_q,
   Tensor & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   WriteQuadratureLocalValues( quad_index, field_q, field, RegisterDimensions{} );
}

template < Integer Dim, typename IntegrationRule, static_vector<Real> VecType, size_t... Is >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   TensorIndex< Dim > quad_index,
   const VecType & field_q,
   QuadraturePointValues< IntegrationRule, static_extent_v<VecType> > & field,
   std::index_sequence< Is... > )
{
   for (LocalIndex comp = 0; comp < static_extent_v<VecType>; comp++)
   {
      field( quad_index[Is]..., comp ) = field_q[comp];
   }
}

template < Integer Dim, typename IntegrationRule, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   TensorIndex< Dim > quad_index,
   const VecType & field_q,
   QuadraturePointValues< IntegrationRule, static_extent_v<VecType> > & field )
{
   // TODO: add check that we're acccessing inside the bounds
   WriteQuadratureLocalValues( quad_index, field_q, field, std::make_index_sequence< Dim >{} );
}

// template < Integer Dim, typename Tensor, static_vector<Real> VecType, size_t... I >
// GENDIL_HOST_DEVICE
// void WriteQuadratureLocalValues(
//    const TensorIndex< Dim > & quad_index,
//    const VecType & field_q,
//    Tensor & field,
//    std::index_sequence< I... > )
// {
//    for ( LocalIndex comp = 0; comp < static_extent_v<VecType>; ++comp )
//    {
//       field( quad_index[I]..., comp ) = field_q[ comp ];
//    }
// }

template < typename KernelContext, Integer Dim, typename Tensor, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const VecType & field_q,
   Tensor & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   WriteQuadratureLocalValues( quad_index, field_q, field, RegisterDimensions{} );
}

template < typename KernelContext, Integer Dim, static_vector<Real> VecType, typename ... Tensors >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const VecType & field_q,
   std::tuple< Tensors ... > & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   constexpr Integer vdim = sizeof...( Tensors );
   ConstexprLoop< vdim >( [&]( auto i )
   {
      WriteQuadratureLocalValues( quad_index, field_q[i], std::get< i >( field ), RegisterDimensions{} );
   });
}

/**
 * @brief Write vector values to tuple storage.
 * Input: SerialRecursiveArray<Real, NumComp>
 * Output: tuple<scalar_component_values...>
 */
template < typename KernelContext, Integer Dim, Integer NumComp, typename ... Tensors >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const SerialRecursiveArray<Real,NumComp> & field_q,
   std::tuple< Tensors ... > & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   constexpr Integer vdim = sizeof...( Tensors );
   ConstexprLoop< vdim >( [&]( auto i )
   {
      WriteQuadratureLocalValues( quad_index, field_q(i), std::get< i >( field ), RegisterDimensions{} );
   });
}

/**
 * @brief Helper: Write tuple of scalar gradients componentwise.
 * Extracts grad(component, direction) and writes to each component's scalar gradient storage.
 *
 * Reuses existing vector write path: field(q..., d) = grad_c[d]
 */
template < Integer NumComp, Integer GradDim, typename RegisterDimensions, Integer Dim, typename ... Tensors >
GENDIL_HOST_DEVICE
void WriteTupleScalarGradients(
   const TensorIndex< Dim > & quad_index,
   const SerialRecursiveArray<Real, NumComp, GradDim> & field_q,
   std::tuple< Tensors ... > & field )
{
   static_assert(sizeof...(Tensors) == NumComp, "Tuple size must match NumComp");

   ConstexprLoop< NumComp >( [&]( auto c )
   {
      // Build gradient vector for this component
      std::array<Real, GradDim> grad_c;
      ConstexprLoop< GradDim >( [&]( auto d )
      {
         grad_c[d] = field_q(c, d);
      });

      // Write using existing vector-valued write overload
      // This writes: field(quad_index[Is]..., d) = grad_c[d]
      WriteQuadratureLocalValues( quad_index, grad_c, std::get< c >( field ), RegisterDimensions{} );
   });
}

/**
 * @brief Write vector gradients to tuple storage.
 * Input: SerialRecursiveArray<Real, NumComp, GradDim> with orientation grad(component, direction)
 * Output: tuple<scalar_component_gradients...>
 */
template < typename KernelContext, Integer Dim, Integer NumComp, Integer GradDim, typename ... Tensors >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const SerialRecursiveArray<Real, NumComp, GradDim> & field_q,
   std::tuple< Tensors ... > & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   WriteTupleScalarGradients<NumComp, GradDim, RegisterDimensions>(quad_index, field_q, field);
}

template < Integer Dim, typename QuadraturePointTensor >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const TensorIndex< Dim > & quad_index,
   const Real & field_q,
   QuadraturePointTensor & field )
{
   WriteQuadratureLocalValues( quad_index, field_q, field, std::make_index_sequence< Dim >{} );
}

template < Integer Dim, typename QuadraturePointTensor, static_vector<Real> VecType, size_t... Is >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const TensorIndex< Dim > & quad_index,
   const VecType & field_q,
   QuadraturePointTensor & field,
   std::index_sequence< Is... > )
{
   ConstexprLoop< static_extent_v<VecType> >(
      [&] ( auto comp )
      {
         field( quad_index[ Is ]..., comp ) = field_q[ comp ];
      }
   );
}

template < Integer Dim, typename QuadraturePointTensor, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const TensorIndex< Dim > & quad_index,
   const VecType & field_q,
   QuadraturePointTensor & field )
{
   WriteQuadratureLocalValues( quad_index, field_q, field, std::make_index_sequence< Dim >{} );
}

}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/loops.hpp"

namespace gendil {

// Scalar functions
template < Integer Dim, typename IntegrationRule, size_t... Is >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   TensorIndex< Dim > quad_index,
   const QuadraturePointValues< IntegrationRule > & field,
   Real & field_q,
   std::index_sequence< Is... > )
{
   field_q = field( quad_index[Is]... );
}

template < Integer Dim, typename IntegrationRule >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   TensorIndex< Dim > quad_index,
   const QuadraturePointValues< IntegrationRule > & field,
   Real & field_q )
{
   // TODO: add check that we're acccessing inside the bounds
   ReadQuadratureLocalValues( quad_index, field, field_q, std::make_index_sequence< Dim >{} );
}

template < Integer Dim, typename QuadPointTensor, size_t... Is>
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   const TensorIndex< Dim > & quad_index,
   const QuadPointTensor & field,
   Real & field_q,
   std::index_sequence< Is... >)
{
   field_q = field( quad_index[Is]... );
}

template < Integer Dim, typename QuadPointTensor, size_t... Is>
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   const TensorIndex< Dim > & quad_index,
   const QuadPointTensor & field,
   Real & field_q )
{
   ReadQuadratureLocalValues( quad_index, field, field_q, std::make_index_sequence< Dim >{} );
}

template < Integer Dim, typename Tensor, size_t... I >
GENDIL_HOST_DEVICE
Real ReadQuadratureLocalValues(
   const TensorIndex< Dim > & quad_index,
   const Tensor & field,
   std::index_sequence< I... > )
{
   return field( quad_index[I]... );
}

template < typename KernelContext, Integer Dim, typename Tensor >
GENDIL_HOST_DEVICE
Real ReadQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const Tensor & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   return ReadQuadratureLocalValues( quad_index, field, RegisterDimensions{} );
}

// Vector functions
template < typename KernelContext, Integer Dim, typename ... Tensors >
GENDIL_HOST_DEVICE
auto ReadQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const std::tuple< Tensors... > & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   constexpr Integer vdim = sizeof...( Tensors );
   SerialRecursiveArray<Real, vdim> result;
   ConstexprLoop< vdim >( [&]( auto i )
   {
      result(i) = ReadQuadratureLocalValues( quad_index, std::get< i >( field ), RegisterDimensions{} );
   });
   return result;
}

template < typename KernelContext, Integer Dim, Integer NComp, typename ... Tensors >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const std::tuple< Tensors... > & field,
   Real (&result)[NComp] )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   constexpr Integer vdim = sizeof...( Tensors );
   static_assert(
      NComp == vdim,
      "NComp and vdim must be equal."
   );
   ConstexprLoop< vdim >( [&]( auto i )
   {
      result[i] = ReadQuadratureLocalValues( quad_index, std::get< i >( field ), RegisterDimensions{} );
   });
}

template < typename KernelContext, Integer Dim, typename Tensor, Integer NumComp, size_t... Is >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   const KernelContext & thread,
   TensorIndex< Dim > quad_index,
   const Tensor & field,
   Real (& field_q)[NumComp],
   std::index_sequence< Is... > )
{
   for (LocalIndex comp = 0; comp < NumComp; comp++)
   {
      field_q[comp] = field( quad_index[Is]..., comp );
   }
}

template < Integer Dim, typename KernelContext, typename Tensor, Integer NumComp >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const Tensor & field,
   Real (& field_q)[NumComp] )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   return ReadQuadratureLocalValues( thread, quad_index, field, field_q, RegisterDimensions{} );
}

template < Integer Dim, typename IntegrationRule, Integer NumComp, size_t... Is >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   TensorIndex< Dim > quad_index,
   const QuadraturePointValues< IntegrationRule, NumComp > & field,
   Real (& field_q)[NumComp],
   std::index_sequence< Is... > )
{
   for (LocalIndex comp = 0; comp < NumComp; comp++)
   {
      field_q[comp] = field( quad_index[Is]..., comp );
   }
}

template < Integer Dim, typename IntegrationRule, Integer NumComp >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   TensorIndex< Dim > quad_index,
   const QuadraturePointValues< IntegrationRule, NumComp > & field,
   Real (& field_q)[NumComp] )
{
   // TODO: add check that we're acccessing inside the bounds
   ReadQuadratureLocalValues( quad_index, field, field_q, std::make_index_sequence< Dim >{} );
}


template < Integer Dim, typename QuadPointTensor, Integer NumComp, size_t ... Is >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   const TensorIndex< Dim > & quad_index,
   const QuadPointTensor & field,
   Real field_q[NumComp],
   std::index_sequence< Is... > )
{
   ConstexprLoop< NumComp >(
      [&]( auto comp )
      {
         field_q[ comp ] = field( quad_index[ Is ] ..., comp );
      }
   );
}

template < Integer Dim, typename QuadPointTensor, Integer NumComp >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   const TensorIndex< Dim > & quad_index,
   const QuadPointTensor & field,
   Real field_q[NumComp] )
{
   ReadQuadratureLocalValues( quad_index, field, field_q, std::make_index_sequence< Dim >{} );
}

// Matrix functions
template < Integer NumComp, Integer Dim, typename QuadPointTensors, size_t ... Is >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   const TensorIndex< Dim > & quad_index,
   const QuadPointTensors & field,
   Real (& field_q)[NumComp][Dim],
   std::index_sequence< Is... > )
{
   // TODO static_assert NumComp == tuple_size_v< QuadPointTensors >
   ConstexprLoop< NumComp >([&]( auto i )
   {
      auto & comp_field = std::get< i >( field );
      ConstexprLoop< Dim >([&]( auto d )
      {
         field_q[i][d] = comp_field( quad_index[ Is ] ..., d );
      });
   });
}

template < typename KernelContext, Integer NumComp, Integer Dim, typename ... QuadPointTensors >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues(
   const KernelContext & kernel_conf,
   const TensorIndex< Dim > & quad_index,
   const std::tuple< QuadPointTensors ... > & field,
   Real (& field_q)[NumComp][Dim] )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   ReadQuadratureLocalValues( quad_index, field, field_q, RegisterDimensions{} );
}

}

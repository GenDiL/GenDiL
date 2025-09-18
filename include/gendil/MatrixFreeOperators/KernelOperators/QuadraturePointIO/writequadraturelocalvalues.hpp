// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/loops.hpp"

namespace gendil {

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
void WriteQuadratureLocalValues( const KernelContext & thread, const TensorIndex< Dim > & quad_index, const Real & field_q, Tensor & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   WriteQuadratureLocalValues( quad_index, field_q, field, RegisterDimensions{} );
}

template < Integer Dim, typename IntegrationRule, Integer NumComp, size_t... Is >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues( TensorIndex< Dim > quad_index,
                                 const Real (& field_q)[ NumComp ],
                                 QuadraturePointValues< IntegrationRule, NumComp > & field,
                                 std::index_sequence< Is... > )
{
   for (LocalIndex comp = 0; comp < NumComp; comp++)
   {
      field( quad_index[Is]..., comp ) = field_q[comp];
   }
}

template < Integer Dim, typename IntegrationRule, Integer NumComp >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues( TensorIndex< Dim > quad_index,
                                 const Real (& field_q)[ NumComp ],
                                 QuadraturePointValues< IntegrationRule, NumComp > & field )
{
   // TODO: add check that we're acccessing inside the bounds
   WriteQuadratureLocalValues( quad_index, field_q, field, std::make_index_sequence< Dim >{} );
}

template < Integer Dim, typename Tensor, Integer NumComp, size_t... I >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues( const TensorIndex< Dim > & quad_index, const Real (& field_q)[ NumComp ], Tensor & field, std::index_sequence< I... > )
{
   for ( LocalIndex comp = 0; comp < NumComp; ++comp )
   {
      field( quad_index[I]..., comp ) = field_q[ comp ];
   }
}

template < typename KernelContext, Integer Dim, Integer NumComp, typename ... Tensors >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const Real (& field_q)[ NumComp ],
   std::tuple< Tensors ... > & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   constexpr Integer vdim = sizeof...( Tensors );
   ConstexprLoop< vdim >( [&]( auto i )
   {
      WriteQuadratureLocalValues( quad_index, field_q[i], std::get< i >( field ), RegisterDimensions{} );
   });
}

template < typename KernelContext, Integer Dim, typename Tensor, Integer NumComp >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const Real (& field_q)[ NumComp ],
   Tensor & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   WriteQuadratureLocalValues( quad_index, field_q, field, RegisterDimensions{} );
}

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

template < Integer Dim, typename QuadraturePointTensor >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues( const TensorIndex< Dim > & quad_index,
                                 const Real & field_q,
                                 QuadraturePointTensor & field )
{
   WriteQuadratureLocalValues( quad_index, field_q, field, std::make_index_sequence< Dim >{} );
}

template < Integer Dim, typename QuadraturePointTensor, size_t NumComp, size_t... Is >
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues( const TensorIndex< Dim > & quad_index,
                                 const Real field_q[ NumComp ],
                                 QuadraturePointTensor & field,
                                 std::index_sequence< Is... > )
{
   ConstexprLoop< NumComp >(
      [&] ( auto comp )
      {
         field( quad_index[ Is ]..., comp ) = field[ comp ];
      }
   );
}

template < Integer Dim, typename QuadraturePointTensor, size_t NumComp>
GENDIL_HOST_DEVICE
void WriteQuadratureLocalValues( const TensorIndex< Dim > & quad_index,
                                 const Real field_q[ NumComp ],
                                 QuadraturePointTensor & field )
{
   WriteQuadratureLocalValues( quad_index, field_q, field, std::make_index_sequence< Dim >{} );
}

}

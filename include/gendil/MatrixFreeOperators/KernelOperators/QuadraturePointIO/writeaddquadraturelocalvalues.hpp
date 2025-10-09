// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

template < Integer Dim, typename QuadraturePointTensor, size_t... Is >
GENDIL_HOST_DEVICE
void WriteAddQuadratureLocalValues(
   TensorIndex< Dim > quad_index,
   const Real & field_q,
   QuadraturePointTensor & field,
   std::index_sequence< Is... > )
{
   AtomicAdd( field( quad_index[Is]... ) , field_q );
}

template < Integer Dim, typename QuadraturePointTensor >
GENDIL_HOST_DEVICE
void WriteAddQuadratureLocalValues(
   TensorIndex< Dim > quad_index,
   const Real & field_q,
   QuadraturePointTensor & field )
{
   // TODO: add check that we're acccessing inside the bounds
   WriteAddQuadratureLocalValues( quad_index, field_q, field, std::make_index_sequence< Dim >{} );
}

template < Integer Dim, typename QuadraturePointTensor, Integer NumComp, size_t... Is >
GENDIL_HOST_DEVICE
void WriteAddQuadratureLocalValues(
   TensorIndex< Dim > quad_index,
   const Real (& field_q)[ NumComp ],
   QuadraturePointTensor & field,
   std::index_sequence< Is... > )
{
   for (LocalIndex comp = 0; comp < NumComp; comp++)
   {
      AtomicAdd( field( quad_index[Is]..., comp ), field_q[comp] );
   }
}

template < Integer Dim, Integer NumComp, typename QuadraturePointTensor >
GENDIL_HOST_DEVICE
void WriteAddQuadratureLocalValues(
   TensorIndex< Dim > quad_index,
   const Real (& field_q)[ NumComp ],
   QuadraturePointTensor & field )
{
   // TODO: add check that we're acccessing inside the bounds
   WriteAddQuadratureLocalValues( quad_index, field_q, field, std::make_index_sequence< Dim >{} );
}

template < typename KernelContext, Integer Dim, typename LocalField, typename GlobalField >
GENDIL_HOST_DEVICE
void WriteAddQuadratureLocalValues(
   const KernelContext & kernel_conf,
   TensorIndex< Dim > quad_index,
   const LocalField & field_q,
   GlobalField & field )
{
   // TODO: add check that we're acccessing inside the bounds
   // using RegisterDimensions = typename KernelContext::RegisterDimensions;
   // TODO: Currently this is only used to accumulate in shared memory, ideally we want to check the type of GlobalField.
   WriteAddQuadratureLocalValues( quad_index, field_q, field );
}

template < typename KernelContext, Integer Dim, Integer NumComp, typename ... Tensors >
GENDIL_HOST_DEVICE
void WriteAddQuadratureLocalValues(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const Real (& field_q)[ NumComp ],
   std::tuple< Tensors ... > & field )
{
   constexpr Integer vdim = sizeof...( Tensors );
   ConstexprLoop< vdim >( [&]( auto i )
   {
      WriteAddQuadratureLocalValues( quad_index, field_q[i], std::get< i >( field ) );
   });
}

}

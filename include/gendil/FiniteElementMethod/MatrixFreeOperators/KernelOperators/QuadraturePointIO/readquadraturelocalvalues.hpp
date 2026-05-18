// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"
#include "gendil/Algebra/staticvector.hpp"
#include "gendil/Algebra/accessors.hpp"

namespace gendil {

template < Integer Dim, typename IntegrationRule, size_t... Is >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues( TensorIndex< Dim > quad_index,
                                const QuadraturePointValues< IntegrationRule > & field,
                                Real & field_q,
                                std::index_sequence< Is... > )
{
   field_q = field( quad_index[Is]... );
}

template < Integer Dim, typename IntegrationRule >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues( TensorIndex< Dim > quad_index,
                                const QuadraturePointValues< IntegrationRule > & field,
                                Real & field_q )
{
   // TODO: add check that we're acccessing inside the bounds
   ReadQuadratureLocalValues( quad_index, field, field_q, std::make_index_sequence< Dim >{} );
}

template < Integer Dim, typename Tensor, size_t... I >
GENDIL_HOST_DEVICE
Real ReadQuadratureLocalValues( const TensorIndex< Dim > & quad_index, const Tensor & field, std::index_sequence< I... > )
{
   return field( quad_index[I]... );
}

template < typename KernelContext, Integer Dim, typename Tensor >
GENDIL_HOST_DEVICE
Real ReadQuadratureLocalValues( const KernelContext & thread, const TensorIndex< Dim > & quad_index, const Tensor & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   return ReadQuadratureLocalValues( quad_index, field, RegisterDimensions{} );
}

/**
 * @brief Read scalar gradient at a quadrature point.
 *
 * For scalar gradient storage with shape [quad_points..., Dim],
 * reads all gradient components at the given quadrature point.
 *
 * Storage layout: field(qx, qy, qz, direction)
 * Returns: std::array<Real, Dim> where result[d] = ∂u/∂x_d
 *
 * Mirrors WriteQuadratureLocalValues convention:
 *   write: field(q..., d) = field_q[d]
 *   read:  field_q[d] = field(q..., d)
 *
 * @return std::array<Real, Dim> - gradient vector at quadrature point
 */
template < typename KernelContext, Integer Dim, typename Tensor, size_t... Is >
GENDIL_HOST_DEVICE
auto ReadQuadratureLocalGradients(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const Tensor & field,
   std::index_sequence< Is... > )
{
   std::array<Real, Dim> result;

   ConstexprLoop< Dim >( [&]( auto d )
   {
      result[d] = field( quad_index[Is]..., d );
   });

   return result;
}

template < typename KernelContext, Integer Dim, typename Tensor >
GENDIL_HOST_DEVICE
auto ReadQuadratureLocalGradients(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const Tensor & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   return ReadQuadratureLocalGradients( thread, quad_index, field, RegisterDimensions{} );
}

/**
 * @brief Read vector gradients from tuple storage.
 *
 * Vector gradient storage is tuple-per-component:
 *   tuple<component_0_gradient_storage, component_1_gradient_storage, ...>
 * where each component storage has shape [quad_points..., Dim].
 *
 * Reads all components and spatial directions at a given quadrature point.
 *
 * @return SerialRecursiveArray<Real, NumComp, Dim>
 *         Orientation: result(component, direction) = ∂u_component/∂x_direction
 *         Example for 3D vector field: result(0, 1) = ∂u_0/∂y
 */
template < typename KernelContext, Integer Dim, typename ... Tensors >
GENDIL_HOST_DEVICE
auto ReadQuadratureLocalGradients(
   const KernelContext & thread,
   const TensorIndex< Dim > & quad_index,
   const std::tuple< Tensors... > & field )
{
   constexpr Integer NumComp = sizeof...(Tensors);

   SerialRecursiveArray<Real, NumComp, Dim> result;

   ConstexprLoop< NumComp >( [&]( auto c )
   {
      auto grad_c = ReadQuadratureLocalGradients( thread, quad_index, std::get< c >( field ) );

      ConstexprLoop< Dim >( [&]( auto d )
      {
         result(c, d) = vector_access(grad_c, d);
      });
   });

   return result;
}

/**
 * @brief Helper: Read tuple of scalar QP values into vector.
 * Returns: SerialRecursiveArray<Real, NumComp>
 */
template < Integer NumComp, typename RegisterDimensions, Integer Dim, typename ... Tensors >
GENDIL_HOST_DEVICE
SerialRecursiveArray<Real, NumComp> ReadTupleScalarValues(
   const TensorIndex< Dim > & quad_index,
   const std::tuple< Tensors... > & field )
{
   SerialRecursiveArray<Real, NumComp> result;
   ConstexprLoop< NumComp >( [&]( auto i )
   {
      result(i) = ReadQuadratureLocalValues( quad_index, std::get< i >( field ), RegisterDimensions{} );
   });
   return result;
}

/**
 * @brief Helper: Read tuple of scalar QP gradients into matrix.
 * Returns: SerialRecursiveArray<Real, NumComp, GradDim> with orientation grad(component, direction)
 */
template < Integer NumComp, Integer GradDim, typename RegisterDimensions, Integer Dim, typename ... Tensors >
GENDIL_HOST_DEVICE
SerialRecursiveArray<Real, NumComp, GradDim> ReadTupleScalarGradients(
   const TensorIndex< Dim > & quad_index,
   const std::tuple< Tensors... > & field )
{
   SerialRecursiveArray<Real, NumComp, GradDim> result;

   ConstexprLoop< NumComp >( [&]( auto c )
   {
      auto grad_c = ReadQuadratureLocalValues( quad_index, std::get< c >( field ), RegisterDimensions{} );

      ConstexprLoop< GradDim >( [&]( auto d )
      {
         result(c, d) = vector_access(grad_c, d);
      });
   });

   return result;
}

/**
 * @brief Generic tuple ReadQuadratureLocalValues for vector fields (return-value version).
 *
 * Dispatches based on component read type:
 * - Scalar component → SerialRecursiveArray<Real, NumComp> (vector values)
 * - Vector component → SerialRecursiveArray<Real, NumComp, Dim> (vector gradients)
 */
template < typename KernelContext, Integer Dim, typename ... Tensors >
GENDIL_HOST_DEVICE
auto ReadQuadratureLocalValues( const KernelContext & thread, const TensorIndex< Dim > & quad_index, const std::tuple< Tensors... > & field )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   constexpr Integer NumComp = sizeof...( Tensors );

   // Read first component to determine type
   auto comp0_val = ReadQuadratureLocalValues( quad_index, std::get<0>( field ), RegisterDimensions{} );
   using Comp0Type = std::remove_cvref_t<decltype(comp0_val)>;

   if constexpr (std::is_same_v<Comp0Type, Real>)
   {
      // Tuple of scalar values → vector
      return ReadTupleScalarValues<NumComp, RegisterDimensions>(quad_index, field);
   }
   else if constexpr (static_extent_v<Comp0Type> != std::dynamic_extent)
   {
      // Tuple of gradient vectors → matrix
      constexpr Integer GradDim = static_extent_v<Comp0Type>;
      return ReadTupleScalarGradients<NumComp, GradDim, RegisterDimensions>(quad_index, field);
   }
   else
   {
      static_assert(always_false<Comp0Type>::value,
         "ReadQuadratureLocalValues(tuple): Component must be Real or static_vector.");
   }
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
void ReadQuadratureLocalValues( const KernelContext & thread, const TensorIndex< Dim > & quad_index, const Tensor & field, Real (& field_q)[NumComp] )
{
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;
   return ReadQuadratureLocalValues( thread, quad_index, field, field_q, RegisterDimensions{} );
}

template < Integer Dim, typename IntegrationRule, Integer NumComp, size_t... Is >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues( TensorIndex< Dim > quad_index,
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
void ReadQuadratureLocalValues( TensorIndex< Dim > quad_index,
                                const QuadraturePointValues< IntegrationRule, NumComp > & field,
                                Real (& field_q)[NumComp] )
{
   // TODO: add check that we're acccessing inside the bounds
   ReadQuadratureLocalValues( quad_index, field, field_q, std::make_index_sequence< Dim >{} );
}

template < Integer Dim, typename QuadPointTensor, size_t... Is>
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues( const TensorIndex< Dim > & quad_index,
                                const QuadPointTensor & field,
                                Real & field_q,
                                std::index_sequence< Is... >)
{
   field_q = field( quad_index[Is]... );
}

template < Integer Dim, typename QuadPointTensor, size_t... Is>
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues( const TensorIndex< Dim > & quad_index,
                                const QuadPointTensor & field,
                                Real & field_q )
{
   ReadQuadratureLocalValues( quad_index, field, field_q, std::make_index_sequence< Dim >{} );
}

template < Integer Dim, typename QuadPointTensor, Integer NumComp, size_t ... Is >
GENDIL_HOST_DEVICE
void ReadQuadratureLocalValues( const TensorIndex< Dim > & quad_index,
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
void ReadQuadratureLocalValues( const TensorIndex< Dim > & quad_index,
                                const QuadPointTensor & field,
                                Real field_q[NumComp] )
{
   ReadQuadratureLocalValues( quad_index, field, field_q, std::make_index_sequence< Dim >{} );
}

}

// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/doflayout.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/dofloop.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/Utilities/View/Layouts/orientedlayout.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/FiniteElementMethod/finiteelementorders.hpp"
#include "gendil/FiniteElementMethod/restriction.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/MathHelperFunctions/sum.hpp"

#include <array>
#include <tuple>
#include <utility>

namespace gendil {

/**
 * @brief Utility function to transform a vector into a structured tensor.
 * 
 * @tparam FiniteElementSpace The finite element space type
 * @tparam Orders The list of polynomial orders for each dimension.
 * @tparam T The type of data.
 * @param finite_element_space The finite element space.
 * @param data The data to represent as a tensor.
 * @return auto The tensor sized according to the finite element space.
 */
template <
   typename FiniteElementSpace,
   Integer ... Orders,
   typename T >
auto MakeTensor(
   FiniteElementSpace const & finite_element_space,
   T * data,
   FiniteElementOrders< Orders... > )
{
   const GlobalIndex num_elem = finite_element_space.GetNumberOfFiniteElements();
   return MakeFIFOView( data, ((GlobalIndex)Orders + 1)..., num_elem );
}

/**
 * @brief Utility function to transform a vector of deegrees of freedom into a structured tensor of degrees of freedom.
 * 
 * @note automatically size the tensor with numbers of degrees of freedom for each dimension.
 * 
 * @tparam FiniteElementSpace The finite element space type
 * @tparam T The type of data.
 * @param finite_element_space The finite element space.
 * @param data The data to represent as a tensor.
 * @return auto The tensor sized according to the finite element space.
 */
template < typename FiniteElementSpace,
         typename T >
auto MakeTensor(
   FiniteElementSpace const & finite_element_space,
   T * data )
{
   // Assumes tensor elements
   using orders = typename FiniteElementSpace::finite_element_type::shape_functions::orders;
   return MakeTensor( finite_element_space, data, orders{} );
}

template <
   typename T,
   size_t ... dof_shapes >
auto MakeTensor(
   T * data,
   GlobalIndex num_elements,
   std::index_sequence< dof_shapes... > )
{
   return MakeFIFOView( data, (GlobalIndex)dof_shapes..., num_elements );
}

template <
   typename T,
   size_t ... dof_shapes >
auto MakeIndirectedTensor(
   T * data,
   const HostDevicePointer< const int > & indices,
   GlobalIndex num_elements,
   std::index_sequence< dof_shapes... > )
{
   return MakeIndirectedFIFOView(
      data,
      indices,
      static_cast< GlobalIndex >( dof_shapes )...,
      num_elements );
}

/**
 * @brief Utility function to transform a vector into a structured tensor.
 * 
 * @tparam FiniteElementSpace The finite element space type
 * @tparam Orders The list of polynomial orders for each dimension.
 * @tparam T The type of data.
 * @param finite_element_space The finite element space.
 * @param data The data to represent as a tensor.
 * @return auto The tensor sized according to the finite element space.
 */
template < typename FiniteElementSpace,
         Integer ... Orders,
         typename T >
auto MakeIndirectedTensor(
   FiniteElementSpace const & finite_element_space,
   T data,
   FiniteElementOrders< Orders... > )
{
   const Integer num_elem = finite_element_space.GetNumberOfFiniteElements();
   return MakeIndirectedFIFOView( data, finite_element_space.restriction.indices, (Orders + 1)..., num_elem );
}

/**
 * @brief Utility function to transform a vector of deegrees of freedom into a structured tensor of degrees of freedom.
 * 
 * @note automatically size the tensor with numbers of degrees of freedom for each dimension.
 * 
 * @tparam FiniteElementSpace The finite element space type
 * @tparam T The type of data.
 * @param finite_element_space The finite element space.
 * @param data The data to represent as a tensor.
 * @return auto The tensor sized according to the finite element space.
 */
template < typename FiniteElementSpace,
         typename T >
auto MakeIndirectedTensor(
   FiniteElementSpace const & finite_element_space,
   T data )
{
   // Assumes tensor elements
   using orders = typename FiniteElementSpace::finite_element_type::shape_functions::orders;
   return MakeIndirectedTensor( finite_element_space, data, orders{} );
}

template < typename FiniteElementSpace, typename T >
auto MakeScalarElementTensorView(
   const FiniteElementSpace & finite_element_space,
   T * data )
{
   using Restriction = typename FiniteElementSpace::restriction_type;
   if constexpr ( std::is_same_v< Restriction, L2Restriction > )
   {
      const GlobalIndex dof_shift = finite_element_space.restriction.shift;
      return MakeTensor( finite_element_space, data + dof_shift );
   }
   else if constexpr ( std::is_same_v< Restriction, H1Restriction > )
   {
      return MakeIndirectedTensor( finite_element_space, data );
   }
   else if constexpr ( is_tensor_product_restriction_v< Restriction > )
   {
      using ShapeFunctions =
         typename FiniteElementSpace::finite_element_type::shape_functions;
      using ProductDofShape = finite_element_dof_shape_t< ShapeFunctions >;
      static_assert(
         !is_vector_shape_functions_v< ShapeFunctions >,
         "TensorProductRestriction v1 supports scalar finite element spaces only." );
      return MakeView(
         data,
         MakeTensorProductLayout< ProductDofShape >(
            finite_element_space.restriction ) );
   }
   else
   {
      static_assert(
         dependent_false_v< Restriction >,
         "MakeScalarElementTensorView supports only scalar L2Restriction, H1Restriction, and TensorProductRestriction." );
   }
}

template < typename FiniteElementSpace, typename T, size_t ... v_dims >
auto MakeVectorElementTensorView(
   const FiniteElementSpace & finite_element_space,
   T * data,
   std::index_sequence< v_dims... > )
{
   using ShapeFunctions =
      typename FiniteElementSpace::finite_element_type::shape_functions;
   using Restriction = typename FiniteElementSpace::restriction_type;
   const GlobalIndex num_elements = finite_element_space.GetNumberOfFiniteElements();
   using dof_shape = typename ShapeFunctions::dof_shape;

   if constexpr ( std::is_same_v< Restriction, L2Restriction > )
   {
      const GlobalIndex dof_shift = finite_element_space.restriction.shift;
      return std::make_tuple(
         MakeTensor(
            data + dof_shift + VectorOffset(
               dof_shape{},
               num_elements,
               std::make_index_sequence< v_dims >{} ),
            num_elements,
            std::tuple_element_t< v_dims, dof_shape >{} )... );
   }
   else if constexpr ( is_vector_h1_restriction_v< Restriction > )
   {
      static_assert(
         Restriction::num_comp == ShapeFunctions::vector_dim,
         "VectorH1Restriction<NComp> must match the vector finite element component count." );
      static_assert(
         VectorComponentDofShapesMatchFirst< ShapeFunctions >(),
         "VectorH1Restriction currently requires identical scalar component DoF shapes." );

      return std::make_tuple(
         MakeIndirectedTensor(
            data + static_cast< GlobalIndex >( v_dims ) *
               static_cast< GlobalIndex >( finite_element_space.restriction.scalar_num_dofs ),
            finite_element_space.restriction.indices,
            num_elements,
            std::tuple_element_t< v_dims, dof_shape >{} )... );
   }
   else
   {
      static_assert(
         dependent_false_v< Restriction >,
         "MakeVectorElementTensorView supports only L2Restriction and VectorH1Restriction." );
   }
}

template < typename FiniteElementSpace, typename T >
auto MakeVectorElementTensorView(
   const FiniteElementSpace & finite_element_space,
   T * data )
{
   constexpr Integer v_dim = FiniteElementSpace::finite_element_type::shape_functions::vector_dim;
   return MakeVectorElementTensorView( finite_element_space, data, std::make_index_sequence< v_dim >{} );
}
/**
 * @brief Utility function to transform a vector of deegrees of freedom into a tensor view of degrees of freedom.
 * 
 * @note automatically size the tensor with numbers of degrees of freedom for each dimension.
 * 
 * @tparam FiniteElementSpace The finite element space type
 * @tparam T The type of data.
 * @param finite_element_space The finite element space.
 * @param data The data to represent as a tensor.
 * @return auto The tensor sized according to the finite element space.
 */
template < typename FiniteElementSpace, typename T >
auto MakeElementTensorView(
   const FiniteElementSpace & finite_element_space,
   T * data )
{
   if constexpr ( is_vector_shape_functions_v< typename FiniteElementSpace::finite_element_type::shape_functions > )
   {
      return MakeVectorElementTensorView( finite_element_space, data );
   }
   else
   {
      return MakeScalarElementTensorView( finite_element_space, data );
   }
}

template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeReadOnlyElementTensorView(
   const FiniteElementSpace & finite_element_space,
   const Vector & data )
{
   if constexpr ( is_host_configuration_v< KernelPolicy > )
   {
      return MakeElementTensorView( finite_element_space, data.ReadHostData() );
   }
   else
   {
      return MakeElementTensorView( finite_element_space, data.ReadDeviceData() );
   }
}

template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeWriteOnlyElementTensorView(
   const FiniteElementSpace & finite_element_space,
   Vector & data )
{
   if constexpr ( is_host_configuration_v< KernelPolicy > )
   {
      return MakeElementTensorView( finite_element_space, data.WriteHostData() );
   }
   else
   {
      return MakeElementTensorView( finite_element_space, data.WriteDeviceData() );
   }
}

template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeReadWriteElementTensorView(
   const FiniteElementSpace & finite_element_space,
   Vector & data )
{
   if constexpr ( is_host_configuration_v< KernelPolicy > )
   {
      return MakeElementTensorView( finite_element_space, data.ReadWriteHostData() );
   }
   else
   {
      return MakeElementTensorView( finite_element_space, data.ReadWriteDeviceData() );
   }
}

#ifdef GENDIL_USE_MFEM
template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeReadOnlyElementTensorView(
   const FiniteElementSpace & finite_element_space,
   const mfem::Vector & data )
{
   if constexpr ( is_host_configuration_v< KernelPolicy > )
   {
      return MakeElementTensorView( finite_element_space, data.HostRead() );
   }
   else
   {
      return MakeElementTensorView( finite_element_space, data.Read() );
   }
}

template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeWriteOnlyElementTensorView(
   const FiniteElementSpace & finite_element_space,
   mfem::Vector & data )
{
   if constexpr ( is_host_configuration_v< KernelPolicy > )
   {
      return MakeElementTensorView( finite_element_space, data.HostWrite() );
   }
   else
   {
      return MakeElementTensorView( finite_element_space, data.Write() );
   }
}

template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeReadWriteElementTensorView(
   const FiniteElementSpace & finite_element_space,
   mfem::Vector & data )
{
   if constexpr ( is_host_configuration_v< KernelPolicy > )
   {
      return MakeElementTensorView( finite_element_space, data.HostReadWrite() );
   }
   else
   {
      return MakeElementTensorView( finite_element_space, data.ReadWrite() );
   }
}
#endif // GENDIL_USE_MFEM

}

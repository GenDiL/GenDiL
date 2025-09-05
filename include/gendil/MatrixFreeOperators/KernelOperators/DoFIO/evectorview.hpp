// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/LoopHelpers/dofloop.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/Utilities/View/Layouts/orientedlayout.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/FiniteElementMethod/finiteelementorders.hpp"
#include "gendil/FiniteElementMethod/restriction.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/vector.hpp"
#include "gendil/Utilities/MathHelperFunctions/sum.hpp"

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
auto MakeScalarEVectorView(
   const FiniteElementSpace & finite_element_space,
   T * data )
{
   if constexpr ( std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction > )
   {
      return MakeTensor( finite_element_space, data );
   }
   else // H1Restriction
   {
      return MakeIndirectedTensor( finite_element_space, data );
   }
}

template <
   typename Tuple,
   size_t ... I >
size_t VectorOffset( Tuple dof_shapes, GlobalIndex num_elements, std::index_sequence< I ... > )
{
   return Sum( ( num_elements * Product( std::tuple_element_t< I, Tuple >{} ) )... );
}

template < typename FiniteElementSpace, typename T, size_t ... v_dims >
auto MakeVectorEVectorView(
   const FiniteElementSpace & finite_element_space,
   T * data,
   std::index_sequence< v_dims... > )
{
   const GlobalIndex num_elements = finite_element_space.GetNumberOfFiniteElements();
   using dof_shape = typename FiniteElementSpace::finite_element_type::shape_functions::dof_shape;
   return std::make_tuple(
      MakeTensor(
         data + VectorOffset( dof_shape{}, num_elements, std::make_index_sequence< v_dims >{} ),
         num_elements,
         std::tuple_element_t< v_dims, dof_shape >{}
      )...
   );
}

template < typename FiniteElementSpace, typename T >
auto MakeVectorEVectorView(
   const FiniteElementSpace & finite_element_space,
   T * data )
{
   constexpr Integer v_dim = FiniteElementSpace::finite_element_type::shape_functions::vector_dim;
   return MakeVectorEVectorView( finite_element_space, data, std::make_index_sequence< v_dim >{} );
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
auto MakeEVectorView(
   const FiniteElementSpace & finite_element_space,
   T * data )
{
      const GlobalIndex dof_shift = finite_element_space.restriction.shift;
   if constexpr ( is_vector_shape_functions_v< typename FiniteElementSpace::finite_element_type::shape_functions > )
   {
      return MakeVectorEVectorView( finite_element_space, data + dof_shift);
   }
   else
   {
      return MakeScalarEVectorView( finite_element_space, data + dof_shift);
   }
}

template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeReadOnlyEVectorView(
   const FiniteElementSpace & finite_element_space,
   const Vector & data )
{
   if constexpr ( is_serial_v< KernelPolicy > )
   {
      return MakeEVectorView( finite_element_space, data.ReadHostData() );
   }
   else
   {
      return MakeEVectorView( finite_element_space, data.ReadDeviceData() );
   }
}

template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeWriteOnlyEVectorView(
   const FiniteElementSpace & finite_element_space,
   Vector & data )
{
   if constexpr ( is_serial_v< KernelPolicy > )
   {
      return MakeEVectorView( finite_element_space, data.WriteHostData() );
   }
   else
   {
      return MakeEVectorView( finite_element_space, data.WriteDeviceData() );
   }
}

template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeReadWriteEVectorView(
   const FiniteElementSpace & finite_element_space,
   Vector & data )
{
   if constexpr ( is_serial_v< KernelPolicy > )
   {
      return MakeEVectorView( finite_element_space, data.ReadWriteHostData() );
   }
   else
   {
      return MakeEVectorView( finite_element_space, data.ReadWriteDeviceData() );
   }
}

#ifdef GENDIL_USE_MFEM
template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeReadOnlyEVectorView(
   const FiniteElementSpace & finite_element_space,
   const mfem::Vector & data )
{
   if constexpr ( is_serial_v< KernelPolicy > )
   {
      return MakeEVectorView( finite_element_space, data.HostRead() );
   }
   else
   {
      return MakeEVectorView( finite_element_space, data.Read() );
   }
}

template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeWriteOnlyEVectorView(
   const FiniteElementSpace & finite_element_space,
   mfem::Vector & data )
{
   if constexpr ( is_serial_v< KernelPolicy > )
   {
      return MakeEVectorView( finite_element_space, data.HostWrite() );
   }
   else
   {
      return MakeEVectorView( finite_element_space, data.Write() );
   }
}

template < typename KernelPolicy, typename FiniteElementSpace >
auto MakeReadWriteEVectorView(
   const FiniteElementSpace & finite_element_space,
   mfem::Vector & data )
{
   if constexpr ( is_serial_v< KernelPolicy > )
   {
      return MakeEVectorView( finite_element_space, data.HostReadWrite() );
   }
   else
   {
      return MakeEVectorView( finite_element_space, data.ReadWrite() );
   }
}
#endif // GENDIL_USE_MFEM

}
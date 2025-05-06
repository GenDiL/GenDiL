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
   if constexpr ( std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction > )
   {
      return MakeTensor( finite_element_space, data );
   }
   else // H1Restriction
   {
      return MakeIndirectedTensor( finite_element_space, data );
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
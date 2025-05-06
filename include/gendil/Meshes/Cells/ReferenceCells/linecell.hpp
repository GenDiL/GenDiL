// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/doftoquad.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/GLLshapefunctions.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

namespace gendil {

/**
 * @brief A simple structure representing an isoparametric segment cell.
 * 
 * @tparam D1D The number of degrees-of-freedom in the isoparametric transformation.
 */
template < int D1D >
struct LineCell
{
   static constexpr Integer Dim = 1;

   Real nodes[D1D];

   using physical_coordinates = std::array< Real, Dim >;
   using jacobian = std::array< std::array< Real, Dim >, Dim >;

   using basis = std::tuple< GaussLobattoLegendreShapeFunctions<D1D-1> >;
   template < typename IntRule >
   using QuadData =  std::tuple<
                        DofToQuad<
                           std::tuple_element_t<0, basis>,
                           std::tuple_element_t<0, typename IntRule::points::points_1d_tuple >
                        >
                     >;

   GENDIL_HOST_DEVICE
   LineCell( const StridedView<1, const Real> & mesh_nodes,
             const HostDeviceStridedView<2, const int> & restriction,
             const GlobalIndex element_index )
   {
      for (LocalIndex dof = 0; dof < D1D; dof++)
      {
         const GlobalIndex dof_index = restriction( dof, element_index );
         nodes[ dof ] = mesh_nodes( dof_index );
      }
   }

   template < typename DofToQuad >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( TensorIndex< 1 > const & quad_index,
                              const DofToQuad & basis,
                              physical_coordinates & X,
                              jacobian & J_mesh ) const
   {
      GetValuesAndJacobian( quad_index[0], basis, X, J_mesh );
   }

   template < typename DofToQuad >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( const LocalIndex qx,
                              const DofToQuad & basis,
                              physical_coordinates & X,
                              jacobian & J_mesh ) const
   {
      for (LocalIndex i = 0; i < Dim; i++)
      {
         X[ i ] = 0.0;
         for (LocalIndex j = 0; j < Dim; j++)
         {
            J_mesh[ i ][ j ] = 0.0;
         }
      }
      for (LocalIndex dx = 0; dx < D1D; ++dx)
      {
         const Real bx = std::get<0>(basis).values(qx,dx);
         const Real gx = std::get<0>(basis).gradients(qx,dx);
         Real x = nodes[ dx ];
         const Real b = bx;
         X[0] += b * x;
         const Real Gx = gx;
         J_mesh[0][0] += Gx * x;
      }
   }
};

}
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
 * @brief A simple structure representing an isoparametric quadrilateral cell.
 * 
 * @tparam D1D The number of degrees-of-freedom per dimension in the isoparametric transformation.
 */
template < int D1D >
struct QuadCell
{
   static constexpr Integer Dim = 2;

   Real nodes[ D1D ][ D1D ][ Dim ];

   using basis = std::tuple< GaussLobattoLegendreShapeFunctions<D1D-1>, GaussLobattoLegendreShapeFunctions<D1D-1> >;
   using physical_coordinates = std::array< Real, Dim >;
   using jacobian = std::array< std::array< Real, Dim >, Dim >;

   template < typename IntRule >
   using QuadData =  std::tuple<
                        DofToQuad<
                           std::tuple_element_t<0, basis>,
                           std::tuple_element_t<0, typename IntRule::points::points_1d_tuple >
                        >,
                        DofToQuad<
                           std::tuple_element_t<1, basis>,
                           std::tuple_element_t<1, typename IntRule::points::points_1d_tuple >
                        >
                     >;

   GENDIL_HOST_DEVICE
   QuadCell( const StridedView<2, const Real> & mesh_nodes,
             const HostDeviceStridedView<3, const int> & restriction,
             const GlobalIndex element_index )
   {
      for (LocalIndex j = 0; j < D1D; j++)
      {
         for (LocalIndex i = 0; i < D1D; i++)
         {
            const GlobalIndex dof_index = restriction( i, j, element_index );
            for (LocalIndex comp = 0; comp < Dim; comp++)
            {
               nodes[ i ][ j ][ comp ] = mesh_nodes( comp, dof_index ); // FIXME: is that correct? byNodes, byVDim?
            }
         }
      }
   }

   template < typename DofToQuad >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( TensorIndex< 2 > const & quad_index,
                              const DofToQuad & basis,
                              physical_coordinates & X,
                              jacobian & J_mesh ) const
   {
      GetValuesAndJacobian( quad_index[0], quad_index[1], basis, X, J_mesh );
   }

   template < typename DofToQuad >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( const LocalIndex qx,
                              const LocalIndex qy,
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
      for (LocalIndex dy = 0; dy < D1D; ++dy)
      {
         const Real by = std::get<1>(basis).values(qy,dy);
         const Real gy = std::get<1>(basis).gradients(qy,dy);
         for (LocalIndex dx = 0; dx < D1D; ++dx)
         {
            const Real bx = std::get<0>(basis).values(qx,dx);
            const Real gx = std::get<0>(basis).gradients(qx,dx);
            Real x[2] = { nodes[ dx ][ dy ][ 0 ],
                           nodes[ dx ][ dy ][ 1 ] };
            const Real b = bx * by;
            X[0] += b * x[0];
            X[1] += b * x[1];
            const Real Gx = gx * by;
            const Real Gy = bx * gy;
            J_mesh[0][0] += Gx * x[0];
            J_mesh[1][0] += Gy * x[0];
            J_mesh[0][1] += Gx * x[1];
            J_mesh[1][1] += Gy * x[1];
         }
      }
   }
};

}
// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/doftoquad.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/GLLshapefunctions.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

namespace gendil {

/**
 * @brief A simple structure representing an isoparametric hexahedron cell.
 * 
 * @tparam D1D The number of degrees-of-freedom per dimension in the isoparametric transformation.
 */
template < int D1D >
struct HexCell
{
   static constexpr Integer Dim = 3;

   Real nodes[D1D][D1D][D1D][Dim];

   using basis =  std::tuple<
                     GaussLobattoLegendreShapeFunctions<D1D-1>,
                     GaussLobattoLegendreShapeFunctions<D1D-1>,
                     GaussLobattoLegendreShapeFunctions<D1D-1>
                  >;
   using physical_coordinates = std::array< Real, Dim >;
   using jacobian = std::array< std::array< Real, Dim >, Dim >;

   template < typename IntRule >
   using QuadData =  std::tuple<
                        CachedDofToQuad<
                           std::tuple_element_t<0, basis>,
                           std::tuple_element_t<0, typename IntRule::points::points_1d_tuple >
                        >,
                        CachedDofToQuad<
                           std::tuple_element_t<1, basis>,
                           std::tuple_element_t<1, typename IntRule::points::points_1d_tuple >
                        >,
                        CachedDofToQuad<
                           std::tuple_element_t<2, basis>,
                           std::tuple_element_t<2, typename IntRule::points::points_1d_tuple >
                        >
                     >;

   GENDIL_HOST_DEVICE
   HexCell( const StridedView<2, const Real> & mesh_nodes,
            const HostDeviceStridedView<4, const int> & restriction,
            const GlobalIndex element_index )
   {
      for (LocalIndex k = 0; k < D1D; k++)
      {
         for (LocalIndex j = 0; j < D1D; j++)
         {
            for (LocalIndex i = 0; i < D1D; i++)
            {
               const GlobalIndex dof_index = restriction( i, j, k, element_index );
               for (GlobalIndex comp = 0; comp < Dim; comp++)
               {
                  nodes[ i ][ j ][ k ][ comp ] = mesh_nodes( comp, dof_index ); // FIXME: is that correct? byNodes, byVDim?
               }
            }
         }
      }
   }

   template < typename QuadData >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( TensorIndex< 3 > const & quad_index,
                              QuadData const & quad_data,
                              physical_coordinates & X,
                              jacobian & J_mesh ) const
   {
      GetValuesAndJacobian( quad_index[0], quad_index[1], quad_index[2], quad_data, X, J_mesh );
   }

   template < typename QuadData >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( const LocalIndex qx,
                              const LocalIndex qy,
                              const LocalIndex qz,
                              QuadData const & quad_data,
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
      for (LocalIndex dz = 0; dz < D1D; ++dz)
      {
         const Real bz = std::get<2>(quad_data).values(qz,dz);
         const Real gz = std::get<2>(quad_data).gradients(qz,dz);
         for (LocalIndex dy = 0; dy < D1D; ++dy)
         {
            const Real by = std::get<1>(quad_data).values(qy,dy);
            const Real gy = std::get<1>(quad_data).gradients(qy,dy);
            for (LocalIndex dx = 0; dx < D1D; ++dx)
            {
               const Real bx = std::get<0>(quad_data).values(qx,dx);
               const Real gx = std::get<0>(quad_data).gradients(qx,dx);
               Real x[3] = { nodes[ dx ][ dy ][ dz ][ 0 ],
                              nodes[ dx ][ dy ][ dz ][ 1 ],
                              nodes[ dx ][ dy ][ dz ][ 2 ] };
               const Real b = bx * by * bz;
               X[0] += b * x[0];
               X[1] += b * x[1];
               X[2] += b * x[2];
               const Real Gx = gx * by * bz;
               const Real Gy = bx * gy * bz;
               const Real Gz = bx * by * gz;
               J_mesh[0][0] += Gx * x[0];
               J_mesh[1][0] += Gy * x[0];
               J_mesh[2][0] += Gz * x[0];
               J_mesh[0][1] += Gx * x[1];
               J_mesh[1][1] += Gy * x[1];
               J_mesh[2][1] += Gz * x[1];
               J_mesh[0][2] += Gx * x[2];
               J_mesh[1][2] += Gy * x[2];
               J_mesh[2][2] += Gz * x[2];
            }
         }
      }
   }

   GENDIL_HOST_DEVICE
   jacobian ComputeJacobian( const Point< Dim > & ref_point ) const
   {
      jacobian J_mesh{};
      for (LocalIndex dz = 0; dz < D1D; ++dz)
      {
         const Real bz = std::tuple_element<2,basis>::ComputeValue(dz, ref_point[2]);
         const Real gz = std::tuple_element<2,basis>::ComputeGradientValue(dz, ref_point[2]);
         for (LocalIndex dy = 0; dy < D1D; ++dy)
         {
            const Real by = std::tuple_element<1,basis>::ComputeValue(dy, ref_point[1]);
            const Real gy = std::tuple_element<1,basis>::ComputeGradientValue(dy, ref_point[1]);
            for (LocalIndex dx = 0; dx < D1D; ++dx)
            {
               const Real bx = std::tuple_element<0,basis>::ComputeValue(dx, ref_point[0]);
               const Real gx = std::tuple_element<0,basis>::ComputeGradientValue(dx, ref_point[0]);
               Real x[3] = { nodes[ dx ][ dy ][ dz ][ 0 ],
                              nodes[ dx ][ dy ][ dz ][ 1 ],
                              nodes[ dx ][ dy ][ dz ][ 2 ] };
               const Real Gx = gx * by * bz;
               const Real Gy = bx * gy * bz;
               const Real Gz = bx * by * gz;
               J_mesh[0][0] += Gx * x[0];
               J_mesh[1][0] += Gy * x[0];
               J_mesh[2][0] += Gz * x[0];
               J_mesh[0][1] += Gx * x[1];
               J_mesh[1][1] += Gy * x[1];
               J_mesh[2][1] += Gz * x[1];
               J_mesh[0][2] += Gx * x[2];
               J_mesh[1][2] += Gy * x[2];
               J_mesh[2][2] += Gz * x[2];
            }
         }
      }
      return J_mesh;
   }
};

template <int D1D>
GENDIL_HOST_DEVICE
void ApplyOrientationToCell(const Permutation<3>& orientation,
                            HexCell<D1D>& cell)
{
   constexpr Integer v_dim = 3;
   constexpr size_t data_size = D1D * D1D * D1D;

   std::array<size_t, 3> dofs_sizes = {D1D, D1D, D1D};

   ConstexprLoop<v_dim>([&](auto i)
   {
      Real data[data_size];

      auto oriented_view =
         MakeOrientedView(data, dofs_sizes, orientation);

      auto reference_view =
         MakeFIFOView(data, dofs_sizes);

      auto cell_comp_value =
         [&](size_t x, size_t y, size_t z) -> Real&
         {
            return cell.nodes[x][y][z][i];
         };

      // Read native cell component and write it into oriented/canonical storage.
      for (size_t x = 0; x < D1D; ++x)
      {
         for (size_t y = 0; y < D1D; ++y)
         {
            for (size_t z = 0; z < D1D; ++z)
            {
               oriented_view(x, y, z) = cell_comp_value(x, y, z);
            }
         }
      }

      // Copy canonical/reference-oriented data back into the cell.
      for (size_t x = 0; x < D1D; ++x)
      {
         for (size_t y = 0; y < D1D; ++y)
         {
            for (size_t z = 0; z < D1D; ++z)
            {
               cell_comp_value(x, y, z) = reference_view(x, y, z);
            }
         }
      }
   });
}

}

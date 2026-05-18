// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#include "getoffset.hpp"
#include "gendil/Algebra/staticvector.hpp"
#include "gendil/Algebra/accessors.hpp"

namespace gendil {

template < size_t offset, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void ApplyMapping( const Real & inv_J, VecType & Gu )
{
   Gu[offset] = inv_J * Gu[offset];
}

template < size_t offset, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void ApplyMapping( const Real (& inv_J)[1][1], VecType & Gu )
{
   Gu[offset] = inv_J[0][0] * Gu[offset];
}

template < size_t offset, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void ApplyMapping( const Real (& inv_J)[2][2], VecType & Gu )
{
   const Real x = inv_J[0][0] * Gu[offset+0] + inv_J[0][1] * Gu[offset+1];
   const Real y = inv_J[1][0] * Gu[offset+0] + inv_J[1][1] * Gu[offset+1];
   Gu[offset+0] = x;
   Gu[offset+1] = y;
}

template < size_t offset, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void ApplyMapping( const Real (& inv_J)[3][3], VecType & Gu )
{
   const Real x = inv_J[0][0] * Gu[offset+0] + inv_J[0][1] * Gu[offset+1] + inv_J[0][2] * Gu[offset+2];
   const Real y = inv_J[1][0] * Gu[offset+0] + inv_J[1][1] * Gu[offset+1] + inv_J[1][2] * Gu[offset+2];
   const Real z = inv_J[2][0] * Gu[offset+0] + inv_J[2][1] * Gu[offset+1] + inv_J[2][2] * Gu[offset+2];
   Gu[offset+0] = x;
   Gu[offset+1] = y;
   Gu[offset+2] = z;
}

template < size_t Dim, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void ApplyMapping( const Real (& inv_J)[Dim][Dim], VecType & Gu )
{
   ApplyMapping<0>( inv_J, Gu );
}

template < size_t offset, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void ApplyMapping( const std::array< std::array< Real, 1 >, 1 > & inv_J, VecType & Gu )
{
   Gu[offset] = inv_J[0][0] * Gu[offset];
}

template < size_t offset, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void ApplyMapping( const std::array< std::array< Real, 2 >, 2 > & inv_J, VecType & Gu )
{
   const Real x = inv_J[0][0] * Gu[offset+0] + inv_J[0][1] * Gu[offset+1];
   const Real y = inv_J[1][0] * Gu[offset+0] + inv_J[1][1] * Gu[offset+1];
   Gu[offset+0] = x;
   Gu[offset+1] = y;
}

template < size_t offset, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void ApplyMapping( const std::array< std::array< Real, 3 >, 3 > & inv_J, VecType & Gu )
{
   const Real x = inv_J[0][0] * Gu[offset+0] + inv_J[0][1] * Gu[offset+1] + inv_J[0][2] * Gu[offset+2];
   const Real y = inv_J[1][0] * Gu[offset+0] + inv_J[1][1] * Gu[offset+1] + inv_J[1][2] * Gu[offset+2];
   const Real z = inv_J[2][0] * Gu[offset+0] + inv_J[2][1] * Gu[offset+1] + inv_J[2][2] * Gu[offset+2];
   Gu[offset+0] = x;
   Gu[offset+1] = y;
   Gu[offset+2] = z;
}

template < size_t Dim, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void ApplyMapping( const std::array< std::array< Real, Dim >, Dim > & inv_J, VecType & Gu )
{
   ApplyMapping<0>( inv_J, Gu );
}

template < size_t offset = 0, size_t DimJ, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void ApplyMapping( const std::array< Real, DimJ > & inv_J, VecType & Gu )
{
   ConstexprLoop< DimJ >( [&] ( auto i )
   {
      Gu[offset+i] = inv_J[i] * Gu[offset+i];
   });
}

// template < size_t offset = 0, Integer Dim, static_vector<Real, Dim> VecType, typename... MatrixTypes >
// template < size_t offset = 0, typename... MatrixTypes, class VecType >
// requires static_vector<VecType, Real, static_extent_v<std::remove_cvref_t<VecType>>>
template < size_t offset = 0, typename... MatrixTypes, static_vector<Real> VecType >
GENDIL_HOST_DEVICE
void ApplyMapping( const std::tuple< MatrixTypes... > & inv_J,
                   VecType & Gu )
{
   using MatrixTuple = std::tuple< MatrixTypes... >;

   ConstexprLoop< sizeof...(MatrixTypes) >(
      [&] ( auto i )
      {
         constexpr size_t new_offset = offset + GetOffset< MatrixTuple, i >::value;
         ApplyMapping< new_offset >( std::get< i >( inv_J ), Gu );
      }
   );
}

/**
 * @brief ApplyMapping for vector field gradients with tuple inv_J.
 *
 * Vector gradient tensor: SerialRecursiveArray<Real, NumComp, Dim>
 * Orientation: grad(component, direction) = ∂u_component/∂x_direction
 *
 * Geometric mapping: inv_J_mesh is a tuple based on spatial dimensions, NOT vector components.
 * All vector components share the same geometric mapping (reference-to-physical).
 *
 * Implementation: Applies scalar ApplyMapping componentwise.
 * For each component c:
 *   - Extract gradient row: grad_c[d] = grad(c, d)
 *   - Apply scalar mapping: grad_c' = inv_J * grad_c (reference gradient → physical gradient)
 *   - Write back: grad(c, d) = grad_c'[d]
 */
template < typename... MatrixTypes, Integer NumComp, Integer Dim >
GENDIL_HOST_DEVICE
void ApplyMapping(
   const std::tuple< MatrixTypes... > & inv_J_tuple,
   SerialRecursiveArray<Real, NumComp, Dim> & grad )
{
   ConstexprLoop< NumComp >( [&]( auto c )
   {
      // Extract component gradient into std::array for scalar mapping
      std::array<Real, Dim> grad_c;
      ConstexprLoop< Dim >( [&]( auto d )
      {
         grad_c[d] = grad(c, d);
      });

      // Apply scalar mapping using the tuple-based path
      // inv_J_tuple represents spatial dimension mapping, shared by all components
      ApplyMapping<0>( inv_J_tuple, grad_c );

      // Write mapped gradient back
      ConstexprLoop< Dim >( [&]( auto d )
      {
         grad(c, d) = grad_c[d];
      });
   });
}

/**
 * @brief ApplyMapping for vector field gradients with std::array inv_J (diagonal Jacobian).
 *
 * CartesianMesh uses diagonal Jacobian represented as std::array<Real, Dim>.
 * This overload handles vector field gradients with diagonal geometric mapping.
 *
 * Vector gradient tensor: SerialRecursiveArray<Real, NumComp, Dim>
 * Orientation: grad(component, direction) = ∂u_component/∂x_direction
 *
 * Geometric mapping: inv_J is diagonal std::array<Real, Dim>, shared by all vector components.
 * All vector components share the same geometric mapping (reference-to-physical).
 *
 * Implementation: Applies component-wise scaling by diagonal elements.
 * For each component c and direction d:
 *   grad(c, d) *= inv_J[d]
 */
template < Integer Dim, Integer NumComp >
GENDIL_HOST_DEVICE
void ApplyMapping(
   const std::array< Real, Dim > & inv_J,
   SerialRecursiveArray<Real, NumComp, Dim> & grad )
{
   // For diagonal Jacobian, each gradient direction is scaled independently
   ConstexprLoop< NumComp >( [&]( auto c )
   {
      ConstexprLoop< Dim >( [&]( auto d )
      {
         grad(c, d) *= inv_J[d];
      });
   });
}

/**
 * @brief ApplyMapping for vector field gradients with dense std::array inv_J matrix.
 *
 * QuadMesh and general unstructured meshes use dense Jacobian matrices
 * represented as std::array<std::array<Real, Dim>, Dim>.
 * This overload handles vector field gradients with dense geometric mapping.
 *
 * Vector gradient tensor: SerialRecursiveArray<Real, NumComp, Dim>
 * Orientation: grad(component, direction) = ∂u_component/∂x_direction
 *
 * Geometric mapping: inv_J is dense matrix std::array<std::array<Real, Dim>, Dim>,
 * shared by all vector components.
 * All vector components share the same geometric mapping (reference-to-physical).
 *
 * Implementation: Applies component-wise matrix-vector multiplication.
 * For each component c:
 *   grad_phys[c][:] = inv_J * grad_ref[c][:]
 */
template < Integer Dim, Integer NumComp >
GENDIL_HOST_DEVICE
void ApplyMapping(
   const std::array< std::array< Real, Dim >, Dim > & inv_J,
   SerialRecursiveArray<Real, NumComp, Dim> & grad )
{
   // For each component, apply the same geometric mapping
   ConstexprLoop< NumComp >( [&]( auto c )
   {
      // Extract component gradient into temporary array
      std::array<Real, Dim> grad_c_ref;
      ConstexprLoop< Dim >( [&]( auto d )
      {
         grad_c_ref[d] = grad(c, d);
      });

      // Apply matrix-vector multiplication: grad_phys = inv_J * grad_ref
      std::array<Real, Dim> grad_c_phys;
      ConstexprLoop< Dim >( [&]( auto i )
      {
         Real sum = 0.0;
         ConstexprLoop< Dim >( [&]( auto j )
         {
            sum += inv_J[i][j] * grad_c_ref[j];
         });
         grad_c_phys[i] = sum;
      });

      // Write mapped gradient back
      ConstexprLoop< Dim >( [&]( auto d )
      {
         grad(c, d) = grad_c_phys[d];
      });
   });
}

}

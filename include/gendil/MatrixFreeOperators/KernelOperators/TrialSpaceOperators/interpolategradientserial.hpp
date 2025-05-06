// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TensorContraction/contractionhelper.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatevalues.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolategradientthreaded.hpp"

namespace gendil {

namespace details
{

template <
   Integer DiffDim,
   Integer ActiveDim,
   typename DofToQuad,
   typename InputTensor>
GENDIL_HOST_DEVICE
inline auto InterpolateGradientsImpl( const DofToQuad & quad_data, const InputTensor & u )
{
   constexpr Integer Rank = get_rank_v< InputTensor >;
   static_assert( ActiveDim < Rank );

   auto& B = std::get< ActiveDim >( quad_data );

   constexpr bool Gradient = DiffDim == ActiveDim;
   if constexpr ( ActiveDim+1 == Rank )
      return InterpContraction< Gradient, ActiveDim >( u, B );
   else
      return InterpContraction< Gradient, ActiveDim >( InterpolateGradientsImpl< DiffDim, ActiveDim+1 >( quad_data, u ), B );
}

template < Integer ActiveDim, typename InputTensor, typename OutputTensor, typename Op, size_t ... Is >
GENDIL_HOST_DEVICE
void InterpolateGradientSumFactSerial(
   InputTensor const & u, Op const & quad_data, OutputTensor & Gu )
{
   auto gu = InterpolateGradientsImpl< ActiveDim, 0 >( quad_data, u );
   using quad_shape = make_contraction_output_shape< Op >;
   Loop< quad_shape >([&]( auto ... indices )
   {
      Gu( indices..., ActiveDim ) = gu( indices... );
   });
}

template < Integer ActiveDim, typename InputTensor, typename OutputTensor, typename Op1D, size_t ... Is >
GENDIL_HOST_DEVICE
void GradContractionAtQPoints( InputTensor const & u, OutputTensor & Gu, Op1D const & B, std::index_sequence< Is ... > )
{
   constexpr Integer ND = Op1D::num_quads;

   Loop< get_tensor_size_v< Is, InputTensor > ... >(
      [&] ( auto ... indices_ )
      {
         auto indices = std::make_tuple( indices_ ... );
         const Integer q = std::get< ActiveDim >( indices );

         Real value = 0;

         auto& d = std::get< ActiveDim >( indices );
         for ( d = 0; d < ND; ++d )
         {
            const Real g = B.quad_gradients( q, d );
            const Real ud = u( std::get< Is >( indices ) ... );

            value += g * ud;
         }

         Gu( indices_ ..., ActiveDim ) = value;
      }
   );
}

template < Integer ActiveDim, typename InputTensor, typename OutputTensor, typename Op1D >
GENDIL_HOST_DEVICE
inline auto GradContractionAtQPoints( InputTensor const& u, OutputTensor & Gu, Op1D const & B )
{
   constexpr Integer Rank = InputTensor::rank;
   return GradContractionAtQPoints< ActiveDim >( u, Gu, B, std::make_index_sequence< Rank >{} );
}

} // namespace details

/**
 * @brief N-dimensional implementation of an operator interpolating values and gradient values at quadrature points
 * from the given degrees-of-freedom.
 * 
 * @tparam FiniteElementSpace The finite element space.
 * @tparam IntegrationRule The integration rule.
 * @tparam ElementDofToQuad A tuple of DofToQuad types.
 * @param element_quad_data The tuple containing data at quadrature point for each dimension.
 * @param u The input degrees-of-freedom.
 * @param Bu The output field values at quadrature points.
 * @param Gu The ouput field gradient values at quadrature points.
 * 
 * @note Assumes tensor finite element with tensor integration rule.
 */
template < typename InputTensor, typename OutputTensor, typename ElementQuadToQuad >
GENDIL_HOST_DEVICE
void InterpolateGradientSerial(
   const ElementQuadToQuad & element_quad_data,
   const InputTensor & u,
   OutputTensor & Gu )
{
   constexpr Integer Rank = std::tuple_size_v< ElementQuadToQuad >;
   auto&& uq = details::InterpolateValuesImpl<0>( element_quad_data, u );
   ConstexprLoop< Rank >( [&] ( auto ActiveDim )
   {
      auto & B1d = std::get< ActiveDim >( element_quad_data );
      using Op1D = std::decay_t< decltype( B1d ) >;
      if constexpr ( Op1D::num_quads >= Op1D::num_dofs)
      {
         details::GradContractionAtQPoints< ActiveDim >( uq, Gu, B1d );
      }
      else
      {
         details::InterpolateGradientSumFactSerial< ActiveDim >( u, element_quad_data, Gu );
      }
   } );
}

// /**
//  * @brief 6D implementation of an operator interpolating values and gradient values at quadrature points
//  * from the given degrees-of-freedom.
//  * 
//  * @tparam FiniteElementSpace The finite element space.
//  * @tparam IntegrationRule The integration rule.
//  * @tparam ElementDofToQuad A tuple of DofToQuad types.
//  * @param element_quad_data The tuple containing data at quadrature point for each dimension.
//  * @param u The input degrees-of-freedom.
//  * @param Bu The output field values at quadrature points.
//  * @param Gu The ouput field gradient values at quadrature points.
//  * 
//  * @note Assumes tensor finite element with tensor integration rule.
//  */
// template < typename FiniteElementSpace,
//            typename IntegrationRule,
//            typename ElementDofToQuad >
// GENDIL_HOST_DEVICE
// void InterpolateGradient(
//    const ElementDofToQuad & element_quad_data,
//    const ElementDoF< FiniteElementSpace > & u,
//    QuadraturePointValues< IntegrationRule, 6 > & Gu )
// {
//    // Assumes tensor elements
//    using orders = typename FiniteElementSpace::finite_element_type::shape_functions::orders;
//    constexpr Integer ndofs_dim_0 = get_num_dofs_of_dim_v< orders, 0 >::value;
//    constexpr Integer ndofs_dim_1 = get_num_dofs_of_dim_v< orders, 1 >::value;
//    constexpr Integer ndofs_dim_2 = get_num_dofs_of_dim_v< orders, 2 >::value;
//    constexpr Integer ndofs_dim_3 = get_num_dofs_of_dim_v< orders, 3 >::value;
//    constexpr Integer ndofs_dim_4 = get_num_dofs_of_dim_v< orders, 4 >::value;
//    constexpr Integer ndofs_dim_5 = get_num_dofs_of_dim_v< orders, 5 >::value;

//    using num_points_tensor = typename IntegrationRule::points::num_points_tensor;
//    constexpr Integer nquads_dim_0 = get_num_points_of_dim_v< num_points_tensor, 0 >::value;
//    constexpr Integer nquads_dim_1 = get_num_points_of_dim_v< num_points_tensor, 1 >::value;
//    constexpr Integer nquads_dim_2 = get_num_points_of_dim_v< num_points_tensor, 2 >::value;
//    constexpr Integer nquads_dim_3 = get_num_points_of_dim_v< num_points_tensor, 3 >::value;
//    constexpr Integer nquads_dim_4 = get_num_points_of_dim_v< num_points_tensor, 4 >::value;
//    constexpr Integer nquads_dim_5 = get_num_points_of_dim_v< num_points_tensor, 5 >::value;

//    Real B5u[ndofs_dim_0][ndofs_dim_1][ndofs_dim_2][ndofs_dim_3][ndofs_dim_4][nquads_dim_5];
//    Real G5u[ndofs_dim_0][ndofs_dim_1][ndofs_dim_2][ndofs_dim_3][ndofs_dim_4][nquads_dim_5];
//    for (LocalIndex d4 = 0; d4 < ndofs_dim_4; ++d4)
//    {
//       for (LocalIndex d3 = 0; d3 < ndofs_dim_3; ++d3)
//       {
//          for (LocalIndex d2 = 0; d2 < ndofs_dim_2; ++d2)
//          {
//             for (LocalIndex d1 = 0; d1 < ndofs_dim_1; ++d1)
//             {
//                for (LocalIndex d0 = 0; d0 < ndofs_dim_0; ++d0)
//                {
//                   for (LocalIndex q5 = 0; q5 < nquads_dim_5; ++q5)
//                   {
//                      Real value = 0.0;
//                      Real derivative = 0.0;
//                      for (LocalIndex d5 = 0; d5 < ndofs_dim_5; ++d5)
//                      {
//                         const Real b = std::get<5>( element_quad_data ).values(q5,d5);
//                         const Real g = std::get<5>( element_quad_data ).gradients(q5,d5);

//                         const Real dof = u(d0, d1, d2, d3, d4, d5);
//                         value += b * dof;
//                         derivative += g * dof;
//                      }
//                      B5u[d0][d1][d2][d3][d4][q5] = value;
//                      G5u[d0][d1][d2][d3][d4][q5] = derivative;
//                   }
//                }
//             }
//          }
//       }
//    }

//    Real B4B5u[ndofs_dim_0][ndofs_dim_1][ndofs_dim_2][ndofs_dim_3][nquads_dim_4][nquads_dim_5];
//    Real B4G5u[ndofs_dim_0][ndofs_dim_1][ndofs_dim_2][ndofs_dim_3][nquads_dim_4][nquads_dim_5];
//    Real G4B5u[ndofs_dim_0][ndofs_dim_1][ndofs_dim_2][ndofs_dim_3][nquads_dim_4][nquads_dim_5];
//    for (LocalIndex q5 = 0; q5 < nquads_dim_5; ++q5)
//    {
//       for (LocalIndex d3 = 0; d3 < ndofs_dim_3; ++d3)
//       {
//          for (LocalIndex d2 = 0; d2 < ndofs_dim_2; ++d2)
//          {
//             for (LocalIndex d1 = 0; d1 < ndofs_dim_1; ++d1)
//             {
//                for (LocalIndex d0 = 0; d0 < ndofs_dim_0; ++d0)
//                {
//                   for (LocalIndex q4 = 0; q4 < nquads_dim_4; q4++)
//                   {
//                      Real value = 0.0;
//                      Real derivative_5 = 0.0;
//                      Real derivative_4 = 0.0;
//                      for (LocalIndex d4 = 0; d4 < ndofs_dim_4; ++d4)
//                      {
//                         const Real b = std::get<4>( element_quad_data ).values(q4,d4);
//                         const Real g = std::get<4>( element_quad_data ).gradients(q4,d4);

//                         const Real b5u = B5u[d0][d1][d2][d3][d4][q5];
//                         const Real g5u = G5u[d0][d1][d2][d3][d4][q5];
//                         value += b * b5u;
//                         derivative_5 += b * g5u;
//                         derivative_4 += g * b5u;
//                      }
//                      B4B5u[d0][d1][d2][d3][q4][q5] = value;
//                      B4G5u[d0][d1][d2][d3][q4][q5] = derivative_5;
//                      G4B5u[d0][d1][d2][d3][q4][q5] = derivative_4;
//                   }
//                }
//             }
//          }
//       }
//    }

//    Real B3B4B5u[ndofs_dim_0][ndofs_dim_1][ndofs_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real B3B4G5u[ndofs_dim_0][ndofs_dim_1][ndofs_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real B3G4B5u[ndofs_dim_0][ndofs_dim_1][ndofs_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real G3B4B5u[ndofs_dim_0][ndofs_dim_1][ndofs_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    for (LocalIndex q4 = 0; q4 < nquads_dim_4; q4++)
//    {
//       for (LocalIndex q5 = 0; q5 < nquads_dim_5; ++q5)
//       {
//          for (LocalIndex d2 = 0; d2 < ndofs_dim_2; ++d2)
//          {
//             for (LocalIndex d1 = 0; d1 < ndofs_dim_1; ++d1)
//             {
//                for (LocalIndex d0 = 0; d0 < ndofs_dim_0; ++d0)
//                {
//                   for (LocalIndex q3 = 0; q3 < nquads_dim_3; q3++)
//                   {
//                      Real value = 0.0;
//                      Real derivative_5 = 0.0;
//                      Real derivative_4 = 0.0;
//                      Real derivative_3 = 0.0;
//                      for (LocalIndex d3 = 0; d3 < ndofs_dim_3; ++d3)
//                      {
//                         const Real b = std::get<3>( element_quad_data ).values(q3,d3);
//                         const Real g = std::get<3>( element_quad_data ).gradients(q3,d3);

//                         const Real bu = B4B5u[d0][d1][d2][d3][q4][q5];
//                         const Real g5u = B4G5u[d0][d1][d2][d3][q4][q5];
//                         const Real g4u = G4B5u[d0][d1][d2][d3][q4][q5];

//                         value += b * bu;
//                         derivative_5 += b * g5u;
//                         derivative_4 += b * g4u;
//                         derivative_3 += g * bu;
//                      }
//                      B3B4B5u[d0][d1][d2][q3][q4][q5] = value;
//                      B3B4G5u[d0][d1][d2][q3][q4][q5] = derivative_5;
//                      B3G4B5u[d0][d1][d2][q3][q4][q5] = derivative_4;
//                      G3B4B5u[d0][d1][d2][q3][q4][q5] = derivative_3;
//                   }
//                }
//             }
//          }
//       }
//    }

//    Real B2B3B4B5u[ndofs_dim_0][ndofs_dim_1][nquads_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real B2B3B4G5u[ndofs_dim_0][ndofs_dim_1][nquads_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real B2B3G4B5u[ndofs_dim_0][ndofs_dim_1][nquads_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real B2G3B4B5u[ndofs_dim_0][ndofs_dim_1][nquads_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real G2B3B4B5u[ndofs_dim_0][ndofs_dim_1][nquads_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    for (LocalIndex q3 = 0; q3 < nquads_dim_3; q3++)
//    {
//       for (LocalIndex q4 = 0; q4 < nquads_dim_4; q4++)
//       {
//          for (LocalIndex q5 = 0; q5 < nquads_dim_5; ++q5)
//          {
//             for (LocalIndex d1 = 0; d1 < ndofs_dim_1; ++d1)
//             {
//                for (LocalIndex d0 = 0; d0 < ndofs_dim_0; ++d0)
//                {
//                   for (LocalIndex q2 = 0; q2 < nquads_dim_2; ++q2)
//                   {
//                      Real value = 0.0;
//                      Real derivative_5 = 0.0;
//                      Real derivative_4 = 0.0;
//                      Real derivative_3 = 0.0;
//                      Real derivative_2 = 0.0;
//                      for (LocalIndex d2 = 0; d2 < ndofs_dim_2; ++d2)
//                      {
//                         const Real b = std::get<2>( element_quad_data ).values(q2,d2);
//                         const Real g = std::get<2>( element_quad_data ).gradients(q2,d2);

//                         const Real bu = B3B4B5u[d0][d1][d2][q3][q4][q5];
//                         const Real g5u = B3B4G5u[d0][d1][d2][q3][q4][q5];
//                         const Real g4u = B3G4B5u[d0][d1][d2][q3][q4][q5];
//                         const Real g3u = G3B4B5u[d0][d1][d2][q3][q4][q5];

//                         value += b * bu;
//                         derivative_5 += b * g5u;
//                         derivative_4 += b * g4u;
//                         derivative_3 += b * g3u;
//                         derivative_2 += g * bu;
//                      }
//                      B2B3B4B5u[d0][d1][q2][q3][q4][q5] = value;
//                      B2B3B4G5u[d0][d1][q2][q3][q4][q5] = derivative_5;
//                      B2B3G4B5u[d0][d1][q2][q3][q4][q5] = derivative_4;
//                      B2G3B4B5u[d0][d1][q2][q3][q4][q5] = derivative_3;
//                      G2B3B4B5u[d0][d1][q2][q3][q4][q5] = derivative_2;
//                   }
//                }
//             }
//          }
//       }
//    }

//    Real B1B2B3B4B5u[ndofs_dim_0][nquads_dim_1][nquads_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real B1B2B3B4G5u[ndofs_dim_0][nquads_dim_1][nquads_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real B1B2B3G4B5u[ndofs_dim_0][nquads_dim_1][nquads_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real B1B2G3B4B5u[ndofs_dim_0][nquads_dim_1][nquads_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real B1G2B3B4B5u[ndofs_dim_0][nquads_dim_1][nquads_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    Real G1B2B3B4B5u[ndofs_dim_0][nquads_dim_1][nquads_dim_2][nquads_dim_3][nquads_dim_4][nquads_dim_5];
//    for (LocalIndex q2 = 0; q2 < nquads_dim_2; ++q2)
//    {
//       for (LocalIndex q3 = 0; q3 < nquads_dim_3; q3++)
//       {
//          for (LocalIndex q4 = 0; q4 < nquads_dim_4; q4++)
//          {
//             for (LocalIndex q5 = 0; q5 < nquads_dim_5; ++q5)
//             {
//                for (LocalIndex d0 = 0; d0 < ndofs_dim_0; ++d0)
//                {
//                   for (LocalIndex q1 = 0; q1 < nquads_dim_1; ++q1)
//                   {
//                      Real value = 0.0;
//                      Real derivative_5 = 0.0;
//                      Real derivative_4 = 0.0;
//                      Real derivative_3 = 0.0;
//                      Real derivative_2 = 0.0;
//                      Real derivative_1 = 0.0;
//                      for (LocalIndex d1 = 0; d1 < ndofs_dim_1; ++d1)
//                      {
//                         const Real b = std::get<1>( element_quad_data ).values(q1,d1);
//                         const Real g = std::get<1>( element_quad_data ).gradients(q1,d1);

//                         const Real bu   = B2B3B4B5u[d0][d1][q2][q3][q4][q5];
//                         const Real g5u  = B2B3B4G5u[d0][d1][q2][q3][q4][q5];
//                         const Real g4u  = B2B3G4B5u[d0][d1][q2][q3][q4][q5];
//                         const Real g3u  = B2G3B4B5u[d0][d1][q2][q3][q4][q5];
//                         const Real g2u  = G2B3B4B5u[d0][d1][q2][q3][q4][q5];

//                         value += b * bu;
//                         derivative_5 += b * g5u;
//                         derivative_4 += b * g4u;
//                         derivative_3 += b * g3u;
//                         derivative_2 += b * g2u;
//                         derivative_1 += g * bu;
//                      }
//                      B1B2B3B4B5u[d0][q1][q2][q3][q4][q5] = value;
//                      B1B2B3B4G5u[d0][q1][q2][q3][q4][q5] = derivative_5;
//                      B1B2B3G4B5u[d0][q1][q2][q3][q4][q5] = derivative_4;
//                      B1B2G3B4B5u[d0][q1][q2][q3][q4][q5] = derivative_3;
//                      B1G2B3B4B5u[d0][q1][q2][q3][q4][q5] = derivative_2;
//                      G1B2B3B4B5u[d0][q1][q2][q3][q4][q5] = derivative_1;
//                   }
//                }
//             }
//          }
//       }
//    }

//    for (LocalIndex q1 = 0; q1 < nquads_dim_1; ++q1)
//    {
//       for (LocalIndex q2 = 0; q2 < nquads_dim_2; ++q2)
//       {
//          for (LocalIndex q3 = 0; q3 < nquads_dim_3; q3++)
//          {
//             for (LocalIndex q4 = 0; q4 < nquads_dim_4; q4++)
//             {
//                for (LocalIndex q5 = 0; q5 < nquads_dim_5; ++q5)
//                {
//                   for (LocalIndex q0 = 0; q0 < nquads_dim_0; ++q0)
//                   {
//                      Real derivative_5 = 0.0;
//                      Real derivative_4 = 0.0;
//                      Real derivative_3 = 0.0;
//                      Real derivative_2 = 0.0;
//                      Real derivative_1 = 0.0;
//                      Real derivative_0 = 0.0;
//                      for (LocalIndex d0 = 0; d0 < ndofs_dim_0; ++d0)
//                      {
//                         const Real bx = std::get<0>( element_quad_data ).values(q0,d0);
//                         const Real gx = std::get<0>( element_quad_data ).gradients(q0,d0);

//                         const Real bu   = B1B2B3B4B5u[d0][q1][q2][q3][q4][q5];
//                         const Real g5u  = B1B2B3B4G5u[d0][q1][q2][q3][q4][q5];
//                         const Real g4u  = B1B2B3G4B5u[d0][q1][q2][q3][q4][q5];
//                         const Real g3u  = B1B2G3B4B5u[d0][q1][q2][q3][q4][q5];
//                         const Real g2u  = B1G2B3B4B5u[d0][q1][q2][q3][q4][q5];
//                         const Real g1u  = G1B2B3B4B5u[d0][q1][q2][q3][q4][q5];

//                         derivative_5 += bx * g5u;
//                         derivative_4 += bx * g4u;
//                         derivative_3 += bx * g3u;
//                         derivative_2 += bx * g2u;
//                         derivative_1 += bx * g1u;
//                         derivative_0 += gx * bu;
//                      }
//                      Gu(q0, q1, q2, q3, q4, q5, 5) = derivative_5;
//                      Gu(q0, q1, q2, q3, q4, q5, 4) = derivative_4;
//                      Gu(q0, q1, q2, q3, q4, q5, 3) = derivative_3;
//                      Gu(q0, q1, q2, q3, q4, q5, 2) = derivative_2;
//                      Gu(q0, q1, q2, q3, q4, q5, 1) = derivative_1;
//                      Gu(q0, q1, q2, q3, q4, q5, 0) = derivative_0;
//                   }
//                }
//             }
//          }
//       }
//    }
// }

// /**
//  * @brief 3D implementation of an operator interpolating values and gradient values at quadrature points
//  * from the given degrees-of-freedom.
//  * 
//  * @tparam FiniteElementSpace The finite element space.
//  * @tparam IntegrationRule The integration rule.
//  * @tparam ElementDofToQuad A tuple of DofToQuad types.
//  * @param element_quad_data The tuple containing data at quadrature point for each dimension.
//  * @param u The input degrees-of-freedom.
//  * @param Bu The output field values at quadrature points.
//  * @param Gu The ouput field gradient values at quadrature points.
//  * 
//  * @note Assumes tensor finite element with tensor integration rule.
//  */
// template < typename FiniteElementSpace,
//            typename IntegrationRule,
//            typename ElementDofToQuad >
// GENDIL_HOST_DEVICE
// void InterpolateGradient(
//    const ElementDofToQuad & element_quad_data,
//    const ElementDoF< FiniteElementSpace > & u,
//    QuadraturePointValues< IntegrationRule, 3 > & Gu )
// {
//    // Assumes tensor elements
//    using orders = typename FiniteElementSpace::finite_element_type::shape_functions::orders;
//    constexpr Integer ndofs_dim_0 = get_num_dofs_of_dim_v< orders, 0 >::value;
//    constexpr Integer ndofs_dim_1 = get_num_dofs_of_dim_v< orders, 1 >::value;
//    constexpr Integer ndofs_dim_2 = get_num_dofs_of_dim_v< orders, 2 >::value;

//    using num_points_tensor = typename IntegrationRule::points::num_points_tensor;
//    constexpr Integer nquads_dim_0 = get_num_points_of_dim_v< num_points_tensor, 0 >::value;
//    constexpr Integer nquads_dim_1 = get_num_points_of_dim_v< num_points_tensor, 1 >::value;
//    constexpr Integer nquads_dim_2 = get_num_points_of_dim_v< num_points_tensor, 2 >::value;

//    Real B2u[ndofs_dim_0][ndofs_dim_1][nquads_dim_2];
//    Real G2u[ndofs_dim_0][ndofs_dim_1][nquads_dim_2];
//    for (LocalIndex d1 = 0; d1 < ndofs_dim_1; ++d1)
//    {
//       for (LocalIndex d0 = 0; d0 < ndofs_dim_0; ++d0)
//       {
//          for (LocalIndex q2 = 0; q2 < nquads_dim_2; ++q2)
//          {
//             Real value = 0.0;
//             Real derivative_2 = 0.0;
//             for (LocalIndex d2 = 0; d2 < ndofs_dim_2; ++d2)
//             {
//                const Real b = std::get<2>( element_quad_data ).values(q2,d2);
//                const Real g = std::get<2>( element_quad_data ).gradients(q2,d2);

//                const Real bu = u(d0, d1, d2);

//                value += b * bu;
//                derivative_2 += g * bu;
//             }
//             B2u[d0][d1][q2] = value;
//             G2u[d0][d1][q2] = derivative_2;
//          }
//       }
//    }

//    Real B1B2u[ndofs_dim_0][nquads_dim_1][nquads_dim_2];
//    Real B1G2u[ndofs_dim_0][nquads_dim_1][nquads_dim_2];
//    Real G1B2u[ndofs_dim_0][nquads_dim_1][nquads_dim_2];
//    for (LocalIndex q2 = 0; q2 < nquads_dim_2; ++q2)
//    {
//       for (LocalIndex d0 = 0; d0 < ndofs_dim_0; ++d0)
//       {
//          for (LocalIndex q1 = 0; q1 < nquads_dim_1; ++q1)
//          {
//             Real value = 0.0;
//             Real derivative_2 = 0.0;
//             Real derivative_1 = 0.0;
//             for (LocalIndex d1 = 0; d1 < ndofs_dim_1; ++d1)
//             {
//                const Real b = std::get<1>( element_quad_data ).values(q1,d1);
//                const Real g = std::get<1>( element_quad_data ).gradients(q1,d1);

//                const Real bu   = B2u[d0][d1][q2];
//                const Real g2u  = G2u[d0][d1][q2];

//                value += b * bu;
//                derivative_2 += b * g2u;
//                derivative_1 += g * bu;
//             }
//             B1B2u[d0][q1][q2] = value;
//             B1G2u[d0][q1][q2] = derivative_2;
//             G1B2u[d0][q1][q2] = derivative_1;
//          }
//       }
//    }

//    for (LocalIndex q1 = 0; q1 < nquads_dim_1; ++q1)
//    {
//       for (LocalIndex q2 = 0; q2 < nquads_dim_2; ++q2)
//       {
//          for (LocalIndex q0 = 0; q0 < nquads_dim_0; ++q0)
//          {
//             Real derivative_2 = 0.0;
//             Real derivative_1 = 0.0;
//             Real derivative_0 = 0.0;
//             for (LocalIndex d0 = 0; d0 < ndofs_dim_0; ++d0)
//             {
//                const Real b = std::get<0>( element_quad_data ).values(q0,d0);
//                const Real g = std::get<0>( element_quad_data ).gradients(q0,d0);

//                const Real bu   = B1B2u[d0][q1][q2];
//                const Real g2u  = B1G2u[d0][q1][q2];
//                const Real g1u  = G1B2u[d0][q1][q2];

//                derivative_2 += b * g2u;
//                derivative_1 += b * g1u;
//                derivative_0 += g * bu;
//             }
//             Gu(q0, q1, q2, 2) = derivative_2;
//             Gu(q0, q1, q2, 1) = derivative_1;
//             Gu(q0, q1, q2, 0) = derivative_0;
//          }
//       }
//    }
// }

}
